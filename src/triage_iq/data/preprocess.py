"""Preprocessing pipeline: raw JSON -> cleaned parquet."""

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 10_000

# Per-repo label normalization rules.
# Keys: owner/repo format.
# Values: {facet: pattern} where pattern is either:
#   - a regex string with one capture group (extracts group 1 as value)
#   - a list of known label strings (label matched case-insensitively; raw label used as value)
LABEL_FACET_PATTERNS: dict[str, dict[str, str | list[str]]] = {
    "kubernetes/kubernetes": {
        "component": r"^area/(.+)$",
        "type": r"^kind/(.+)$",
        "priority": r"^priority/(.+)$",
    },
    "tensorflow/tensorflow": {
        "component": r"^comp:(.+)$",
        "type": r"^type:(.+)$",
        "priority": r"^stat:(.+)$",
    },
    "pytorch/pytorch": {
        "component": r"^module:\s*(.+)$",
        "type": r"^(bug|feature|enhancement)$",
    },
    "microsoft/vscode": {
        # vscode uses flat labels; enumerate known component labels (case-insensitive match)
        "component": [
            # Core editor
            "editor", "editor-core", "editor-folding", "editor-find", "editor-rendering",
            "editor-input", "editor-commands", "suggest",
            # Workbench
            "workbench", "workbench-editors", "workbench-tabs", "workbench-os-integration",
            "file-explorer", "search", "settings", "keybindings", "themes",
            # Languages
            "javascript", "typescript", "html", "css", "css-less-scss", "json",
            "markdown", "snippets", "languages", "languages-basic", "php",
            # Platform / runtime
            "terminal", "tasks", "debug", "extensions", "api", "ipc", "build",
            "install-update", "electron", "performance",
            # Version control
            "git", "scm",
            # Other capabilities
            "accessibility", "telemetry", "error-telemetry", "config",
            "release-notes", "settings-sync", "ux", "l10n-platform",
            "file-watcher", "VIM",
            # Additional editor sub-components
            "editor-bracket-matching", "editor-autoclosing", "editor-multicursor",
            "editor-wrapping", "keyboard-layout", "suggest", "emmet",
            # Additional workbench sub-components
            "workbench-electron", "workbench-multiroot", "workbench-editor-grid",
            "workbench-diagnostics", "workbench-run-as-admin", "output",
            "error-list", "menus", "layout",
            # Other
            "file-io", "vscode-build", "nodejs", "perf", "php", "electron",
        ],
        "type": ["bug", "feature-request", "enhancement", "documentation", "question"],
    },
    "apache/airflow": {
        "component": r"^area:(.+)$",
        "type": r"^kind:(.+)$",
    },
}


def load_raw_issues(repo: str, cache_dir: str = "data/raw") -> pd.DataFrame:
    """Load all JSON files from data/raw/{repo}/ into a DataFrame."""
    raw_dir = Path(cache_dir) / repo
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    records = []
    files = sorted(raw_dir.glob("*.json"), key=lambda p: int(p.stem))
    logger.info("Loading %d issue files from %s", len(files), raw_dir)

    for path in files:
        try:
            issue = json.loads(path.read_text(encoding="utf-8"))
            records.append(_extract_fields(issue))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True)
    df["resolution_hours"] = (
        (df["closed_at"] - df["created_at"]).dt.total_seconds() / 3600
    )
    logger.info("Loaded %d issues. Closed: %d", len(df), df["closed_at"].notna().sum())
    return df


def _extract_fields(issue: dict) -> dict:
    labels_raw = [lbl["name"] for lbl in issue.get("labels", []) if isinstance(lbl, dict)]
    comments_data = issue.get("comments_data", [])
    body = issue.get("body") or ""
    body_clean, code_blocks = clean_text(body)
    assignees = issue.get("assignees") or []

    return {
        "id": issue.get("id"),
        "number": issue.get("number"),
        "title": (issue.get("title") or "").strip(),
        "body_clean": body_clean,
        "code_blocks": code_blocks,
        "labels_raw": labels_raw,
        "state": issue.get("state"),
        "created_at": issue.get("created_at"),
        "closed_at": issue.get("closed_at"),
        "author": (issue.get("user") or {}).get("login"),
        "num_comments": issue.get("comments", len(comments_data)),
        "num_assignees": len(assignees),
    }


def clean_text(text: str) -> tuple[str, str]:
    """Clean issue body text. Returns (clean_body, extracted_code_blocks)."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    code_blocks = "\n\n".join(re.findall(r"```[\s\S]*?```", text))
    code_blocks = code_blocks[:5000]

    text = re.sub(r"```[\s\S]*?```", "[CODE_BLOCK]", text)
    text = re.sub(r"`[^`\n]+`", "[INLINE_CODE]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    text = text[:MAX_BODY_CHARS]

    return text, code_blocks


def _repo_key(repo: str) -> str:
    """Convert filesystem repo name (owner_repo) to owner/repo for pattern lookup."""
    return repo.replace("_", "/", 1)


def normalize_labels(repo: str, labels: list[str]) -> dict[str, str | None]:
    """Map repo-specific labels to standardized facets: component, type, priority."""
    facets: dict[str, str | None] = {"component": None, "type": None, "priority": None}
    patterns = LABEL_FACET_PATTERNS.get(_repo_key(repo), {})

    for facet, pattern in patterns.items():
        if facet not in facets:
            continue
        if isinstance(pattern, list):
            known = {p.lower() for p in pattern}
            for label in labels:
                if label.lower() in known:
                    facets[facet] = label
                    break
        else:
            for label in labels:
                m = re.match(pattern, label, re.IGNORECASE)
                if m:
                    facets[facet] = m.group(1) if m.lastindex else m.group(0)
                    break

    return facets


def build_processed_df(df: pd.DataFrame, repo: str) -> pd.DataFrame:
    """Add normalized label facets to the DataFrame."""
    facets = df["labels_raw"].apply(lambda lbls: normalize_labels(repo, lbls))
    df = df.copy()
    df["component"] = facets.apply(lambda f: f["component"])
    df["type"] = facets.apply(lambda f: f["type"])
    df["priority"] = facets.apply(lambda f: f["priority"])
    return df


def save_processed(df: pd.DataFrame, repo: str, out_dir: str = "data/processed") -> Path:
    """Save processed DataFrame to parquet."""
    out_path = Path(out_dir) / f"issues_{repo}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "id", "number", "title", "body_clean", "code_blocks",
        "labels_raw", "component", "type", "priority",
        "state", "created_at", "closed_at", "resolution_hours",
        "author", "num_comments", "num_assignees",
    ]
    existing = [c for c in cols if c in df.columns]
    df[existing].to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return out_path
