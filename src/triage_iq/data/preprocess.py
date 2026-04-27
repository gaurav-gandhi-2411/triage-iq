"""Preprocessing pipeline: raw JSON -> cleaned parquet."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Max chars to keep per issue body (covers 99%+ of issues)
MAX_BODY_CHARS = 10_000

# Per-repo label prefix mappings to standardized facets
LABEL_FACET_PATTERNS: Dict[str, Dict[str, str]] = {
    "kubernetes_kubernetes": {
        r"^area/": "component",
        r"^kind/": "type",
        r"^priority/": "priority",
        r"^sig/": "sig",
    },
    "microsoft_vscode": {
        r"^feature-request$": "type:feature",
        r"^bug$": "type:bug",
    },
    "tensorflow_tensorflow": {
        r"^comp:": "component",
        r"^type:": "type",
        r"^stat:": "status",
    },
    "pytorch_pytorch": {
        r"^module:": "component",
        r"^triaged$": "status:triaged",
    },
    "apache_airflow": {
        r"^area:": "component",
        r"^kind:": "type",
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


def _extract_fields(issue: Dict) -> Dict:
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
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Extract code blocks before stripping them
    code_blocks = "\n\n".join(re.findall(r"```[\s\S]*?```", text))
    code_blocks = code_blocks[:5000]  # cap code blocks separately

    # Remove fenced code blocks from main text
    text = re.sub(r"```[\s\S]*?```", "[CODE_BLOCK]", text)
    text = re.sub(r"`[^`\n]+`", "[INLINE_CODE]", text)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    # Truncate to max chars
    text = text[:MAX_BODY_CHARS]

    return text, code_blocks


def normalize_labels(repo: str, labels: List[str]) -> Dict[str, Optional[str]]:
    """Map repo-specific labels to standardized facets: component, type, priority."""
    facets: Dict[str, Optional[str]] = {"component": None, "type": None, "priority": None}
    patterns = LABEL_FACET_PATTERNS.get(repo, {})

    for label in labels:
        for pattern, facet in patterns.items():
            if re.search(pattern, label, re.IGNORECASE):
                if ":" in facet:
                    key, val = facet.split(":", 1)
                    if facets.get(key) is None:
                        facets[key] = val
                else:
                    if facets.get(facet) is None:
                        # Use label value (strip prefix)
                        prefix_match = re.match(pattern, label, re.IGNORECASE)
                        value = label[prefix_match.end():] if prefix_match else label
                        facets[facet] = value
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
