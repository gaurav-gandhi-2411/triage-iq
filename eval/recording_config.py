"""One config-aware TriageAssistant builder for the recorder, run_eval and checkpoint validation.

Until 2026-10-29 each of the three paths built its own assistant and ignored the production loader
flags (TRIAGE_VSCODE_CLASSIFIER, TRIAGE_RESOLUTION_EVIDENCE, TRIAGE_K8S_INDEX). D92.6 needs the single
final-configuration re-record to capture them, so the choice lives in a committed
RECORDING_CONFIG.json next to the set being recorded and every path builds through build_assistant().

File absent = today's behaviour, exactly. File present = validated strictly (all three keys, known
values, nothing else); anything else raises RecordingConfigError, never a silent default. The loader's
own helpers are reused, so the eval opens the same artifacts, with the same manifest checks, as /triage.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_FILENAME = "RECORDING_CONFIG.json"
_ALLOWED: dict[str, tuple[Any, ...]] = {
    "vscode_classifier": ("served", "family_v2"),
    "k8s_index": ("served", "v2"),
    "resolution_evidence": (False, True),
}
_VSCODE_SLUG = "microsoft_vscode"


class RecordingConfigError(ValueError):
    """RECORDING_CONFIG.json is unreadable or invalid; the run must stop."""


@dataclass(frozen=True)
class RecordingConfig:
    """Which production-loader variants a recording uses. Defaults = the served artifacts."""

    vscode_classifier: str = "served"
    k8s_index: str = "served"
    resolution_evidence: bool = False


def parse_config(raw: object, source: str = CONFIG_FILENAME) -> RecordingConfig:
    """Validate a decoded JSON value strictly (exact key set, known values, real booleans)."""
    if not isinstance(raw, dict):
        raise RecordingConfigError(f"{source}: expected a JSON object, got {type(raw).__name__}")
    unknown, missing = sorted(set(raw) - set(_ALLOWED)), sorted(set(_ALLOWED) - set(raw))
    if unknown or missing:
        raise RecordingConfigError(f"{source}: unknown keys {unknown}, missing keys {missing}")
    for key, allowed in _ALLOWED.items():
        # type(...) check: json true/false only, so 1/0 (== True/False in Python) are rejected
        if not any(raw[key] == a and type(raw[key]) is type(a) for a in allowed):
            raise RecordingConfigError(
                f"{source}: {key}={raw[key]!r}, expected one of {list(allowed)}"
            )
    return RecordingConfig(**raw)


def config_search_dirs(root: Path, cassette_dir: Path | None = None) -> list[Path]:
    """Directories searched for RECORDING_CONFIG.json: the recording set dir (TRIAGEIQ_RECORDING_SET_DIR),
    the cassette dir (default: <set dir>/cassettes, else eval/cassettes) and, when the cassette dir
    sits inside a set (its parent holds eval_set.jsonl), that set dir. The last one lets a replay
    with no env var (run_eval, qp5, checkpoint validation) find a set's own config."""
    set_dir = os.environ.get("TRIAGEIQ_RECORDING_SET_DIR")
    base = Path(set_dir) if set_dir else None
    cass = cassette_dir or (base / "cassettes" if base else root / "eval" / "cassettes")
    dirs: list[Path | None] = [base, cass]
    # eval/ holds the 64-set's eval_set.jsonl but never a config: skip it so that set resolves
    # exactly as before.
    if cass.parent != root / "eval" and (cass.parent / "eval_set.jsonl").is_file():
        dirs.append(cass.parent)
    return [d for d in dirs if d is not None]


def load_recording_config(root: Path, cassette_dir: Path | None = None) -> RecordingConfig:
    """The config for this run; defaults when no file exists, an error when two files exist
    (ambiguous) or the one found is invalid."""
    found = [d / CONFIG_FILENAME for d in config_search_dirs(root, cassette_dir)]
    # resolve(): one directory reached by two spellings must not read as two files.
    found = [p for p in {p.resolve(): p for p in found}.values() if p.is_file()]
    if len(found) > 1:
        raise RecordingConfigError(
            f"ambiguous: {CONFIG_FILENAME} exists in both {found[0]} and {found[1]}"
        )
    if not found:
        return RecordingConfig()
    try:
        raw = json.loads(found[0].read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecordingConfigError(f"{found[0]}: unreadable ({exc})") from exc
    return parse_config(raw, str(found[0]))


def issue_to_row(issue: dict) -> pd.Series:
    """The row the pipeline is called with. body_raw (the uncleaned request body) is carried only
    when the eval-set issue has it, as app.py sets it on /triage; absent = the previous row."""
    fields: dict[str, Any] = {
        "title": issue["title"],
        "body_clean": issue["body"],
        "number": issue["number"],
        "created_at": (
            pd.Timestamp(issue["created_at"])
            if issue.get("created_at")
            else pd.Timestamp("now", tz="UTC")
        ),
    }
    if issue.get("body_raw") is not None:
        fields["body_raw"] = issue["body_raw"]
    return pd.Series(fields)


def build_assistant(
    repo: str,
    slug: str,
    *,
    models_dir: Path,
    processed_dir: Path,
    detector: Any,
    config: RecordingConfig,
    **assistant_kwargs: Any,
) -> dict[str, Any]:
    """Load classifier, predictor and train_df for one repo and build its TriageAssistant.

    config.vscode_classifier == "family_v2" opens the retrained vscode bundle through the loader's
    manifest-checked load_family_v2_classifier; config.resolution_evidence loads the k8s evidence
    parquet through the loader's helper and fails closed if it is not usable. k8s_index only changes
    the artifact fingerprint here: eval similar issues come from the frozen retriever (`detector`).
    """
    from triage_iq.api.loader import _RESOLUTION_EVIDENCE_SLUGS, _load_resolution_evidence
    from triage_iq.models.component_classifier import load_classifier, load_family_v2_classifier
    from triage_iq.models.resolution import ResolutionTimePredictor
    from triage_iq.models.triage import TriageAssistant

    if config.vscode_classifier == "family_v2" and slug == _VSCODE_SLUG:
        classifier: Any = load_family_v2_classifier(models_dir)
    else:
        # load_classifier() dispatches on the pkl's model_kind marker (ADR-0036).
        classifier = load_classifier(models_dir, slug)
    predictor = ResolutionTimePredictor.load(str(models_dir / f"resolution_predictor_{slug}.pkl"))
    train_df = pd.read_parquet(processed_dir / f"{slug}_temporal_train.parquet")
    if config.resolution_evidence and slug in _RESOLUTION_EVIDENCE_SLUGS:
        evidence = _load_resolution_evidence(models_dir, slug)
        if evidence is None:  # the loader's helper degrades to None; a recording must not
            raise RecordingConfigError(
                f"resolution_evidence=true but no usable evidence for {slug}"
            )
        assistant_kwargs["resolution_evidence"] = evidence
    assistant = TriageAssistant(
        repo=repo,
        classifier=classifier,
        detector=detector,
        predictor=predictor,
        train_df=train_df,
        **assistant_kwargs,
    )
    return {
        "classifier": classifier,
        "predictor": predictor,
        "train_df": train_df,
        "assistant": assistant,
    }
