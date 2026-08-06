"""LEVER 1 measurement (spec.md item pending an ADR number): quantify how much of each
issue's text the corpus-side truncation in similar_issues.py::_build_text() is discarding.

_build_text() truncates body to `max_body=512` CHARACTERS before encoding. BGE-base-en-v1.5's
actual limit is 512 TOKENS (~2000+ chars for English prose). This script measures, per repo,
the token length of the FULL title+body text (as the model's own tokenizer would count it)
against the token length of what's currently being encoded (title + body[:512 chars]) --
using the real BGE tokenizer, not a char/4 approximation.

Reads:  data/processed/issues_{repo}.parquet
Writes: reports/lever1_truncation_measurement.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
REPORTS = Path("reports")
REPOS = ["kubernetes_kubernetes", "microsoft_vscode"]
MAX_BODY_CHARS_CURRENT = 512
MODEL_MAX_TOKENS = (
    512  # BGE-base-en-v1.5's actual sequence limit, verified via model.max_seq_length
)


def build_text(title: str, body: str, max_body_chars: int | None) -> str:
    t = (title or "").strip()
    b = (body or "").strip()
    if max_body_chars is not None:
        b = b[:max_body_chars]
    return f"{t}. {b}"


def token_lengths(tokenizer, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Count tokens WITHOUT truncation (so we can see the true, uncapped length),
    including special tokens the way the model would actually see them."""
    lengths = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=False, padding=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.array(lengths)


def pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def main() -> None:
    log.info("Loading BAAI/bge-base-en-v1.5 tokenizer...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    tokenizer = model.tokenizer
    log.info("model.max_seq_length = %d", model.max_seq_length)

    out: dict = {"model_max_seq_length": model.max_seq_length, "per_repo": {}}

    for repo in REPOS:
        df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
        titles = df["title"].fillna("").astype(str).tolist()
        bodies = df["body_clean"].fillna("").astype(str).tolist()

        full_texts = [build_text(t, b, None) for t, b in zip(titles, bodies, strict=True)]
        current_texts = [
            build_text(t, b, MAX_BODY_CHARS_CURRENT) for t, b in zip(titles, bodies, strict=True)
        ]

        log.info("[%s] tokenizing %d FULL (untruncated) texts...", repo, len(full_texts))
        full_lens = token_lengths(tokenizer, full_texts)
        log.info(
            "[%s] tokenizing %d CURRENT (512-char-truncated) texts...", repo, len(current_texts)
        )
        current_lens = token_lengths(tokenizer, current_texts)

        # How much of the FULL token budget is actually being encoded today?
        # (current_lens is capped by both the 512-char cut AND, if that still exceeds 512
        # tokens, by the model's own truncation at encode time -- but 512 chars is ~replaced
        # by roughly a hundred-odd tokens for English text, so we expect current_lens well
        # under 512 almost always; measuring directly rather than assuming.)
        frac_content_dropped_today = 1.0 - (
            current_lens.astype(float) / np.maximum(full_lens.astype(float), 1.0)
        )
        # What fraction of issues would STILL be truncated even at the model's true 512-token
        # limit (i.e. genuinely too long for BGE regardless of the character-truncation bug)?
        frac_still_over_model_limit = float(np.mean(full_lens > MODEL_MAX_TOKENS))
        # What fraction of issues are currently truncated at all (full > current, i.e. body
        # actually exceeded 512 chars)?
        frac_truncated_today = float(np.mean(full_lens > current_lens))

        repo_result = {
            "n_issues": len(df),
            "full_text_tokens": {
                "p50": pct(full_lens, 50),
                "p90": pct(full_lens, 90),
                "p95": pct(full_lens, 95),
                "mean": float(full_lens.mean()),
                "max": int(full_lens.max()),
            },
            "current_encoded_tokens_512char_cut": {
                "p50": pct(current_lens, 50),
                "p90": pct(current_lens, 90),
                "p95": pct(current_lens, 95),
                "mean": float(current_lens.mean()),
                "max": int(current_lens.max()),
            },
            "frac_issues_truncated_today": frac_truncated_today,
            "frac_issues_still_over_512_tokens_even_untruncated": frac_still_over_model_limit,
            "mean_frac_content_dropped_today_among_truncated": float(
                frac_content_dropped_today[full_lens > current_lens].mean()
            )
            if frac_truncated_today > 0
            else 0.0,
            "median_tokens_dropped_today": float(
                np.median((full_lens - current_lens)[full_lens > current_lens])
            )
            if frac_truncated_today > 0
            else 0.0,
        }
        out["per_repo"][repo] = repo_result

        log.info(
            "[%s] FULL tokens p50=%.0f p90=%.0f p95=%.0f | CURRENT (512-char) tokens p50=%.0f "
            "p90=%.0f p95=%.0f | truncated today=%.1f%% | still-over-512-tok even if fixed=%.1f%%",
            repo,
            repo_result["full_text_tokens"]["p50"],
            repo_result["full_text_tokens"]["p90"],
            repo_result["full_text_tokens"]["p95"],
            repo_result["current_encoded_tokens_512char_cut"]["p50"],
            repo_result["current_encoded_tokens_512char_cut"]["p90"],
            repo_result["current_encoded_tokens_512char_cut"]["p95"],
            frac_truncated_today * 100,
            frac_still_over_model_limit * 100,
        )

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "lever1_truncation_measurement.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    log.info("Wrote reports/lever1_truncation_measurement.json")


if __name__ == "__main__":
    main()
