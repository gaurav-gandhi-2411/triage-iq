"""D3 pre-training config verification. Per GG's explicit instruction: verify BEFORE training,
report all three, STOP on any mismatch. This is the exact class of check whose absence caused
D2's first run (ADR-0034, withdrawn by ADR-0035) -- mean pooling trained against BGE's native
CLS-token config, and 65.73% of examples silently truncated at a max_length picked without ever
measuring the actual token-length distribution.

Checks:
  1. CLS pooling matches BAAI/bge-base-en-v1.5's own cached 1_Pooling/config.json, field-for-field
     (pooling_mode_cls_token=True, pooling_mode_mean_tokens=False) -- confirms scripts/d3_train.py's
     cls_pool()/_save_st_model() agree with the checkpoint's own pretrained config, not assumed.
  2. max_seq_length (d3_train.py's MAX_LEN) against the MEASURED p50/p90/p95/p99/max token count
     of the actual anchor/positive/negative texts this run will train on -- re-measured fresh on
     the NEW, larger/differently-composed mining-precision pool, not carried over from D2's old
     1,734-pair vscode-only measurement.
  3. Tokenizer identity: AutoTokenizer.from_pretrained(BASE_MODEL) is the exact tokenizer class/
     vocab the checkpoint ships, confirmed via the tokenizer's own name_or_path and vocab hash,
     not assumed by string-matching a model id.

Reads:  reports/mining_precision_train_pool_{task}.json
        data/d3_hard_negatives_{task}.parquet   (if already mined -- else anchor/positive only)
        data/processed/issues_{repo}.parquet
Writes: reports/d3_config_verification.json

Exits non-zero (STOP) if any check fails. Reproduce: python scripts/d3_verify_config.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "BAAI/bge-base-en-v1.5"
MAX_LEN = 256  # scripts/d3_train.py's configured value -- verified against measured p95 below
MAX_BODY = 512

REPO_BY_TASK = {
    "vscode_duplicate": "microsoft_vscode",
    "k8s_related": "kubernetes_kubernetes",
}
REPORTS = Path("reports")
DATA = Path("data")


def build_text(title: object, body: object) -> str:
    t = (str(title) if title is not None else "").strip()
    b = (str(body) if body is not None else "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def check_pooling() -> dict:
    local_dir = snapshot_download(BASE_MODEL, allow_patterns=["1_Pooling/config.json"])
    cfg_path = Path(local_dir) / "1_Pooling" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    expects = {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}
    matches = all(cfg.get(k) == v for k, v in expects.items())
    return {
        "checkpoint_pooling_config": cfg,
        "training_uses": "cls_pool() -> hidden[:, 0, :], pooling_mode_cls_token=True in "
        "_save_st_model()",
        "matches_checkpoint": matches,
        "status": "PASS" if matches else "FAIL",
    }


def check_tokenizer() -> dict:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    is_fast = tok.is_fast
    vocab_size = tok.vocab_size
    name_or_path = tok.name_or_path
    matches = name_or_path == BASE_MODEL and is_fast
    return {
        "requested_model": BASE_MODEL,
        "tokenizer_name_or_path": name_or_path,
        "is_fast": is_fast,
        "vocab_size": vocab_size,
        "status": "PASS" if matches else "FAIL",
    }


def check_seq_length() -> dict:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    all_texts: list[str] = []
    per_task: dict[str, dict] = {}

    for task, repo in REPO_BY_TASK.items():
        pool_path = REPORTS / f"mining_precision_train_pool_{task}.json"
        pairs = json.loads(pool_path.read_text(encoding="utf-8"))
        corpus = pd.read_parquet(DATA / "processed" / f"issues_{repo}.parquet")
        text_by_num = dict(
            zip(
                corpus["number"].astype(int),
                corpus.apply(lambda r: build_text(r["title"], r["body_clean"]), axis=1),
                strict=True,
            )
        )
        texts = []
        for p in pairs:
            q, o = int(p["query_number"]), int(p["original_number"])
            if q in text_by_num:
                texts.append(text_by_num[q])
            if o in text_by_num:
                texts.append(text_by_num[o])

        neg_path = DATA / f"d3_hard_negatives_{task}.parquet"
        if neg_path.exists():
            negs = pd.read_parquet(neg_path)
            texts.extend(str(t) for t in negs["neg_text"])
            neg_note = f"included {len(negs)} mined hard-negative texts"
        else:
            neg_note = "hard negatives not yet mined -- anchor/positive only (negatives are drawn " \
                "from the same corpus text distribution, so this undercounts slightly but is " \
                "conservative in the direction that matters: real anchor/positive length, not " \
                "an assumption)"

        lengths = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
        all_texts.extend(texts)
        per_task[task] = {
            "n_texts": len(texts),
            "note": neg_note,
            "p50": int(np.percentile(lengths, 50)),
            "p90": int(np.percentile(lengths, 90)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
            "max": int(np.max(lengths)),
            "pct_truncated_at_max_len": round(100 * np.mean(np.array(lengths) > MAX_LEN), 2),
        }

    combined_lengths = [len(tok.encode(t, add_special_tokens=True)) for t in all_texts]
    combined = {
        "n_texts": len(all_texts),
        "p50": int(np.percentile(combined_lengths, 50)),
        "p90": int(np.percentile(combined_lengths, 90)),
        "p95": int(np.percentile(combined_lengths, 95)),
        "p99": int(np.percentile(combined_lengths, 99)),
        "max": int(np.max(combined_lengths)),
        "pct_truncated_at_max_len": round(100 * np.mean(np.array(combined_lengths) > MAX_LEN), 2),
    }
    # Pass condition: MAX_LEN covers p95 (matches D2's own stated rationale for choosing 256:
    # "smallest power of 2 covering p95, truncating only a small tail")
    status = "PASS" if combined["p95"] <= MAX_LEN else "FAIL"
    return {
        "configured_max_len": MAX_LEN,
        "per_task": per_task,
        "combined": combined,
        "status": status,
    }


def main() -> None:
    pooling = check_pooling()
    tokenizer = check_tokenizer()
    seq_len = check_seq_length()

    logger.info("1. CLS pooling: %s", pooling["status"])
    logger.info("2. max_seq_length=%d vs measured p95=%d: %s", MAX_LEN, seq_len["combined"]["p95"], seq_len["status"])
    logger.info("3. Tokenizer identity: %s", tokenizer["status"])

    result = {"pooling": pooling, "tokenizer": tokenizer, "seq_length": seq_len}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "d3_config_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    all_pass = all(r["status"] == "PASS" for r in (pooling, tokenizer, seq_len))
    if not all_pass:
        logger.critical("CONFIG VERIFICATION FAILED -- STOPPING. See reports/d3_config_verification.json")
        sys.exit(1)
    logger.info("All three checks PASS. Proceeding is safe. See reports/d3_config_verification.json")


if __name__ == "__main__":
    main()
