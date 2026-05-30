"""Extract related-issue pairs for supervised similar-issue retrieval evaluation.

Strategy (in priority order):
1. vscode issues with '*duplicate' label + explicit #N reference in body  (high confidence)
2. Any body pattern: 'duplicate of #N', 'see #N', 'closes #N' in issues   (medium confidence)
3. Title-similarity pairs: TF-IDF cosine >= threshold on title text        (fallback)

Pairs are (query_id, original_id) where the relationship is captured by a body
reference or title similarity — see ADR-0008 for task framing.

Output: data/gold_related.parquet
Schema: repo, query_number, original_number, query_title, original_title,
        query_body, original_body, source (label|body_ref|title_sim), confidence
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("data/gold_related.parquet")

# Patterns for extracting referenced issue numbers from body text
BODY_PATTERNS = [
    r"[Dd]uplicate[sd]?(?: of)? #?(\d+)",
    r"[Dd]up(?:licate)? of #?(\d+)",
    r"[Ss]ame as #(\d+)",
    r"[Cc]losing as dup(?:licate)? of #?(\d+)",
]
# Weaker signals (closes/fixes suggest "related", not strict duplicate, but useful)
RELATED_PATTERNS = [
    r"[Ss]ee(?: also)? #(\d+)",
    r"[Cc]loses? #(\d+)",
    r"[Ff]ixes? #(\d+)",
]

TITLE_SIM_THRESHOLD = 0.45  # cosine on TF-IDF bigrams
TITLE_SIM_MAX_PAIRS = 300   # cap per repo to prevent explosion


def _extract_ref(text: str, patterns: list[str]) -> int | None:
    for pat in patterns:
        m = re.search(pat, str(text), re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _has_duplicate_label(labels_raw) -> bool:
    return "duplicate" in str(labels_raw).lower()


def extract_for_repo(repo: str) -> list[dict]:
    path = PROCESSED_DIR / f"issues_{repo}.parquet"
    if not path.exists():
        log.warning("No data for %s", repo)
        return []

    df = pd.read_parquet(path)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    num_to_idx = {int(n): i for i, n in enumerate(df["number"])}
    pairs = []

    # ── Strategy 1+2: Explicit body reference (high confidence) ────────────
    # Use both vscode duplicate-labeled issues AND any issue with body ref
    all_refs = []
    for i, row in df.iterrows():
        body = str(row.get("body_clean", ""))
        title = str(row.get("title", ""))
        combined = title + " " + body

        # Try duplicate patterns first (stronger)
        ref = _extract_ref(combined, BODY_PATTERNS)
        source, confidence = "body_ref", "high"

        # Fall back to weak related patterns
        if ref is None:
            ref = _extract_ref(combined, RELATED_PATTERNS)
            source, confidence = "body_related", "medium"

        if ref is not None and ref in num_to_idx and ref != int(row["number"]):
            # Only count as duplicate if query issue is NEWER than original
            orig_row = df.iloc[num_to_idx[ref]]
            if orig_row["created_at"] <= row["created_at"]:
                # If vscode *duplicate label, bump confidence
                if _has_duplicate_label(row.get("labels_raw")):
                    confidence = "high"
                all_refs.append({
                    "repo": repo,
                    "query_number": int(row["number"]),
                    "original_number": int(ref),
                    "query_title": row["title"],
                    "original_title": orig_row["title"],
                    "query_body": str(row.get("body_clean", "")),
                    "original_body": str(orig_row.get("body_clean", "")),
                    "source": source,
                    "confidence": confidence,
                })
    # Deduplicate
    seen = set()
    for p in all_refs:
        key = (p["query_number"], p["original_number"])
        if key not in seen:
            seen.add(key)
            pairs.append(p)
    log.info("[%s] explicit body refs: %d pairs", repo, len(pairs))

    # ── Strategy 3: vscode *duplicate label without body ref ────────────────
    labeled_dup_with_ref = {p["query_number"] for p in pairs}
    if "labels_raw" in df.columns:
        dup_labeled = df[
            df["labels_raw"].apply(_has_duplicate_label) &
            ~df["number"].isin(labeled_dup_with_ref)
        ]
        if len(dup_labeled) > 0:
            log.info("[%s] %d duplicate-labeled issues without body ref — will use title similarity", repo, len(dup_labeled))

    # ── Strategy 4: Title similarity fallback ──────────────────────────────
    titles = (df["title"].fillna("") + " " + df["body_clean"].fillna("").str[:200]).tolist()
    numbers = df["number"].tolist()
    dates = df["created_at"].tolist()

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, min_df=2, stop_words="english")
    try:
        tfidf_mat = vec.fit_transform(titles)
    except Exception as e:
        log.warning("TF-IDF failed: %s", e)
        return pairs

    existing_pairs = seen.copy()
    title_sim_count = 0

    # To avoid O(N^2) pairwise, sample candidate pairs per issue
    n = len(df)
    batch_size = 200  # process in chunks
    sim_pairs_buf = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_sims = cosine_similarity(tfidf_mat[start:end], tfidf_mat)
        for local_i, global_i in enumerate(range(start, end)):
            sim_row = batch_sims[local_i]
            # Find similar issues that predate global_i
            for global_j in np.where(sim_row >= TITLE_SIM_THRESHOLD)[0]:
                if global_j >= global_i:
                    continue  # only take older issues as "original"
                if sim_row[global_j] >= 0.9999:
                    continue  # skip near-identical (same issue)
                q_num = int(numbers[global_i])
                o_num = int(numbers[global_j])
                key = (q_num, o_num)
                if key in existing_pairs:
                    continue
                sim_pairs_buf.append((sim_row[global_j], q_num, o_num, global_i, global_j))

    # Sort by similarity desc, keep top N
    sim_pairs_buf.sort(key=lambda x: -x[0])
    for sim_score, q_num, o_num, gi, gj in sim_pairs_buf:
        if title_sim_count >= TITLE_SIM_MAX_PAIRS:
            break
        key = (q_num, o_num)
        if key in existing_pairs:
            continue
        existing_pairs.add(key)
        title_sim_count += 1

        qr = df.iloc[gi]
        or_ = df.iloc[gj]
        pairs.append({
            "repo": repo,
            "query_number": q_num,
            "original_number": o_num,
            "query_title": qr["title"],
            "original_title": or_["title"],
            "query_body": str(qr.get("body_clean", "")),
            "original_body": str(or_.get("body_clean", "")),
            "source": "title_sim",
            "confidence": "medium" if sim_score >= 0.6 else "low",
        })

    log.info("[%s] title sim added: %d pairs", repo, title_sim_count)
    log.info("[%s] TOTAL: %d pairs", repo, len(pairs))
    return pairs


def main() -> None:
    all_pairs = []
    for repo in ["microsoft_vscode", "kubernetes_kubernetes"]:
        all_pairs.extend(extract_for_repo(repo))

    if not all_pairs:
        log.error("No pairs found!")
        return

    df_out = pd.DataFrame(all_pairs)

    # Ensure bodies are non-empty
    df_out = df_out[
        (df_out["query_body"].str.strip().str.len() > 10) &
        (df_out["original_body"].str.strip().str.len() > 10)
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUTPUT_PATH, index=False)

    log.info("Saved %d pairs to %s", len(df_out), OUTPUT_PATH)
    log.info("\nBreakdown:")
    for (repo, source), g in df_out.groupby(["repo", "source"]):
        log.info("  %-40s %s: %d", repo, source, len(g))
    log.info("\nConfidence distribution:")
    log.info("%s", df_out["confidence"].value_counts().to_string())


if __name__ == "__main__":
    main()
