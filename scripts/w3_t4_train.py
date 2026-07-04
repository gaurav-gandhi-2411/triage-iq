"""T4: Fine-tune BAAI/bge-base-en-v1.5 bi-encoder on gold similar-issue pairs.

Uses a direct HuggingFace training loop (not sentence-transformers model.fit())
to avoid the per-step Python overhead in ST 2.7.0 (which causes ~30s/step).

Trains two variants and picks the winner on val R@5:
  (X) combined  — one model trained on k8s + vscode pairs
  (Y) per-repo  — separate model per repo

MultipleNegativesRankingLoss with hard negatives:
  scores = cosine_sim(anchor, [all_positives | all_negatives]) / temp
  loss = cross_entropy(scores, diagonal_labels)

Saved in sentence-transformers format (mean pooling + L2-normalise) so T5 can
load with SentenceTransformer(model_dir) and rebuild FAISS.

Outputs:
  data/models/bge_finetuned_combined/   — combined model
  data/models/bge_finetuned_k8s/        — k8s-only model
  data/models/bge_finetuned_vsc/        — vscode-only model
  reports/w3_t4_val_results.json        — val R@5 comparison table
"""
from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
BASE_MODEL = "BAAI/bge-base-en-v1.5"
MAX_LEN = 128          # token budget — covers ~95% of issue titles
MAX_BODY = 512         # char budget for text builder
BATCH_SIZE = 16        # per-step batch (triplets); 3 forward passes of 16 each
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.10
TEMPERATURE = 0.05     # MNRL temperature (standard for bge)
EVAL_EVERY = 50        # training steps between val R@5 snapshots
RETRIEVAL_K = 5        # recall@K evaluated on val

REPOS = ["kubernetes_kubernetes", "microsoft_vscode"]
MODEL_DIRS = {
    "combined": "data/models/bge_finetuned_combined",
    "k8s": "data/models/bge_finetuned_k8s",
    "vsc": "data/models/bge_finetuned_vsc",
}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def build_text(title: str | None, body: str | None) -> str:
    t = (title or "").strip()
    b = (body or "").strip()[:MAX_BODY]
    return f"{t}. {b}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripletDataset(torch.utils.data.Dataset):
    """Pre-tokenised triplets for fast DataLoader access."""

    def __init__(
        self,
        anchors: list[str],
        positives: list[str],
        negatives: list[str],
        tokenizer: AutoTokenizer,
    ) -> None:
        assert len(anchors) == len(positives) == len(negatives)
        self.tokenizer = tokenizer
        logger.info("Pre-tokenising %d triplets…", len(anchors))
        self.enc_a = tokenizer(anchors, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        self.enc_p = tokenizer(positives, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        self.enc_n = tokenizer(negatives, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        logger.info("Tokenisation complete.")

    def __len__(self) -> int:
        return self.enc_a["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "a_input_ids": self.enc_a["input_ids"][idx],
            "a_attention_mask": self.enc_a["attention_mask"][idx],
            "p_input_ids": self.enc_p["input_ids"][idx],
            "p_attention_mask": self.enc_p["attention_mask"][idx],
            "n_input_ids": self.enc_n["input_ids"][idx],
            "n_attention_mask": self.enc_n["attention_mask"][idx],
        }


def load_triplets(
    split_df: pd.DataFrame,
    gold: pd.DataFrame,
    hard_negs: pd.DataFrame,
    split: str,
    repos: list[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (anchors, positives, negatives) for the requested split."""
    rows = split_df[split_df["split"] == split]
    if repos:
        rows = rows[rows["repo"].isin(repos)]

    gold_idx = gold.set_index(["repo", "query_number", "original_number"])
    hn_idx = (
        hard_negs
        .sort_values(["repo", "query_number", "original_number", "neg_rank"])
        .set_index(["repo", "query_number", "original_number"])
    )

    anchors, positives, negatives = [], [], []
    for _, row in rows.iterrows():
        key = (row["repo"], int(row["query_number"]), int(row["original_number"]))
        if key not in gold_idx.index:
            continue
        g = gold_idx.loc[key]
        if not isinstance(g, pd.Series):
            g = g.iloc[0]
        anchor = build_text(g["query_title"], g["query_body"])
        positive = build_text(g["original_title"], g["original_body"])

        if key not in hn_idx.index:
            continue  # skip pairs with no hard negatives
        hn = hn_idx.loc[key]
        if isinstance(hn, pd.Series):
            neg_text = str(hn.get("neg_text", ""))
        else:
            neg_text = str(hn.iloc[0].get("neg_text", ""))
        if not neg_text.strip():
            continue

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(neg_text)

    logger.info(
        "Loaded %d triplets for split=%s repos=%s",
        len(anchors), split, repos or "all",
    )
    return anchors, positives, negatives


# ---------------------------------------------------------------------------
# Mean-pool helper
# ---------------------------------------------------------------------------

def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pool over token dimension."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Val evaluator (full-corpus FAISS for the val subset)
# ---------------------------------------------------------------------------

def eval_val_r5(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    split_df: pd.DataFrame,
    gold: pd.DataFrame,
    repos: list[str],
    device: torch.device,
) -> float:
    """Compute R@5 on val pairs using in-memory brute-force search."""
    val_rows = split_df[(split_df["split"] == "val") & (split_df["repo"].isin(repos))]
    if val_rows.empty:
        return 0.0

    gold_idx = gold.set_index(["repo", "query_number", "original_number"])

    # Build query + positive pairs for val
    query_texts: list[str] = []
    positive_texts: list[str] = []
    for _, row in val_rows.iterrows():
        key = (row["repo"], int(row["query_number"]), int(row["original_number"]))
        if key not in gold_idx.index:
            continue
        g = gold_idx.loc[key]
        if not isinstance(g, pd.Series):
            g = g.iloc[0]
        query_texts.append(build_text(g["query_title"], g["query_body"]))
        positive_texts.append(build_text(g["original_title"], g["original_body"]))

    if not query_texts:
        return 0.0

    # All unique texts form the retrieval corpus
    all_texts = list(set(query_texts + positive_texts))
    text_to_idx = {t: i for i, t in enumerate(all_texts)}

    model.eval()
    with torch.no_grad():
        embs = _encode_texts(model, tokenizer, all_texts, device, batch_size=128)

    # Brute-force cosine similarity
    embs = F.normalize(embs, p=2, dim=1)  # (N, D)
    q_indices = [text_to_idx[t] for t in query_texts]
    p_indices = [text_to_idx[t] for t in positive_texts]

    q_embs = embs[q_indices]  # (Q, D)
    scores = q_embs @ embs.T  # (Q, N)

    hits = 0
    for qi, pi in enumerate(p_indices):
        top5 = scores[qi].topk(6).indices.tolist()  # 6 to skip self
        top5 = [i for i in top5 if i != q_indices[qi]][:5]
        if pi in top5:
            hits += 1

    r5 = hits / len(query_texts)
    model.train()
    return r5


def _encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LEN,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        emb = mean_pool(out, enc["attention_mask"])
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    train_dataset: TripletDataset,
    split_df: pd.DataFrame,
    gold: pd.DataFrame,
    repos: list[str],
    out_dir: str,
) -> float:
    """Fine-tune and return best val R@5."""
    torch.cuda.empty_cache()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=== Training: %s | %d triplets | device=%s | out=%s ===",
                name, len(train_dataset), device, out_dir)

    model = AutoModel.from_pretrained(BASE_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    total_steps = len(loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    logger.info("Total steps: %d  warmup: %d", total_steps, warmup_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    best_r5 = 0.0
    best_step = 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in loader:
            # Move to device
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            p_ids = batch["p_input_ids"].to(device)
            p_mask = batch["p_attention_mask"].to(device)
            n_ids = batch["n_input_ids"].to(device)
            n_mask = batch["n_attention_mask"].to(device)

            # Encode
            emb_a = F.normalize(mean_pool(model(a_ids, a_mask).last_hidden_state, a_mask), p=2, dim=1)
            emb_p = F.normalize(mean_pool(model(p_ids, p_mask).last_hidden_state, p_mask), p=2, dim=1)
            emb_n = F.normalize(mean_pool(model(n_ids, n_mask).last_hidden_state, n_mask), p=2, dim=1)

            # MNRL loss: anchor vs [in-batch positives | explicit negatives]
            candidates = torch.cat([emb_p, emb_n], dim=0)  # (2B, D)
            scores = (emb_a @ candidates.T) / TEMPERATURE   # (B, 2B)
            labels = torch.arange(len(emb_a), device=device)  # diagonal = positive
            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % EVAL_EVERY == 0:
                r5 = eval_val_r5(model, tokenizer, split_df, gold, repos, device)
                logger.info(
                    "[%s] step=%d  epoch=%d  loss=%.4f  val_R@5=%.4f  (best=%.4f @ step %d)",
                    name, global_step, epoch, epoch_loss / epoch_steps, r5, best_r5, best_step,
                )
                if r5 > best_r5:
                    best_r5 = r5
                    best_step = global_step
                    _save_st_model(model, tokenizer, out_dir)
                model.train()

        avg_loss = epoch_loss / max(1, epoch_steps)
        r5 = eval_val_r5(model, tokenizer, split_df, gold, repos, device)
        logger.info(
            "[%s] END epoch %d  avg_loss=%.4f  val_R@5=%.4f  (best=%.4f @ step %d)",
            name, epoch, avg_loss, r5, best_r5, best_step,
        )
        if r5 > best_r5:
            best_r5 = r5
            best_step = global_step
            _save_st_model(model, tokenizer, out_dir)

    if best_r5 == 0.0:
        # Save final model even if val R@5 never improved
        _save_st_model(model, tokenizer, out_dir)
        logger.warning("[%s] val R@5 was always 0 — saved final model", name)

    logger.info("[%s] Training complete. Best val R@5 = %.4f @ step %d", name, best_r5, best_step)
    return best_r5


def _save_st_model(model: AutoModel, tokenizer: AutoTokenizer, out_dir: str) -> None:
    """Save fine-tuned weights as a SentenceTransformer (mean-pool + L2-norm)."""
    p = Path(out_dir)
    # Save HF model + tokenizer in the transformer sub-directory
    transformer_dir = p / "0_Transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(transformer_dir))
    tokenizer.save_pretrained(str(transformer_dir))

    # Build SentenceTransformer wrapper around the saved weights.
    # Move to CPU before creating the ST wrapper to avoid GPU memory accumulation across
    # sequential train_model() calls (ST Transformer loads weights onto GPU by default).
    word_embedding_model = st_models.Transformer(str(transformer_dir), max_seq_length=MAX_LEN)
    word_embedding_model = word_embedding_model.cpu()
    pooling_model = st_models.Pooling(word_embedding_model.get_word_embedding_dimension())
    normalize_model = st_models.Normalize()
    st = SentenceTransformer(modules=[word_embedding_model, pooling_model, normalize_model])
    st.save(str(p))
    # Explicit cleanup to release GPU/CPU memory before returning to the training loop
    del st, word_embedding_model, pooling_model, normalize_model
    torch.cuda.empty_cache()
    logger.info("Saved SentenceTransformer model -> %s", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)

    gold = pd.read_parquet("data/gold_related.parquet")
    split_df = pd.read_parquet("data/w3_split.parquet")
    hard_negs = pd.read_parquet("data/w3_hard_negatives.parquet")

    logger.info("Data: gold=%d  split=%d  hard_negs=%d", len(gold), len(split_df), len(hard_negs))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    results: dict[str, float] = {}

    # --- Variant X: combined ---
    a, p, n = load_triplets(split_df, gold, hard_negs, "train", repos=REPOS)
    ds_combined = TripletDataset(a, p, n, tokenizer)
    r5_combined = train_model("combined", ds_combined, split_df, gold, REPOS, MODEL_DIRS["combined"])
    results["combined"] = r5_combined

    # --- Variant Y: per-repo ---
    repo_keys = {"kubernetes_kubernetes": "k8s", "microsoft_vscode": "vsc"}
    for repo, key in repo_keys.items():
        a, p, n = load_triplets(split_df, gold, hard_negs, "train", repos=[repo])
        ds_repo = TripletDataset(a, p, n, tokenizer)
        r5_repo = train_model(key, ds_repo, split_df, gold, [repo], MODEL_DIRS[key])
        results[key] = r5_repo

    # --- Decision ---
    logger.info("=== Val R@5 summary ===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        logger.info("  %-20s  R@5=%.4f", k, v)

    winner = max(results, key=lambda k: results[k])
    logger.info("Winner: %s (R@5=%.4f)", winner, results[winner])

    if winner == "combined":
        winner_dirs = {"combined": MODEL_DIRS["combined"]}
    else:
        winner_dirs = {
            "k8s": MODEL_DIRS["k8s"],
            "vsc": MODEL_DIRS["vsc"],
        }

    report = {
        "val_r5_by_variant": results,
        "winner": winner,
        "winner_model_dirs": winner_dirs,
        "hyperparams": {
            "base_model": BASE_MODEL,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "warmup_ratio": WARMUP_RATIO,
            "temperature": TEMPERATURE,
            "max_len": MAX_LEN,
            "seed": SEED,
        },
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/w3_t4_val_results.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report → reports/w3_t4_val_results.json")


if __name__ == "__main__":
    main()
