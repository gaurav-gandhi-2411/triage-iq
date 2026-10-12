"""Expand the vscode eval pool with held-out issues that carry a real repo label (zero Groq).

Design (v2, 2026-10-12, after owner review of v1; consensus-only items dropped 2026-10-13):
  * Ground truth is the issue's REAL repo component label. The held-out temporal window (temporal val + test,
    minus every train set, the gold set, the eval set and earlier candidates) is searched for labelled issues;
    only those are sampled. Local LLM judges are a cross-check on the labels, never a source of labels: they
    match the real label only ~41% of the time.
  * The judge vocabulary equals the SERVED classifier's label space: 47 classes, read from the production
    classifier artifact after its sha256 is checked against data/models/MANIFEST.sha256.
  * Judges are calibrated on labelled issues two ways: EASY (the component name appears in the title) and HARD
    (it does not). Both accuracies are reported; the pre-registered inclusion rule is easy accuracy >= 0.80.

Subcommands (resumable, deterministic, seed 42):
  vocab             47-class vocabulary + provenance
  sample            the labelled issues (stratified over three resolution buckets by the curation script's own function)
  calibrate-sample  easy and hard calibration sets from labelled issues outside the sample
  run               three judges from three families, BLIND (title + body + allowed list only), temperature 0,
                    JSON-schema-constrained; VRAM-gated
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "vscode_gold_expansion_2026-10-12"
PRIMARY = Path("C:/Users/gaura/ml-projects/triage-iq")  # checkout that holds the gitignored data/ files
SEED = 42
N_SAMPLE = 64
N_CALIB_EASY = 30
N_CALIB_HARD = 40
CALIB_PASS = 0.80  # pre-registered: judge counts if easy-set accuracy >= this (hard-set accuracy is reported)
BODY_CHARS = 2000
OLLAMA = "http://127.0.0.1:11434"
# Three families (Alibaba Qwen, Meta Llama, Google Gemma), all installed locally.
JUDGES = [
    {"model": "qwen3:8b", "family": "Alibaba Qwen", "think_false": True},
    {"model": "llama3.1:8b", "family": "Meta Llama", "think_false": False},
    {"model": "gemma2:9b", "family": "Google Gemma", "think_false": False},
]
NEED_FREE_MIB = 6800  # an 8B model at num_ctx 4096 plus headroom on the 8 GB card
BUCKET = lambda d: "<7d" if d < 7 else ("7-30d" if d < 30 else ">30d")  # noqa: E731


def _curate_module(processed_root: Path):  # noqa: ANN202
    """scripts/10_curate_triage_gold.py with its ROOT pointed at <processed_root> (data/processed inside)."""
    spec = importlib.util.spec_from_file_location("curate_gold", ROOT / "scripts" / "10_curate_triage_gold.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    mod.ROOT = processed_root
    return mod


def _used_numbers() -> set[int]:
    """vscode issue numbers already in the gold set, the eval set, earlier candidates or the SHIPPED train."""
    used: set[int] = set()
    for p in (PRIMARY / "data" / "gold_triage_plans.parquet", PRIMARY / "data" / "gold_expansion_candidates.parquet"):
        if p.exists():
            d = pd.read_parquet(p)
            used |= set(d.loc[d["repo"] == "microsoft/vscode", "number"].astype(int))
    shipped = PRIMARY / "data" / "processed" / "microsoft_vscode_temporal_train.parquet"
    if shipped.exists():  # 4,923 rows the served models trained on (smaller than the regenerated split)
        used |= set(pd.read_parquet(shipped, columns=["number"])["number"].astype(int))
    es = ROOT / "eval" / "eval_set.jsonl"
    if es.exists():
        for line in es.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["repo"] == "microsoft/vscode":
                    used.add(int(r["number"]))
    return used


def _vocab() -> list[str]:
    return json.loads((OUT_DIR / "vocabulary.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------- vocabulary
def cmd_vocab(a: argparse.Namespace) -> None:
    models = Path(a.models_dir)
    pkl = models / "component_classifier_microsoft_vscode.pkl"
    got = hashlib.sha256(pkl.read_bytes()).hexdigest()
    want = next(
        ln.split()[0]
        for ln in Path(a.manifest).read_text(encoding="utf-8").splitlines()
        if ln.endswith("component_classifier_microsoft_vscode.pkl")
    )
    if got != want:
        raise SystemExit(f"classifier sha256 {got[:16]} != MANIFEST {want[:16]}: this is not the served classifier")
    from triage_iq.models.component_classifier import load_classifier

    c = load_classifier(models, "microsoft_vscode").classes_
    classes = sorted(str(x) for x in (c() if callable(c) else c))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vocabulary.json").write_text(json.dumps(classes, indent=1), encoding="utf-8")
    (OUT_DIR / "vocabulary_provenance.json").write_text(
        json.dumps(
            {
                "n_classes": len(classes),
                "classifier_sha256": got,
                "matches_manifest": True,
                "source": "gs://triageiq-prod-260812-models/models/component_classifier_microsoft_vscode.pkl (ADR-0057, 47 classes)",
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(len(classes), "classes; sha256 matches MANIFEST")


# ------------------------------------------------------------------------------------------- sampling
def _row(r: pd.Series, extra: dict) -> dict:
    return {
        "number": int(r["number"]),
        "title": r["title"],
        "body": (r["body_clean"] or "")[:BODY_CHARS],
        "created_at": str(r["created_at"]),
        **extra,
    }


def cmd_sample(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(a.processed_root)
    mod = _curate_module(root)
    vocab = set(_vocab())
    proc = root / "data" / "processed"
    window = pd.concat(
        [pd.read_parquet(proc / f"microsoft_vscode_temporal_{s}.parquet") for s in ("val", "test")], ignore_index=True
    ).drop_duplicates("number")
    n_window = len(window)
    train = mod.load_train_numbers("microsoft_vscode", "temporal_train") | mod.load_train_numbers(
        "microsoft_vscode", "classifier_train"
    )
    used = _used_numbers() | {int(x) for x in a.extra_exclude}
    window = window[~window["number"].astype(int).isin(train | used)].copy()
    n_after = len(window)
    window = window[window["resolution_hours"].notna()]
    labelled_any = window[window["component"].notna()]
    labelled = labelled_any[labelled_any["component"].isin(vocab)]
    unlabelled = window[window["component"].isna()].copy()
    counts = {
        "temporal_val_test_issues": n_window,
        "after_excluding_train_gold_eval_shipped_train": n_after,
        "closed_with_resolution": len(window),
        "with_repo_component_label": len(labelled_any),
        "label_in_served_47_class_space": len(labelled),
        "label_outside_the_47_classes": len(labelled_any) - len(labelled),
        "without_label": len(unlabelled),
    }
    # Real repo labels only (GG decision 2026-10-13): a consensus label from judges that match the real
    # label ~41% of the time is more likely wrong than right, so unlabelled issues are never sampled.
    mod.ISSUES_PER_REPO = N_SAMPLE
    chosen = mod.stratified_sample(labelled, N_SAMPLE // 3 + 1, SEED) if len(labelled) >= N_SAMPLE else labelled
    rows = [
        _row(r, {"resolution_days": round(float(r.resolution_hours) / 24, 4),
                 "bucket": BUCKET(float(r.resolution_hours) / 24),
                 "repo_label_component": r.component, "label_source": "repo"})
        for _, r in chosen.iterrows()
    ]
    (OUT_DIR / "sample.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    counts.update(
        sampled=len(rows),
        sampled_with_real_label=len(rows),
        buckets=dict(Counter(r["bucket"] for r in rows)),
    )
    (OUT_DIR / "sample_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(json.dumps(counts, indent=1))


def cmd_calibrate_sample(a: argparse.Namespace) -> None:
    root = Path(a.processed_root)
    d = pd.read_parquet(root / "data" / "processed" / "issues_microsoft_vscode.parquet")
    vocab = _vocab()
    sample_nums = {json.loads(ln)["number"] for ln in (OUT_DIR / "sample.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()}
    d = d[d["component"].isin(set(vocab)) & ~d["number"].astype(int).isin(sample_nums)].copy()
    d = d[d["title"].str.len() >= 15]

    def in_title(r: pd.Series) -> bool:
        t = str(r.title).lower()
        c = str(r.component).lower()
        return c in t or c.replace("-", " ") in t

    d["in_title"] = d.apply(in_title, axis=1)
    rng = np.random.default_rng(SEED)

    def pick(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        frame = frame.sample(frac=1.0, random_state=SEED).groupby("component").head(3)  # <= 3 per component
        return frame.iloc[rng.permutation(len(frame))[:n]]

    easy = pick(d[d["in_title"]], N_CALIB_EASY)
    hard = pick(d[~d["in_title"]], N_CALIB_HARD)
    for name, frame in (("easy", easy), ("hard", hard)):
        rows = [_row(r, {"repo_label_component": r.component}) for _, r in frame.iterrows()]
        (OUT_DIR / f"calibration_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(name, len(rows), "items over", frame["component"].nunique(), "components")


# ---------------------------------------------------------------------------------------------- Ollama
def _free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    return int(out.stdout.strip().splitlines()[0])


def _ollama_loaded() -> list[str]:
    with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as r:
        return [m["name"] for m in json.load(r).get("models", [])]


def vram_gate(model: str, poll_s: float = 15.0, max_wait_s: float = 6 * 3600) -> str:
    """Wait for a VRAM-safe moment; returns "exclusive" or "shared".

    exclusive: nothing is loaded and the card has headroom for one 8B model, so we load ours (keep_alive 0:
               it is unloaded right after the answer).
    shared:    another session already holds THIS SAME model, which costs no extra VRAM; we reuse it and send
               neither keep_alive nor num_ctx, so we neither evict nor reload the other session's runner.
    Any other model loaded -> keep waiting (a second 5-6 GB model would not fit the 8 GB card)."""
    t0 = time.monotonic()
    ok = 0
    while True:
        try:
            loaded, free = _ollama_loaded(), _free_mib()
        except Exception as exc:  # noqa: BLE001
            loaded, free = ["?"], 0
            print("vram gate: probe failed:", exc, flush=True)
        if loaded == [model]:
            ok += 1
            if ok >= 2:
                return "shared"
        elif not loaded and free >= NEED_FREE_MIB:
            ok += 1
            if ok >= 2:
                return "exclusive"
        else:
            ok = 0
        if time.monotonic() - t0 > max_wait_s:
            raise TimeoutError(f"VRAM never freed (loaded={loaded}, free={free} MiB)")
        time.sleep(poll_s if ok == 0 else 3)


SYSTEM = (
    "You triage GitHub issues for the microsoft/vscode repository. Pick the single best component label for the "
    "issue from the allowed list. Answer with JSON only."
)


def ask(judge: dict, vocab: list[str], title: str, body: str, mode: str = "exclusive") -> tuple[str, str]:
    schema = {
        "type": "object",
        "properties": {"component": {"type": "string", "enum": vocab}},
        "required": ["component"],
    }
    user = (
        f"Allowed components: {', '.join(vocab)}\n\nIssue title: {title}\n\nIssue body:\n{body}\n\n"
        'Return {"component": "<one allowed component>"}.'
    )
    payload = {
        "model": judge["model"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "seed": SEED, "num_predict": 40},
    }
    if mode == "exclusive":
        payload["keep_alive"] = 0
        payload["options"]["num_ctx"] = 4096
    if judge["think_false"]:
        payload["think"] = False
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = json.load(r)["message"]["content"]
    try:
        return json.loads(raw)["component"], raw
    except Exception:  # noqa: BLE001
        return "__invalid__", raw


def _file(kind: str, model: str) -> Path:
    return OUT_DIR / f"{kind}_{model.replace(':', '_').replace('/', '_')}.jsonl"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


KINDS = {"calibrate_easy": "calibration_easy.jsonl", "calibrate_hard": "calibration_hard.jsonl", "label": "sample.jsonl"}


def _run_judge(kind: str, items: list[dict], judge: dict, vocab: list[str]) -> None:
    f = _file(kind, judge["model"])
    done = {r["number"] for r in _load(f)} if f.exists() else set()
    todo = [it for it in items if it["number"] not in done]
    print(f"[{kind}] {judge['model']}: {len(done)} done, {len(todo)} to go", flush=True)
    for it in todo:
        mode = vram_gate(judge["model"])
        label, raw = ask(judge, vocab, it["title"], it["body"], mode)
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"number": it["number"], "label": label, "raw": raw, "mode": mode}) + "\n")


def cmd_run(a: argparse.Namespace) -> None:
    vocab = _vocab()
    items = {k: _load(OUT_DIR / v) for k, v in KINDS.items()}
    # One model at a time (easy calibration, hard calibration, labels), so a model another session already
    # holds on the GPU is used to the end while it is shared.
    for judge in JUDGES:
        for kind in KINDS if a.kind == "all" else (a.kind,):
            _run_judge(kind, items[kind], judge, vocab)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vocab")
    p.add_argument("--models-dir", required=True, help="dir holding the SERVED component_classifier_microsoft_vscode.pkl")
    p.add_argument("--manifest", default=str(PRIMARY / "data" / "models" / "MANIFEST.sha256"))
    for name in ("sample", "calibrate-sample"):
        p = sub.add_parser(name)
        p.add_argument("--processed-root", required=True, help="dir containing data/processed/ regenerated by 02/03")
        if name == "sample":
            p.add_argument("--extra-exclude", nargs="*", default=[])
    p = sub.add_parser("run")
    p.add_argument("kind", choices=[*KINDS, "all"])
    a = ap.parse_args()
    {"vocab": cmd_vocab, "sample": cmd_sample, "calibrate-sample": cmd_calibrate_sample, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
