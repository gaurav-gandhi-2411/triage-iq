"""Phase 1 candidate 4: fine-tuned encoder (multi-label BCE) on the kubernetes sig/* task.

PRE-REGISTERED DESIGN (2026-10-23; written and committed with the validation results BEFORE any
test evaluation; do not change after seeing test results). Task, gold, classes, population,
splits and metrics are exactly those of qp1_tfidf_lr.py / qp1_eval.py (ADR-0067: 21 sig/* classes
with >= 30 train issues + `other`; primary label = earliest surviving `labeled` event).
 * Model: microsoft/deberta-v3-small at pinned revision a36c739020e01763fe789b4b85e2df55d6180012;
   answerdotai/ModernBERT-base (rev 8949b909...) is used ONLY if deberta fails to load on Kaggle,
   and the report says which one ran. Input: title + "\\n" + body[:4000], max_seq_len 384.
 * Head/loss: 22 outputs, multi-label BCE; target = every final sig/* label mapped to the class
   list (`other` for labels outside it). Ranking/metrics/conformal use the sigmoid scores
   renormalised to sum 1 per row. AdamW (wd 0.01), batch 16, linear schedule with 10% warmup,
   grad-clip 1.0, seed 42, fp16 autocast. Grid: lr {2e-5, 5e-5} x epochs {3, 5}; each config is
   trained on TRAIN only and scored on VALIDATION inside the kernel; selection = validation top-3,
   tie macro-F1. Only the selected config's validation and test probabilities are written.
 * Test labels never leave this machine: the Kaggle dataset has labels for train/validation and
   texts only (all classifier_eligible issues, no label information) for test. Test metrics are
   computed locally ONCE (--final-test) with the qp1_eval test-once guard: top-1, top-3,
   any-valid, macro-F1, ECE, NLL with bootstrap CIs; paired bootstrap vs candidate 1 (saved test
   probabilities); the frozen ADR-0069 RAPS(0.1, 1) conformal method, EXPLORATORY.
 * Serving artifact: dynamic-int8 ONNX of the selected model. Its validation parity vs fp32 is
   measured in the kernel; its test probabilities are evaluated as a SEPARATE model id
   (`finetuned_encoder_v1_int8`, secondary, not used for selection or the headline claim).
 * Latency: single-row CPU p50/p95 of the int8 ONNX (tokenise + run), 200 test texts (seed 42
   sample), measured locally in a throwaway venv with onnxruntime + tokenizers (inference only),
   at all cores and at 2 threads (the 2-vCPU container shape); budget 300 ms p50 per added model.
 * Success reading: the model beats candidate 1 only if the paired top-3 CI excludes zero.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qk_kaggle as qk  # noqa: E402
import qp1_embedding_candidates as ec  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_lr as tl  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
MODEL_ID, INT8_ID = "finetuned_encoder_v1", "finetuned_encoder_v1_int8"
SLUG, DS_SLUG, KERNEL = (
    "kubernetes__kubernetes",
    "triageiq-ft-sigcls-k8s",
    "triageiq-finetune-cls-k8s",
)
ROOT = Path(r"D:\triageiq-data")
LATENCY_VENV = ROOT / "venvs" / "onnxlat" / "Scripts" / "python.exe"
BUDGET_MS, N_LAT = 300.0, 200


def paths(a: argparse.Namespace | None = None) -> dict[str, Path]:
    out = ROOT / "models" / SLUG / "finetuned_deberta"
    return {
        "derived": ROOT / "derived_v2" / SLUG, "splits": ROOT / "splits" / SLUG, "out": out,
        "c1_dir": ROOT / "models" / SLUG / "tfidf_lr",
        "report": Path("reports") / f"qp1_finetuned_encoder_{SLUG}_2026-10.json",
        "manifest": Path("reports") / f"qp0_split_manifest_{SLUG}_2026-10.json",
    }  # fmt: skip


def labelled_frame(d: pd.DataFrame, modelled: list[str]) -> pd.DataFrame:
    """number, text, primary class, JSON list of mapped final classes (train/validation only)."""
    y, sets = tl.gold(d, modelled)
    return pd.DataFrame({
        "number": d["number"].astype("int64"), "text": d["text"], "primary": y,
        "labels": [json.dumps(sorted(s)) for s in sets],
    })  # fmt: skip


def eligible_test_texts(derived: Path, splits_dir: Path) -> pd.DataFrame:
    """Texts of every classifier_eligible test issue; reads NO label table (no label leakage)."""
    sp = pd.read_parquet(splits_dir / "test.parquet", columns=["number", "classifier_eligible"])
    sp = sp[sp["classifier_eligible"]]
    issues = pd.read_parquet(derived / "issues.parquet", columns=["number", "title", "body"])
    issues = issues[issues["number"].isin(sp["number"])].sort_values("number")
    text = issues["title"].fillna("") + "\n" + issues["body"].fillna("").str.slice(0, 4000)
    return pd.DataFrame({"number": issues["number"].astype("int64").to_numpy(), "text": text.to_numpy()})  # fmt: skip


def build_frames(derived: Path, splits_dir: Path) -> dict[str, Any]:
    d = tl.load_split(derived, splits_dir, ("train", "validation"))
    modelled, classes = tl.define_classes(d[d["split"] == "train"])
    tr, va = (d[(d["split"] == s) & d["classifier_eligible"]].reset_index(drop=True) for s in ("train", "validation"))  # fmt: skip
    return {"train": labelled_frame(tr, modelled), "validation": labelled_frame(va, modelled),
            "test_texts": eligible_test_texts(derived, splits_dir), "classes": classes}  # fmt: skip


def cmd_build_dataset(a: argparse.Namespace) -> None:
    p = paths()
    fr = build_frames(p["derived"], p["splits"])
    out = qk.STAGING / DS_SLUG
    out.mkdir(parents=True, exist_ok=True)
    for k in ("train", "validation", "test_texts"):
        fr[k].to_parquet(out / f"{k}.parquet", index=False)
    (out / "classes.json").write_text(json.dumps({"classes": fr["classes"]}))
    meta = {"title": DS_SLUG, "id": f"{qk.OWNER}/{DS_SLUG}", "licenses": [{"name": "other"}], "isPrivate": True}  # fmt: skip
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print({k: len(fr[k]) for k in ("train", "validation", "test_texts")}, "classes", len(fr["classes"]))  # fmt: skip
    exe = qk.DEFAULT_KAGGLE
    listing = qk.kaggle(exe, ["datasets", "list", "--mine", "--search", DS_SLUG, "--csv"]).stdout
    verb = ["datasets", "version", "-p", str(out), "-m", "refresh", "-t"] if meta["id"] in listing else ["datasets", "create", "-p", str(out), "-t"]  # fmt: skip
    qk.kaggle(exe, verb)


def cmd_push(a: argparse.Namespace) -> None:
    """Push the kernel; retry every 10 min up to 2 h while the GPU-session quota is full."""
    import shutil

    stage = qk.STAGING / f"kernel-{KERNEL}"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree("kaggle/kernels/finetune_cls", stage, ignore=shutil.ignore_patterns("kernel-metadata.json"))  # fmt: skip
    meta = {**qk.kernel_metadata(qk.OWNER, KERNEL, [f"{qk.OWNER}/{DS_SLUG}"]), "code_file": "finetune_cls.py"}  # fmt: skip
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    deadline = time.monotonic() + 7200
    while True:
        res = qk.kaggle(qk.DEFAULT_KAGGLE, ["kernels", "push", "-p", str(stage), "--accelerator", "NvidiaTeslaT4"])  # fmt: skip
        text = (res.stdout + res.stderr).strip()
        if "successfully pushed" in text.lower():
            print(text)
            return
        print("push not confirmed:", text[-300:], flush=True)
        if time.monotonic() > deadline:
            raise RuntimeError("GPU session quota blocked for 2 h; stop and report")
        time.sleep(600)


def align(df: pd.DataFrame, numbers: pd.Series, classes: list[str]) -> np.ndarray:
    """Probabilities re-ordered to `numbers`; refuses duplicate/missing ids or class drift."""
    if list(df.columns) != ["number", *classes]:
        raise ValueError("probability file columns differ from the class list")
    if df["number"].duplicated().any():
        raise ValueError("duplicate ids in probability file")
    missing = set(numbers.astype(int)) - set(df["number"].astype(int))
    if missing:
        raise KeyError(f"{len(missing)} ids missing from probability file, e.g. {sorted(missing)[:3]}")  # fmt: skip
    return df.set_index("number").loc[numbers.to_numpy(), classes].to_numpy()


def read_run(out: Path, classes: list[str]) -> dict[str, Any]:
    info = json.loads((out / "grid.json").read_text(encoding="utf-8"))
    info["run_info"] = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    return info


def cmd_validate(a: argparse.Namespace) -> None:
    """Confirm the kernel's validation numbers locally; write the design+validation report."""
    p = paths()
    d = tl.load_split(p["derived"], p["splits"], ("train", "validation"))
    modelled, classes = tl.define_classes(d[d["split"] == "train"])
    va = d[(d["split"] == "validation") & d["classifier_eligible"]].reset_index(drop=True)
    yva, sva = tl.gold(va, modelled)
    info = read_run(p["out"], classes)
    writer = ev.ResultsWriter(p["report"], p["manifest"])
    res = {}
    for mid, f in ((MODEL_ID, "validation_probs.parquet"), (INT8_ID, "validation_probs_int8.parquet")):  # fmt: skip
        if not (p["out"] / f).exists():
            continue
        pv = align(pd.read_parquet(p["out"] / f), va["number"], classes)
        res[mid] = ev.evaluate("validation", False, yva, sva, pv, classes)
        writer.add(mid, "validation", res[mid], p["out"] / f)
    sel = info["selected"]
    top3_here = res[MODEL_ID]["top3"]["point"]
    assert abs(top3_here - sel["top3"]) < 1e-9, f"kernel top3 {sel['top3']} != local {top3_here}"
    ri = info["run_info"]
    writer.data.update({
        "design": DESIGN, "harness_design": ev.DESIGN, "repo": "kubernetes/kubernetes", "classes": classes,
        "validation_grid": info["grid"], "selected_config": {"lr": sel["lr"], "epochs": sel["epochs"]},
        "kernel": {k: ri.get(k) for k in ("model", "revision", "gpu", "device", "torch", "load_failures",
                   "train_and_predict_seconds", "total_seconds", "onnx_parity_validation", "onnx_sizes_bytes", "onnx_error")},
    })  # fmt: skip
    writer.save()
    print(json.dumps({"selected": sel, "local_validation_top3": top3_here, "kernel": writer.data["kernel"]}, indent=1))  # fmt: skip


def run_latency(p: dict[str, Path], texts: list[str]) -> dict[str, Any]:
    """Run the onnxruntime probe in the throwaway venv (inference only; light)."""
    tf = p["out"] / "latency_texts.json"
    tf.write_text(json.dumps(texts), encoding="utf-8")
    cmd = [str(LATENCY_VENV), "-I", "scripts/qp1_onnx_latency.py", "--model", str(p["out"] / "model_int8.onnx"),
           "--tokenizer", str(p["out"] / "selected_model" / "tokenizer.json"), "--texts", str(tf), "--max-len", "384"]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    out = json.loads(res.stdout.strip().splitlines()[-1])
    out["budget_ms_p50"] = BUDGET_MS
    out["within_budget_p50"] = {k: v["p50_ms"] <= BUDGET_MS for k, v in out["end_to_end"].items()}
    out["artifact_bytes"] = {f: (p["out"] / f).stat().st_size for f in ("model_int8.onnx", "model_fp32.onnx")}  # fmt: skip
    return out


def final_test(p: dict[str, Path], n_boot: int = ev.N_BOOT, latency: bool = True) -> dict[str, Any]:
    """Stage 2: evaluate TEST once per model id; paired vs candidate 1; conformal; latency."""
    writer = ev.ResultsWriter(p["report"], p["manifest"])
    for m in (MODEL_ID, INT8_ID):  # fail before touching any test label
        if "test" in writer.data["models"].get(m, {}):
            raise ev.TestGuardError(f"test already evaluated for {m} in {p['report'].name}")
    d = tl.load_split(p["derived"], p["splits"], ("train", "validation"))
    modelled, classes = tl.define_classes(d[d["split"] == "train"])
    assert classes == writer.data["classes"], "class definition drifted since validation"
    va = d[(d["split"] == "validation") & d["classifier_eligible"]].reset_index(drop=True)
    yva, _ = tl.gold(va, modelled)
    te = tl.load_split(
        p["derived"], p["splits"], ("test",)
    )  # first and only place test labels are read
    te = te[te["classifier_eligible"]].reset_index(drop=True)
    yte, ste = tl.gold(te, modelled)
    c1_te = ec.load_saved(p["c1_dir"] / "test.parquet", te["number"], classes)
    tops: dict[str, Any] = {}
    for mid, fv, ft in ((MODEL_ID, "validation_probs.parquet", "test_probs.parquet"),
                        (INT8_ID, "validation_probs_int8.parquet", "test_probs_int8.parquet")):  # fmt: skip
        if not (p["out"] / ft).exists():
            continue
        pv = align(pd.read_parquet(p["out"] / fv), va["number"], classes)
        pt = align(pd.read_parquet(p["out"] / ft), te["number"], classes)
        t = ev.evaluate("test", True, yte, ste, pt, classes, n_boot)
        t["paired_vs_candidate1"] = ec.paired_vs_c1(pt, c1_te, yte, ste, classes, n_boot)
        t["conformal_exploratory"] = ec.raps_conformal(pv, yva, pt, yte, ste, classes)
        t["n_test_eligible_labelled"] = len(te)
        t["headline"] = mid == MODEL_ID
        writer.add(mid, "test", t, p["out"] / ft, final=True)
        tops[mid] = t
    if latency:
        sample = np.random.default_rng(ev.SEED).choice(len(te), N_LAT, replace=False)
        texts = te["text"].iloc[sample].tolist()
        writer.data["latency_int8_onnx"] = run_latency(p, texts)
    writer.save()
    return tops


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tuned encoder candidate for the sig/* task")
    ap.add_argument("stage", choices=["build-dataset", "push", "validate", "final-test"])
    a = ap.parse_args(argv)
    if a.stage == "build-dataset":
        cmd_build_dataset(a)
    elif a.stage == "push":
        cmd_push(a)
    elif a.stage == "validate":
        cmd_validate(a)
    else:
        for m, t in final_test(paths()).items():
            print(m, {k: round(t[k]["point"], 4) for k in ("top1", "top3", "anyvalid_top1", "macro_f1", "ece")})  # fmt: skip
            print(" paired top3 vs c1:", t["paired_vs_candidate1"]["top3"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
