"""Phase 1 for vscode (spec-autonomous-run.md Q4): CPU component-classifier candidates on the ADR-0070 taxonomy.

PRE-REGISTERED DESIGN (2026-10-23; do not change after seeing test results). Reuses the k8s
Phase 1 pieces unchanged (qp1_tfidf_lr fit_lr / proba_full / GRID / vectoriser settings,
qp1_eval bootstrap / Wilson / ResultsWriter test-once guard, qp1_conformal score functions); no
k8s file is modified.
 * Taxonomy (ADR-0070): population = issues with >= 1 non-excluded label at final; primary =
   qp0_ceiling_labels.per_issue with family = qp2_vscode_taxonomy.not_excluded; gold class =
   parent_of(primary) if in the 85-class list else `other` (last class). The class list is
   regenerated from the TRAIN split only (parents carried by >= 30 train issues, labels_final) and
   its sha256 is asserted equal to ADR-0070's. Train on train-split classifier_eligible labelled
   issues; validation / test the same filter. Text = title + "\\n" + body[:4000] (as k8s).
 * Candidates (model ids): V0 majority_prior_v0 = train class frequencies as probabilities.
   V1 tfidf_lr_v1 = k8s candidate 1 exactly (TF-IDF word 1-2 grams, min_df 2, sublinear, 200k;
   multinomial LR; 86 classes incl. `other`; grid C {1,4,16,64} x class_weight {None,balanced},
   best on validation by the key below, then smaller C). V2 tfidf_lr_family_only_v2 = same
   features / grid, trained on issues whose gold is NOT `other` (85 classes, `other` column = 0);
   it abstains (top-1 becomes `other`) when its max probability < tau, tau = the value on the
   validation max-probability grid (200 quantiles) maximising abstention F1 against gold `other`
   (ties: higher precision). Production baseline production_vscode_ovr = the served
   data/models classifier (calibrated sigmoid scores, load_classifier), each class mapped through
   parent_of -> 85 classes else `other`, score of a parent = max over its source classes; a fixed
   model, scored on validation and test without selection. Embedding / fine-tuned candidates are
   NOT run (GPU quota).
 * Scoring. HEADLINE (strict): top-k hit iff gold != other and gold in the model's top-k ranked
   classes; a predicted `other` never earns a hit; V2's strict ranking is its family ranking
   (abstention is reported next to it, not applied to it). k8s-convention variant (lenient):
   `other` is an ordinary class, a gold-other issue is a hit when `other` is predicted (top-1) or
   in the top-3; V2's lenient ranking puts `other` first when it abstains. Also: in-family strict
   top-1/top-3 (issues with gold != other), strict any-valid top-1/top-3 (true set minus `other`),
   macro-F1 over classes present in y_true on the lenient top-1, abstention rate / precision /
   recall (abstain = lenient top-1 is `other`; precision vs gold other; recall of gold other),
   the empirical coverage bound (share of gold != other), Wilson 95% for proportions, 2000-sample
   percentile bootstrap (seed 42), paired differences on shared resamples.
 * Selection (validation): among V0 / V1(best row) / V2(best row), the highest strict top-3, tie
   macro-F1, tie simpler (V0 < V1 < V2). Production is never a selection candidate.
 * Conformal (stage `conformal`, VALIDATION ONLY, exploratory per ADR-0069): alpha 0.05 (the
   spec pre-registration; a first run used 0.1 in error, see ADR-0071 amendment); candidate
   scores aps_nonrandomised, aps_randomised, raps (lambda 0.1, k_reg 1, as k8s), lac (qp1_conformal
   definitions); 50 seeded random 50/50 splits of the labelled eligible validation set (calibrate on
   one half, evaluate coverage w.r.t. the primary label, `other` included, and set size on the
   other); choose by: mean coverage within 0.95 +/- 0.02, then smallest mean set size, then name;
   if none qualifies, the coverage closest to 0.95. For V2 the population is in-family issues
   only (its `other` column is 0, so a gold-other issue cannot be covered). The choice is frozen in
   ADR-0071 before the test stage, which refuses to run unless that ADR names the chosen score.
 * Test (stage `test`, explicit): test labels are read only there; selected model, production and
   V0 are each evaluated once per report file (ResultsWriter guard); paired bootstrap of the
   selected model vs production and vs V0 (strict top-1, strict top-3, in-family top-3, lenient
   top-3); conformal calibrated on FULL validation applied to test; CPU latency (batch ms/issue
   and single-issue p50 / p95 over the first 200 test issues). Probabilities and model files go to
   D:\\triageiq-data\\models\\microsoft__vscode\\phase1\\, never the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # triage_iq (production model)
import qp1_conformal as qc  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_lr as t1  # noqa: E402
from qp1_vscode_lib import (  # noqa: E402
    COMPLEXITY,
    CONF_ALPHA,
    FROZEN_SHA,
    Ranked,
    best_abstention_tau,
    score,
    sel_key,
        clf.predict_proba_calibrated(_build_text(pd.Series([r["title"]]), pd.Series([r["body"]])))
        one.append((time.perf_counter() - t0) * 1000)
    lat[IDS["prod"]] |= {"single_issue_ms_p50": float(np.percentile(one, 50)), "single_issue_ms_p95": float(np.percentile(one, 95))}  # fmt: skip
    views[IDS["prod"]] = rank_proba(preds[IDS["prod"]], other)
    if IDS["v0"] in ids and sel_id != IDS["v0"]:
        preds[IDS["v0"]], _ = majority_proba(ytr, classes, len(te))
        views[IDS["v0"]] = rank_proba(preds[IDS["v0"]], other)
    y, sm = ev.to_idx(yte, classes), ev.set_mask(ste, classes)
    hits = {m: hit_arrays(y, sm, views[m], other) for m in ids}
    for m in ids:
        pf = save_pred(out / f"test_{m}.parquet", te["number"], preds[m], classes)
        writer.add(
            m, "test", score("test", True, yte, ste, views[m], classes, n_boot), pf, final=True
        )
    writer.data["paired_vs"] = {
        f"{sel_id}_minus_{m}": paired(hits[sel_id], hits[m], n_boot) for m in ids if m != sel_id
    }
    writer.data["bounds"] = {
        "spec_strict_test_coverage_bound_all_labelled": ceil["coverage_bounded_ceiling"][
            "labelled"
        ]["test"]["primary_in_family_top1_bound"],  # fmt: skip
        "measured_bound_on_eligible_labelled_test": prop(int((y != other).sum()), len(y)),
    }
    writer.data["populations"] |= {
        "test_eligible_total": int(te_all["classifier_eligible"].sum()),
        "test_labelled_eligible": len(te),
        "test_unlabelled_eligible": int(te_all["classifier_eligible"].sum()) - len(te),
    }
    writer.save()
    s = writer.data["models"][sel_id]["test"]
    return {
        k: round(s[k]["point"], 4)
        for k in ("strict_top1", "strict_top3", "lenient_top3", "macro_f1")
    }


def stage_conformal_test(a: argparse.Namespace, derived, splits_dir, models_dir) -> dict[str, Any]:  # noqa: ANN001
    """Apply the ADR-0071 conformal method (alpha 0.05) to TEST once; exploratory (ADR-0069)."""
    report, manifest = report_paths(a)
    writer = ev.ResultsWriter(report, manifest)
    if writer.data.get("conformal_test", {}).get("alpha") == CONF_ALPHA:  # test-once guard
        raise ev.TestGuardError(f"alpha-{CONF_ALPHA} conformal test already evaluated")
    conf = writer.data.get("conformal_selection")
    adr = ADR_PATH.read_text(encoding="utf-8").lower() if ADR_PATH.exists() else ""
    if not conf or conf["alpha"] != CONF_ALPHA or conf["selected"]["name"].lower() not in adr:
        raise RuntimeError("alpha-0.05 conformal choice must be frozen in ADR-0071 before test")
    sel_id, classes = writer.data["selected_model"], writer.data["classes"]
    tag, other, spec = {IDS["v1"]: "v1", IDS["v2"]: "v2"}[sel_id], classes.index(OTHER), conf["selected"]  # fmt: skip
    out = models_dir / "phase1"
    va = labelled_eligible(load_split(derived, splits_dir, "validation"))
    pv = pd.read_parquet(out / f"validation_{tag}.parquet")[classes].to_numpy()
    yv = ev.to_idx(gold(va, classes)[0], classes)
    te = labelled_eligible(load_split(derived, splits_dir, "test"))  # test labels read here
    saved = pd.read_parquet(out / f"test_{sel_id}.parquet")
    assert (saved["number"].to_numpy() == te["number"].to_numpy()).all()
    pt, (yte, ste) = saved[classes].to_numpy(), gold(te, classes)
    y, sm = ev.to_idx(yte, classes), ev.set_mask(ste, classes)
    kv = np.ones(len(yv), bool) if tag == "v1" else yv != other
    q = qc.calibrate(pv[kv], yv[kv], spec, qc.uniforms(int(kv.sum()), 0), CONF_ALPHA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
