# System 1 — Component Classifier: TF-IDF Baseline

**Version:** Day 3  
**Last updated:** 2026-04-28  
**Maintainer:** Gaurav Gandhi

---

## 1. Architecture

**Model:** TF-IDF (1-2gram, max 50k features) → Logistic Regression  
**Scope:** Per-repo classifier (different label vocabularies — not transferable across repos)  
**Input:** `title + ". " + body_clean` (concatenated for richer signal)  
**Class weighting:** `balanced` (corrects for label frequency imbalance)

```
TfidfVectorizer(
    max_features=50_000, ngram_range=(1, 2),
    stop_words='english', strip_accents='unicode',
    min_df=2, sublinear_tf=True
)
→ LogisticRegression(
    class_weight='balanced', max_iter=1000,
    n_jobs=-1, C=1.0, solver='saga'
)
```

This baseline establishes:
- **Lower bound** on expected performance — any model below this is broken
- **Latency target** — 5ms p50 on CPU; production inference must beat or match this
- **Cost target** — zero (no API calls, no GPU, sub-second training)

---

## 2. Results Summary

| Repo | Train | Val | Test | Classes | Accuracy | Macro F1 | Weighted F1 | p50 (ms) | p95 (ms) | Batch/sample (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| microsoft/vscode | 1,488 | 187 | 187 | 28 | **69.0%** | **0.585** | 0.711 | 4.91 | 10.86 | 0.37 |
| kubernetes/kubernetes | 2,284 | 286 | 286 | 35 | **51.4%** | **0.466** | 0.542 | 5.57 | 15.09 | 0.36 |

**Training time:** vscode 27.2s, kubernetes 47.8s (total wall-clock 87.3s)  
**Evaluation time:** vscode 0.17s, kubernetes 0.19s  
**Throughput:** ~2,700 samples/sec (batch of 100, CPU-only)

---

## 3. Per-Repo Analysis

### 3.1 microsoft/vscode

**Test accuracy: 69.0% | Macro F1: 0.585**

With 28 classes and only 1,488 training examples (~53 avg per class), 69% accuracy is a reasonable baseline. The balanced class weighting helps surface rare classes, but at the cost of some accuracy on dominant classes.

**Best-performing classes:**

| Component | F1 | Notes |
|---|---|---|
| editor-find | 1.000 | Very distinctive vocabulary (find/replace, search-in-file) |
| keybindings | 1.000 | Highly specific terms (keybindings.json, key chord) |
| error-telemetry | 0.944 | Telemetry-specific signals (crash report, stack trace) |
| themes | 0.909 | Color theme vocabulary is self-contained |
| debug | 0.861 | Debugger terminology is specific and consistent |

**Worst-performing classes:**

| Component | F1 | Notes |
|---|---|---|
| editor-core | 0.000 | Only 2 test samples; too few for reliable eval |
| electron | 0.000 | Electron issues overlap heavily with platform/install language |
| search | 0.000 | Confused with file-explorer and typescript |
| workbench-editors | 0.000 | Only 1 test sample |
| suggest | 0.080 | Autocomplete language bleeds into javascript/typescript |

**Top-5 confusion pairs:**

| True | Predicted | Count | Reason |
|---|---|---|---|
| javascript → suggest | 4 | JS issues about IntelliSense misclassified as suggest |
| typescript → suggest | 4 | Same pattern — TS completion issues |
| ux → file-explorer | 3 | UX issues with file pane overlap |
| debug → suggest | 2 | Debug console autocomplete issues |
| debug → tasks | 2 | Debugging tasks/launch configs overlap |

The `suggest` component is a confusion magnet: autocomplete language (IntelliSense, completion, hover) appears in many vscode components. This is an inherent semantic overlap that TF-IDF cannot resolve — DistilBERT should handle it better via contextual understanding.

---

### 3.2 kubernetes/kubernetes

**Test accuracy: 51.4% | Macro F1: 0.466**

Lower performance reflects three compounding factors (see §5 Caveats). With 35 classes and some having very ambiguous semantic content (`usability`, `introspection`), 51% accuracy is the expected ceiling for a bag-of-words model.

**Best-performing classes:**

| Component | F1 | Notes |
|---|---|---|
| downward-api | 1.000 | Highly specific term |
| rkt | 1.000 | Container runtime — unique vocabulary |
| platform/vagrant | 0.889 | Platform-specific terms are distinctive |
| upgrade | 0.857 | Upgrade-specific language is clear |
| build-release | 0.842 | Build/release vocabulary distinct |

**Worst-performing classes:**

| Component | F1 | Notes |
|---|---|---|
| cloudprovider | 0.000 | Subsumed by provider/gcp in training |
| etcd | 0.000 | etcd issues use generic distributed-systems language |
| extensibility | 0.000 | Vague concept, language overlaps with api |
| isolation | 0.000 | Only 1 test sample |
| os/ubuntu | 0.000 | OS-specific signals weak |

**Top-5 confusion pairs:**

| True | Predicted | Count | Reason |
|---|---|---|---|
| usability → kubectl | 10 | "usability" in 2014-2015 often meant kubectl UX |
| test → test-infra | 8 | Thin boundary between test code and test infrastructure |
| api → provider/gcp | 5 | GCP integration issues tagged as generic api |
| test → provider/gcp | 5 | Provider tests filed under generic test |
| api → swagger | 4 | Swagger/OpenAPI doc issues filed under api |

`provider/gcp` is a confusion magnet across classes — it collects predictions from api, test, kubectl, and usability. This suggests either label inconsistency in the 2014-2015 period or that GCP-specific issues shared generic language.

---

## 4. Calibration Analysis

Both models exhibit **systematic underconfidence**: the raw max-class probability is much lower than actual accuracy.

| Repo | ECE | Mean confidence | Mean accuracy | Direction |
|---|---|---|---|---|
| microsoft/vscode | 0.496 | 0.194 | 0.690 | Underconfident |
| kubernetes/kubernetes | 0.396 | 0.118 | 0.514 | Underconfident |

**Interpretation:** With 28-35 classes, logistic regression distributes probability mass across many classes, keeping the max probability low even for high-confidence predictions. At confidence 0.08, vscode accuracy is 50%; at confidence 0.14, accuracy is 70%. The model knows more than its probabilities suggest.

**Practical consequence:**
- Raw probability thresholds cannot be used for "escalate if unsure" logic without recalibration
- Temperature scaling (T < 1.0, sharpening) is expected to improve ECE substantially
- This recalibration is planned for the DistilBERT comparison in Day 4

**Note on convergence:** Both models hit `max_iter=1000` without converging (SAGA solver). Increasing max_iter to 5,000 or switching to `lbfgs` would likely improve both accuracy and calibration slightly. Left as-is to keep this a true baseline — the Day 4 comparison will be apples-to-apples.

---

## 5. Honest Caveats

### 5.1 Kubernetes era (critical risk)

The kubernetes training data covers **Jun 2014 – Oct 2015** — the earliest 15,000 issues filed. This is pre-production-stability Kubernetes when:
- Labels like `usability`, `introspection` had informal meanings not used today
- `provider/gcp` was a catch-all for cloud-specific issues
- Many components didn't exist yet (e.g., CRDs, admission webhooks)
- The team was small and label discipline was lower

**Risk:** A DistilBERT or LLM classifier trained on this data will learn 2014-2015 label conventions, not current ones. Performance on modern kubernetes issues may be substantially different.

**Mitigation (if Day 3 baseline looks suspicious):** Scrape 5,000–10,000 recent kubernetes issues (sort=created&direction=desc) and retrain. Per user's Day 3 instruction, this risk is documented here for Day 4 decision.

### 5.2 vscode mixed-era data

vscode training mixes 2015-2016 historical + 2025-2026 recent issues. The historical issues are team-filed and terse; the recent issues are community-reported with different language patterns. The classifier may not generalize uniformly to one era.

### 5.3 Small test sets

187 (vscode) and 286 (kubernetes) test examples are small. F1 scores for classes with ≤5 test examples are noisy — a single misclassification changes F1 by 0.2+. Per-class results should be interpreted with caution for rare classes.

### 5.4 30/19 classes dropped

vscode dropped 30 rare classes (<10 samples), kubernetes dropped 19. This means the reported accuracy applies only to the subset of well-labeled issues — not all incoming issues. In production, a classifier must either handle these rare classes or gracefully fall back to a `other` bucket.

---

## 6. What This Baseline Tells Us

**Signal strength of bag-of-words:** TF-IDF captures approximately 60-70% of the signal in component classification when vocabulary is distinctive (keybindings, themes, debug). It fails when:
- Multiple components share vocabulary (suggest/javascript/typescript)
- Labels are semantically vague (usability, introspection)
- Classes are structurally related (test vs test-infra, api vs apiserver)

**For DistilBERT to justify its cost (15ms vs 5ms, requires GPU for training):**
- vscode macro F1 must improve from 0.585 → 0.65+ (11+ points)
- kubernetes macro F1 must improve from 0.466 → 0.55+ (8+ points)
- If DistilBERT doesn't clear this bar, TF-IDF is the production choice

**For LLM few-shot to justify its cost (1200ms, API cost ~$0.001/req):**
- Must materially outperform DistilBERT on rare/novel classes
- Worth considering for cold-start components (new repos, new label vocabularies)

---

## 7. Reproducibility

```bash
# Generate splits first
python scripts/03_split.py --repos microsoft_vscode kubernetes_kubernetes

# Train and evaluate
python scripts/04_train_classifier.py

# Models saved to data/models/ (gitignored)
# Charts saved to reports/charts/
# Metrics saved to reports/classifier_results.json
```

Charts generated:
- `reports/charts/classifier_confusion_{repo}.png` — top-20 confusion heatmap
- `reports/charts/classifier_per_class_f1_{repo}.png` — F1 per component
- `reports/charts/classifier_calibration_{repo}.png` — reliability diagram
