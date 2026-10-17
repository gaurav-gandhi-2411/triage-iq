# Evaluation history

The record of measurement errors found and corrected in TriageIQ's evaluation, kept verbatim from the README as dated notes (newest topics first where the README ordered them). Where a number here conflicts with the current README evaluation table, the README wins. This file is not checked by `scripts/readme_numbers_check.py`: it is a history of what was reported and when, so its numbers are quoted records, not claims about the current system. Only edits since the move: relative link paths adjusted so links resolve from `docs/`, and "the table above" now means the README's evaluation table.

> **Reading order for the dated notes below.** They are a chronological record of what was reported and when; where a number in a note conflicts with the table above, the table wins (its sources are listed under it). Numbers attributed to `llama-3.1-8b-instant` or to the judge models used before 2026-09 are historical and not comparable to the current baseline.

> **2026-08-10 update — retrieval, resolution, and synthesis cutovers shipped (ADR-0040/0041/0043/0044), supersedes the retrieval/resolution/synthesis numbers in the notes below.**
> **Retrieval (ADR-0040):** two independent bugs fixed — corpus-side text was truncated at 512
> *characters* instead of BGE's real 512-*token* limit (dropped a median 78/165 tokens per
> truncated k8s/vscode issue), and BGE's model-card-documented query instruction prefix was
> never applied. Lever 1 (truncation fix) ships unconditionally for both repos. Lever 2 (query
> instruction) ships **on for k8s only** (`QUERY_INSTRUCTION_REPO_OVERRIDE`) — it's a
> significant additive win for k8s (+6.67pp) but directionally *negative* for vscode's
> duplicate-matching task, so shipping it uniformly would trade away a proven gain for config
> simplicity. Result: **k8s R@5 18.0%→24.67%** [18.0, 31.3], **vscode R@5 50.5%→53.5%**
> [46.5, 60.5]. Live-verified through the real, un-overridden `/triage` code path, not just a
> unit test. A follow-up investigation (2026-08-10,
> [`docs/investigations/vscode-retrieval-lexical-and-hybrid-2026-08-10.md`](investigations/vscode-retrieval-lexical-and-hybrid-2026-08-10.md))
> tested whether near-duplicate/lexical-overlap dominance explains why the instruction helps
> k8s but hurts vscode — that specific hypothesis did not hold up; the asymmetry's real cause
> stays open. Hybrid BM25+dense fusion was re-tested against the corrected corpus and rejected
> again on both repos (dense-only remains stronger).
>
> **Retrieval, k8s eval-population audit (2026-08-11):** the 24.67% k8s number above measures
> R@5 over the full 150-pair eval population, but hand-categorizing a sample of its misses found
> most weren't retrievable-in-principle: umbrella/tracking issues that reference many unrelated
> sub-issues at once (no single embedding can be close to all of them), and citations where the
> target is background/precedent for a topic the query is actually about something else. Built a
> clean subset the honest way to avoid selection bias — **two exclusion criteria (checklist-style
> umbrella queries; causal-only references) were written down *before* any pair was scored
> against retrieval outcomes**, applied blind to hit/miss across all 150 pairs by 5 independent
> reviewers, and only *then* joined to the already-computed results. 66/150 pairs survive as a
> fair content-similarity test; **clean-subset R@5 = 39.39% [27.3, 51.5]**, CI lower bound above
> the unfiltered point estimate. Excluded pairs hit at 13.1% vs. valid pairs' 39.4% — reported as
> the check against selection bias (exclusion labels never saw the outcome, so this gap reflects
> the criteria's validity, not outcome-driven cherry-picking) rather than hidden. Full methodology
> and reproducible scripts:
> [`docs/investigations/2026-08-11-k8s-retrieval-ceiling-and-vscode-resolution-close.md`](investigations/2026-08-11-k8s-retrieval-ceiling-and-vscode-resolution-close.md).
> Read: the prior "weak k8s retriever" framing was substantially an eval artifact, not a model
> quality problem — the unmodified BGE embedder is materially better than 24.67% suggested. The
> ~60% miss rate on the clean subset is still real headroom, not a solved problem.
>
> **Resolution (ADR-0041):** the shipped resolution models were trained on a stale split
> (regenerated 2026-05-30) against a corpus that had grown 99-100% since (regenerated
> 2026-07-11, Phase 2b). Re-splitting from the current corpus roughly doubles k8s's
> bucket-classifier gain over naive: **+3.27pp→+6.35pp** [+5.08, +7.55] (measured with the training-time embedding features; **as served today +5.05pp** [+3.94, +6.22], see the table notes), cutover and
> live-verified. Recency weighting added nothing on top (k8s's corpus is still entirely
> 2014-2016 in calendar time even with 100% more rows, so there's no recency signal to exploit).
> vscode's re-split was tried and **rejected** — it makes the bucket classifier dramatically
> worse (-24.73pp with recency weighting), so vscode's naive-prior fallback is unchanged.
>
> **Synthesis (ADR-0043):** ADR-0039 accepted a k8s synthesis-quality regression (-0.6226 vs.
> the frozen OLD baseline) as a side effect of the ADR-0036 classifier cutover, diagnosed as
> confidence-framing (tighter per-class confidence scores reading to the LLM as "the model is
> unsure," inducing hedged language). The retrieval and resolution fixes above are unrelated to
> that classifier and don't touch its confidence output — but re-running the full pipeline with
> both landed anyway **recovered 61% of the original gap**: k8s's judge mean moved
> 9.8868→**10.2642**/15. Per-dimension analysis confirms *why*: `similar_issues_relevance` and
> `resolution_estimate_reasonableness` (fed directly by the two fixes) recovered strongly or
> fully; `component_match` and `next_steps_actionability` (the two dimensions ADR-0037 traced
> specifically to confidence-framing) barely moved at all — exactly the causal signature you'd
> expect if both mechanisms are real and independent. **10.2642 is now the committed baseline**
> (`test_k8s_quality_regression`'s `xfail` marker removed) — the residual -0.2452 vs. OLD is a
> deliberately accepted cost of the classifier's verified ground-truth accuracy win, not an
> open item this gate tracks going forward. vscode: 8.7273→8.6364, flat within its own noise
> band. A follow-up probe (2026-08-10,
> [`docs/investigations/confidence-structural-representation-2026-08-10.md`](investigations/confidence-structural-representation-2026-08-10.md))
> tested ADR-0037's last untested lever (showing the LLM less/differently-structured confidence
> information, not just different wording) — neither candidate cleared the bar for a full
> confirming run; one candidate introduced a new component-label fabrication failure mode the
> current prompt doesn't have. No further prompt-lever work is planned.
>
> **Fabrication rate promoted to a hard, zero-tolerance blocking gate (ADR-0044).** Previously
> informational; PR #57 promoted both `eval-gate.yml` jobs off `continue-on-error`, and ADR-0044
> then explicitly decided **not** to add per-repo slack (e.g. for vscode's small n=11 sample) —
> a discrete, replay-deterministic correctness signal (no live LLM call, confirmed
> byte-identical across replays) doesn't get the same statistical tolerance band as the
> continuous judge-score gate. As of 2026-08-10: 0/53 k8s, 0/11 vscode (the 2026-09-23 baseline reads 1/53 k8s, ADR-0061; this is a consistency check, not an error rate, see the table notes).
> Full reasoning: [`docs/architecture/adr/0040-retrieval-truncation-and-query-instruction.md`](architecture/adr/0040-retrieval-truncation-and-query-instruction.md), [`docs/architecture/adr/0041-resolution-stale-split-recency-weighting.md`](architecture/adr/0041-resolution-stale-split-recency-weighting.md), [`docs/architecture/adr/0043-combined-cutover-synthesis-quality-recovery.md`](architecture/adr/0043-combined-cutover-synthesis-quality-recovery.md), [`docs/architecture/adr/0044-fabrication-rate-hard-gate-bound.md`](architecture/adr/0044-fabrication-rate-hard-gate-bound.md).
>
> **Headline finding (2026-07-19, final framing per ADR-0033, supersedes the 2026-07-16
> framing below): retrieval quality is now measured on a clean, hand-verified, disjoint
> eval set — and it's honestly weaker than previously reported, not better.** k8s's clean
> product-task R@5 (issue → related issue, its only available task — vscode has a
> duplicate-comment channel k8s never had) is **9.3% [5.3, 14.0]** (n=150,
> hand-verified genuine by construction) — down from the previously-reported 23.5%, which
> was itself measured on a gold set later found to be title_sim-contaminated (ADR-0032).
> Two factors plausibly explain the drop, not fully disentangled: cleaner (harder, less
> lexically-inflated) pairs, and a ~2x larger candidate corpus (30,000 vs. the old index's
> 15,000) that mechanically lowers recall. **vscode reverses its "unmeasured" status**:
> its `dup_comment` channel — never precision-audited before ADR-0033 — turned out to be
> vscode's largest, cleanest channel (85% precision, n=2,242), giving a properly powered
> duplicate-task R@5 of **43.5% [37.0, 50.5]** (n=200). vscode's separate related-issue
> task (non-duplicate) stays underpowered (n=19, directional only, reported but never
> gated) — vscode is genuinely two different tasks, reported separately, never blended.
> Full reasoning: [`docs/architecture/adr/0033-clean-retrieval-data-trustworthy-eval.md`](architecture/adr/0033-clean-retrieval-data-trustworthy-eval.md).
>
> **Harness correction (2026-07-24, ADR-0035, supersedes the 2026-07-23 framing below) —
> three measurement bugs found and fixed; the corrected picture is weaker-but-real baselines
> and NO SIGNAL from fine-tuning, not a confirmed regression.** GG's call after five
> independent retrieval-improvement techniques all failed: "stronger evidence of a broken
> harness than of a uniquely hard task." The audit found (1) every eval query was **title-only**
> while production embeds title+body untruncated (`triage.py`) — eval-set JSON never populated
> `query_body`; (2) the D2 fine-tune trained/saved **mean pooling** against BGE-base's native
> **CLS-token pooling**; (3) training truncated at 128 tokens, cutting **65.73%** of examples
> (measured token-length p95=230, corrected to 256); (4) all three ADR-0031 levers were measured
> against an unaudited, pre-D1 pair population (277/292 pairs, only 72%/20% hand-verified
> genuine) instead of D1's clean, disjoint eval sets. **Corrected canonical baselines: k8s R@5
> 9.3%→18.0% [12.0, 24.0], vscode R@5 43.5%→50.5% [43.0, 57.5]** (both nearly double/gain 7pts
> — production was always this good, only the measurement undersold it). **All four ADR-0031
> levers are still rejected** on the corrected harness, but **weighted fusion flipped sign** —
> originally reported as marginally shipping (+3.25pp k8s, +4.11pp vscode, both CIs excluding
> zero), it's now flat on k8s and a **significant vscode regression** (−8.0pp, CI [−13.5,−3.0]).
> BM25-alone (newly measured) is the *weaker* system on both repos, not a diluted strong
> signal. **The D2 fine-tune conclusion is WITHDRAWN**: re-run with both bugs fixed, the result
> is **+2.0pp R@5 (50.5%→52.5%), CI [−4.5, +8.5]** — every metric moved positive (a full sign
> reversal from the withdrawn "confirmed regression"), but no CI excludes zero. At n=200 the CI
> half-width is ≈±6.5pp — **this is underpowered, not disproven**; whether more data or
> different hyperparameters would show a real, detectable lift is genuinely open. Still
> flagged, not fixed: corpus/query truncation asymmetry (512 vs. untruncated), BGE's unused
> query instruction prefix, BM25 tokenization uninspected. Nothing in production changed —
> confirmed the served index and model are untouched throughout. Full reasoning:
> [`docs/architecture/adr/0035-retrieval-harness-correction.md`](architecture/adr/0035-retrieval-harness-correction.md).
>
> **D2 fine-tune result (2026-07-23, ADR-0034 — historical, WITHDRAWN above, kept for
> provenance): confirmed regression, BGE-base off-the-shelf stays the shipped retriever.**
> D1's clean, disjoint vscode_duplicate data (1,734 training pairs, leakage guard PASSED
> 0-overlap on every run) was fine-tuned locally (BGE-base, MNRL loss, measure-first single
> run: 5 epochs, lr=2e-5). Result: **R@5 38.5% vs. the 43.5% baseline (−5.0pp, CI [−11.0,
> +0.5])** — every metric moved backward. A diagnostic leg then tested the leading alternative
> explanation (overfitting) directly: 2 epochs, lr=1e-5 — the textbook anti-overfit fix, ending
> training at a 5.7× higher, far-less-memorized loss (0.255 vs. 0.045). If overfitting were the
> cause, this should have recovered some of the gap. Instead **R@5 dropped further to 33.5%
> (−10.0pp, CI [−16.0, −4.5], excludes zero)** — reported at the time as a statistically
> confirmed regression. **This reasoning is now known to be wrong** (ADR-0035, above): both
> runs trained with mean pooling against BGE's native CLS pooling and truncated 65.73% of
> training examples at 128 tokens — the diagnostic leg never isolated either confound, so it
> could not have ruled out overfitting as *the* mechanism. Kept here for historical accuracy
> of what was reported and when; superseded finding is above. Original ADR:
> [`docs/architecture/adr/0034-d2-retrieval-finetune-honest-negative.md`](architecture/adr/0034-d2-retrieval-finetune-honest-negative.md).
>
> **Prior framing (2026-07-16, ADR-0030/0031/0032 — historical, superseded above): k8s
> product-task retrieval is genuinely weak, hand-verified; vscode product-task retrieval is
> currently UNMEASURED.** k8s Recall@5 on the actual product task ("given a new issue, find
> related issues"), against the actually-deployed live index: **23.5% [18.4, 28.5]** (n=277) —
> the lowest absolute score of any model in this system's per-model audit (classifier top-3
> 82.5%/90.4%, resolution's real k8s bucket gains, synthesis's floor-fail rates). A
> hand-judged precision audit (ADR-0032) confirms this number is real, not eval noise: 72%
> of the underlying gold pairs are genuinely related, and pulling the incidental 28% out
> doesn't move Recall@5 outside its own confidence interval. Three untried,
> zero-training improvement levers were then tried against it and **all three failed to
> clear the bar** (ADR-0031): hybrid BM25+dense fusion (CI crosses zero), a pretrained
> cross-encoder reranker (quality regresses on both repos, +190-330x latency), and a
> stronger pretrained embedder (CI crosses zero). The earlier "vscode is statistically
> indistinguishable from k8s, both ~23-27%" framing was retired: the same precision audit
> (ADR-0032) found vscode's gold pairs were only 20% genuinely related (94.9% sourced from
> an unaudited title-similarity channel, dominated by blank-template and
> generic-crash-title collisions). Full reasoning:
> [`docs/architecture/adr/0030-phaseC-product-task-feasibility.md`](architecture/adr/0030-phaseC-product-task-feasibility.md),
> [`docs/architecture/adr/0031-retrieval-quality-improvement.md`](architecture/adr/0031-retrieval-quality-improvement.md),
> [`docs/architecture/adr/0032-product-task-gold-pair-quality-audit.md`](architecture/adr/0032-product-task-gold-pair-quality-audit.md).
>
> **Classifier metric correction (2026-07-11).** The product never surfaces a single
> label — `triage.py` builds `classifier_top3` and `grounding.py::verify_plan_grounding`
> defines a correct prediction as top-3 membership, not top-1 equality. Reporting only
> top-1 materially understated the classifier's real-world usefulness (a 21–31pp gap,
> non-overlapping CIs, same model + test split — no retraining involved). Additionally,
> 30.4% of the kubernetes test set and 8.0% of vscode's have more than one valid
> component label that gets collapsed to one at preprocessing time; crediting a
> prediction that hits any valid label (not just the collapsed one) raises accuracy
> further to 59.4% (k8s) / 71.7% (vscode) top-1. Full methodology:
> [`reports/model_eval_audit.json`](../reports/model_eval_audit.json) → `component_classifier`.
>
> **DistilBERT dismissal correction (2026-07-24).** `05_train_distilbert.py`'s architecture
> comparison was dismissed as "TF-IDF latency and accuracy were sufficient at this data
> scale" — using **top-1**, the exact metric the correction above found was the wrong one
> to evaluate this classifier on. Re-evaluated the existing trained artifacts (no
> retraining) at top-3, the product's real correctness definition: **DistilBERT loses on
> both repos** — 88.2% vs. TF-IDF's 90.4% (vscode) and 74.5% vs. 82.5% (k8s, an 8pp gap) —
> despite appearing to *win* on vscode's top-1 (75.4% vs. 69.0%), which is what the original
> dismissal cited. Same wrong-metric error class as the retrieval corrections
> (ADR-0035): a comparison decided on a metric the product doesn't use. Calibration and
> latency were also materially worse (ECE 0.40 vs. 0.14 vscode; CPU p50 97.7ms/197.2ms vs.
> TF-IDF's ~5ms) — TF-IDF+LR remains the shipped classifier, now for a corrected reason.
> Full numbers: [`reports/distilbert_results_top3_corrected.json`](../reports/distilbert_results_top3_corrected.json).
>
> **Multi-label supervision fix SHIPPED (2026-07-24, ADR-0036) — same TF-IDF+LR architecture,
> corrected supervision.** `preprocess.py::normalize_labels()` keeps only the first matching
> component label per issue, discarding the rest — 30.4% of k8s test issues / 8.0% of
> vscode's genuinely have more than one. Retrained as one-vs-rest logistic regression over
> ALL valid labels (same TF-IDF features, only the supervision changes): **k8s top-3
> 82.5%→87.1% (+4.55pp, CI [+0.35,+8.39], excludes zero)**, vscode top-3 flat (90.4%→89.8%,
> ceiling + smaller collapse rate) but **top-1 improves significantly on both repos**
> (k8s +9.09pp, vscode +7.49pp, both CIs excluding zero) — consistent direction across three
> independent CIs, unlike ADR-0031's rejected weighted-fusion result (marginal + inconsistent
> direction across repos). Recalibrated (temperature scaling generalized to OvR's independent
> per-class logits, argmax-preserving, verified empirically): ECE improves on both repos
> (0.086→0.053 vscode, 0.111→0.090 k8s) and k8s's new overconfidence (a real risk the naive
> calibration objective first made *worse*, caught before shipping) is corrected. The
> selective-prediction abstention gate's fixed thresholds (ADR-0021, off by default) are now
> stale against this new confidence distribution — documented, not silently left dead. **Note:**
> displayed `component_confidence` values run structurally higher post-cutover (independent
> per-class sigmoids vs. a competing 35-way softmax, e.g. 0.221→0.391 for a comparable
> prediction) — a scale change, not increased model certainty; ECE confirms the new numbers are
> equally or more honest. Full reasoning:
> [`docs/architecture/adr/0036-classifier-multilabel-supervision-fix.md`](architecture/adr/0036-classifier-multilabel-supervision-fix.md).
>
> **Retriever metric correction (2026-07-11).** The advertised vscode Recall@5 (36.7%)
> was measured against `data/gold_related.parquet` (v1), which is only 74.0% genuine
> issue→issue pairs — the rest are PR→issue or duplicate-comment pairs, an easier proxy
> task the product doesn't perform. The next candidate honest number, measured against
> the actually-deployed live index, was 26.7% [21.6, 31.9] — a ~10pp inflation, CIs do not
> overlap. (This superseded ADR-0028's originally-reported 22.4% [17.7, 28.0] — n=254,
> ADR-0031 (2026-07-12) found no committed script ever reproduced that number and its
> denominator didn't match the full product stratum; 26.7%/n=292 is the byte-reproducible
> number via `scripts/phaseC_vscode_live_product_eval.py`.) **This 26.7% number was itself
> retired five days later (ADR-0032, below) — it wasn't a wrong computation, but a
> computation against a gold set later found to be ~80% incidental.**
>
> **k8s retriever re-eval (2026-07-12, ADR-0030).** kubernetes was first reported as
> "unmeasurable" (zero product-task pairs land in the *test-split* of the live corpus).
> That framing was wrong: the live-serving retriever is an off-the-shelf pretrained
> `BAAI/bge-base-en-v1.5` checkpoint, never trained on any gold pair — only a separate,
> unshipped fine-tuned artifact is — so the train/val/test split (built to prevent
> leakage for *that* fine-tune) carries zero leakage risk for the live model. Every
> product-task pair whose query and target both fall in the live index's number range
> is usable for measurement regardless of split label: 277 of 776 pairs, evaluated with
> the same method and bootstrap CI as vscode's number above
> (`scripts/phaseC_k8s_live_product_eval.py`) — see the headline finding above for the
> result. A W3 fine-tune was evaluated against both a proxy gate task and the product task
> (`docs/architecture/adr/0027-w3-retry-stratified-eval.md`): proxy-task gains were real
> and significant (k8s +14.3pp, vscode +4.6pp) — this gate stratum is unaffected by the
> gold-pair-quality finding below (ADR-0032 Finding 4: it's sourced from PR-query/
> dup_comment pairs, not the contaminated `title_sim` channel). Product-task gains were
> directionally positive and underpowered on both repos (k8s +3.5pp, vscode +3.2pp, both
> CIs cross zero) — **held, and stays held on both repos, per
> `docs/architecture/adr/0030-phaseC-product-task-feasibility.md`: NO-GO on mining more
> product-task pairs to gate either fine-tune, decided on value, not feasibility.** k8s's
> data ask was disproportionate anyway (~6,075 pairs, ~8x the current stratum), but
> that's not the operative reason — proving a ~3.5pp lift on top of a 23.5% base rate
> would still ship a retriever that misses the related issue 3 times out of 4. vscode's
> data gap is smaller and closeable (~664 more pairs), but closing it would only prove a
> +3.2pp lift against a ~23-27% base rate — the same weak-baseline problem, just with an
> easier data path. Neither fine-tune is a near-miss awaiting data; both are marginal
> against a weak baseline.
>
> **Retrieval-quality-lever pass (2026-07-12, ADR-0031).** Hybrid BM25+dense fusion, a
> pretrained cross-encoder reranker (the ADR-0006 retry against the corrected metric), and
> a stronger pretrained embedder were each tried against the product-task R@5 bar. All
> three rejected — see the headline finding above and
> [`docs/architecture/adr/0031-retrieval-quality-improvement.md`](architecture/adr/0031-retrieval-quality-improvement.md)
> for full per-lever results. The two remaining paths (in-domain fine-tuning, which
> reopens the leakage question ADR-0030 deliberately kept closed; or new related-pair
> mining, already NO-GO'd on value grounds) are both harder and were explicitly deferred,
> not attempted here.
>
> **Gold pair quality audit (2026-07-16, ADR-0032) — k8s confirmed real, vscode retired
> and marked unmeasured.** Before accepting ~23-27% as a ceiling, a hand-judged 50-pair
> sample (25/repo, fixed seed) checked whether the gold pairs behind these R@5 numbers
> are genuinely related. **k8s: 72.0% precision [52.4, 85.7]** — the pair set is 86%
> reference-mined ("See #N"/"Forked from #N"), and pulling incidental pairs out doesn't
> move R@5 outside its own CI (27.8% genuine-only vs. 23.5% full-set) — **the k8s finding
> holds up as real**, and stands as this project's headline retrieval number (above).
> **vscode: 20.0% precision [8.9, 39.1]** — 94.9% of its live-evaluated pairs come from
> `title_sim` (title-text similarity), a channel never precision-audited before, and the
> sample is dominated by boilerplate-template collisions ("Bayou"↔"11", both blank issue
> forms) and generic auto-bucket title matches on different actual bugs. **Conclusion:
> vscode's 26.7% is retired, not caveated** — it was measuring a ~80%-incidental gold set,
> not retrieval quality. vscode product-task retrieval is reported as **UNMEASURED** above
> and in the eval table, not as a number with an asterisk; no properly-powered clean
> re-measurement exists yet (the genuine-only subsample, n=5, is too small to read).
> Separately: of the genuinely-related pairs the retriever misses, 0/17 have zero shared
> vocabulary with their target (overlap up to 90 shared tokens on near-duplicate crash
> reports) — misses are lexically findable in principle, not an unfindable-by-any-method
> ceiling, for the k8s finding above. **Broader caution:** `title_sim` isn't scoped to this
> eval — it also feeds the (HELD, unshipped) W3 fine-tune's hard-negative mining, its
> train/val split, and its own training and val-monitoring data; the fine-tune's primary
> CI-gated metric (the `gate` stratum) is unaffected (PR-query/`dup_comment`-sourced, not
> `title_sim`), but any future revival of that fine-tune thread should re-audit or exclude
> `title_sim` pairs first. Full methodology, every pair's judgment, and the full consumer
> trace:
> [`docs/architecture/adr/0032-product-task-gold-pair-quality-audit.md`](architecture/adr/0032-product-task-gold-pair-quality-audit.md).
>
> **Synthesis quality metric redesign (2026-07-11).** The mean-band score is a
> *regression detector* (fails only if it drops below its own prior baseline by more
> than measured noise) — it was never a quality floor, and it structurally cannot catch
> fabrication: the judge scores the final `TriagePlan` JSON but never sees
> `classifier_top3` or the retrieved-issue set. Direct proof: a known hallucinated
> component (vscode issue #311836) scored 9/15 — *above* its own repo's 8.36/15 mean.
> Two new metrics close this gap, both computed with zero additional LLM calls
> (reused from signals the pipeline already produces): **floor-fail rate** (fraction of
> plans hitting the judge's own worst band — `component_match==0` or
> `similar_issues_relevance==0`) is reported here *alongside* the mean rather than
> averaged away — vscode's 45.5% floor-fail rate is invisible behind its
> passable-looking 55.8% mean. **Fabrication rate** promotes the existing deterministic
> grounding check (`grounding.py::verify_plan_grounding`) from informational-only to a
> named quality signal — currently reported informationally
> (`eval/test_quality_regression.py`'s CI job is non-blocking) pending an observation
> window before any promotion to a hard gate. The mean-band gate itself is unchanged.
>
> **Resolution reporting correction (2026-07-11).** This is the one model where the
> hard modeling/gating work (ADR-0009/0010/0021/0023/0025) was already rigorous and
> live-correct — the numbers above were stale (a 2026-05-30-era snapshot) and the
> table conflated the point-estimate and bucket-classifier outputs, which are two
> separate signals with separate gates. k8s: both the point estimate and the bucket
> classifier genuinely beat naive, CIs excluding zero. vscode: the point estimate
> (MAE) is served as-is despite losing to naive (−70.5%) — there's no fallback gate for
> the point/interval, only a transparency badge; the bucket classifier's raw output
> *does* have a trust gate (`BUCKET_CLASSIFIER_TRUSTED`, ADR-0025) and, because it loses
> to naive by −22.08pp, is not served — vscode's `resolution_bucket` field is the naive
> majority-class prior, honestly labeled low-confidence. Both losing numbers were
> already correctly reported internally (`reports/w6_resolution_diagnosis.json`); they
> just never reached this README, and a transcription error in ADR-0025's own table
> and a proxy-conflated prose note in `reports/eval_summary.json` (citing the point
> estimate's −70.5% while describing the bucket classifier's rejection reason) are also
> fixed alongside this.
