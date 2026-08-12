# Investigation: mining-channel precision characterization (product-task retrieval)

**Date:** 2026-08-11
**Status:** Investigation only — no training data built, no fine-tune run. Reports before
proposing any build, per standing instruction and this project's own ADR-0046 prior.

## Context

The 2026-08-11 k8s R@5 investigation found the product-task retriever's real headline is 39.4%
(clean subset), and separately flagged that the gap between "genuinely related" (D1's loose
audit, 83-84% on the dominant k8s channels) and "genuinely related AND a fair single-vector
retrieval test" (the stricter rubric used to build that clean eval subset, 44% on the exact same
channel) is itself the finding: roughly half of even real relationships are the wrong *shape* for
training or evaluating an embedding retriever — umbrella/tracking issues (one query, many
loosely-related targets) and causal-only citations (target cited as background, not the query's
actual topic). That gap had never been measured at the channel level before this session; this
investigation does that.

## Method

Reused this project's own established blind-labeling protocol (pre-registered criteria, written
before any pair was scored, applied by independent labelers blind to channel/source/stratum
provenance — same discipline as D1/ADR-0033 and the 2026-08-11 k8s clean-eval build):

- **VALID** — query and target share genuine, substantive topical content overlap. Default label.
- **EXCLUDE_UMBRELLA** — query or target is a checklist/tracking issue referencing many distinct
  sub-issues; no single embedding could fairly be expected to be close to any one of them.
- **EXCLUDE_CAUSAL_ONLY** — target is cited only as background/precedent for a topic the query is
  actually, substantively about something else.
- **EXCLUDE_OTHER** — any other reason (near-empty body, cross-repo confusion, mismatched
  content), with the specific reason stated.

Six independent batches (five of 30 pairs, one of 10 — matching the D1/2026-08-11 batch-size
precedent) were dispatched to independent labelers with zero visibility into channel, source,
stratum, or issue numbers — pure blind text judgment (`scripts/mining_precision_strict_audit.py`,
`reports/mining_precision_strict_labeled.json`, `reports/mining_precision_strict_audit.json`).
Sampled channels were chosen specifically because they had **never** been measured under this
stricter rubric before (only under D1's looser genuine/incidental rubric, which the 2026-08-11
finding shows systematically overstates training-fitness):

1. `vscode_dup_scrape`/`dup_comment` (n=60 of 2,242) — vscode's largest channel by far, currently
   used only for the "gate" (duplicate-detection) stratum, never for product-task training.
2. `k8s_extended_mine`/`body_related_ext` (n=50 of 200) — the other half of k8s's product-stratum
   regex mining, alongside `k8s_forward_scrape` (already strict-audited via the 150-pair clean
   eval set built 2026-08-11).
3. `vscode_body_refs`/`body_related_ext` (n=50 of 206) — vscode's regex-mined channel; only its
   stronger-pattern sibling (`body_related`, n=11) had ever been read before (D1, loose rubric).

## Finding 1 — precision, strict rubric, by channel

| Channel | n sampled | VALID | Precision | 95% CI |
|---|---|---|---|---|
| `k8s_forward_scrape`/`body_related` (k8s) | 38 | 16 | 42.1% | — |
| `k8s_forward_scrape`/`body_related_ext` (k8s) | 96 | 43 | 44.8% | — |
| `legacy_gold_v1`/`body_related` (k8s) | 16 | 7 | 43.8% | — |
| `k8s_extended_mine`/`body_related_ext` (k8s) | 50 | 27 | **54.0%** | [40.4, 67.0] |
| `vscode_body_refs`/`body_related_ext` (vscode) | 50 | 37 | **74.0%** | [60.5, 84.1] |
| `vscode_dup_scrape`/`dup_comment` (vscode) | 60 | 46 | **76.7%** | [64.6, 85.6] |

(The three k8s rows without a fresh CI are read directly from the already-committed 150-pair
clean-eval build, 2026-08-11 — same rubric, same labeling discipline, reused rather than
re-measured. `legacy_gold_v1`/`title_sim` (k8s, n=38 product-stratum) was not sampled here — it's
the same channel class ADR-0032 already measured at ~20% precision on vscode and is excluded from
consideration below on that prior alone.)

**The pattern is stark and consistent across both repos: dup_comment and the current-pattern
body-reference channels (`vscode_body_refs`, whose stronger sibling `body_related` scored 90.9%
under D1's looser rubric) outperform every k8s regex-mined channel by 20-35 points, even under
the stricter rubric.** The mechanism is structural, not incidental: `dup_comment` pairs are mined
from a `/duplicate #N` triage-bot command or an explicit "duplicate of #N" comment — by
construction a 1:1 declarative statement, immune to the umbrella-issue failure mode (1/60
EXCLUDE_UMBRELLA in this sample, vs. 3-5 per 50-pair sample everywhere else). k8s's channels mine
looser in-body citation patterns ("related to #N", "refs #N") written by the *reporter* at issue
creation time, which are far more likely to sit inside a tracking/umbrella issue or a
background-citation sentence.

## Finding 2 — realistic yield per channel, applied to available volume

| Channel | Total pairs available | Precision | Est. VALID pairs |
|---|---|---|---|
| `vscode_dup_scrape`/`dup_comment` | 2,242 | 76.7% | **~1,719** |
| `vscode_body_refs`/`body_related_ext` | 206 | 74.0% | ~152 |
| `vscode_body_refs`/`body_related` | 11 | 90.9%¹ | ~10 |
| `legacy_gold_v1`/`body_related` (vscode, incl. train_only) | 107 | 80.0%¹ | ~86 |
| **vscode total (excl. title_sim)** | **2,566** | **~76.7% weighted** | **~1,967** |
| `k8s_forward_scrape`/`body_related_ext` | 454 | 44.8% | ~203 |
| `k8s_extended_mine`/`body_related_ext` | 200 | 54.0% | ~108 |
| `k8s_forward_scrape`/`body_related` | 45 | 42.1% | ~19 |
| `legacy_gold_v1`/`body_related` (k8s, incl. train_only) | 78 | 43.8%¹ | ~34 |
| **k8s total (excl. title_sim/body_ref)** | **777** | **~47.1% weighted** | **~364** |

¹ D1's looser genuine/incidental rubric, not yet re-measured strict — carried forward as the best
available estimate, flagged as such, not blended into the headline vscode/k8s totals below without
that caveat.

**vscode has a genuinely large, previously-untapped high-precision pool (~1,967 valid pairs)
sitting almost entirely in `dup_comment` — a channel that exists today, requires no new scraping,
and has simply never been routed to product-task training** (it was deliberately siloed into the
"gate"/duplicate-detection stratum by `phase2b_merge_gold_v2.py`'s stratify design). k8s's
ceiling from existing channels is real and much lower (~364 valid pairs) — consistent with, not
contradicting, this session's earlier finding that k8s's problem is mining *precision*, not raw
pair count (4,030 raw pairs, only ~47% usable).

**Caveat, stated plainly:** these are point-estimate extrapolations from n=38-96 samples per
channel, not full censuses — the Wilson CIs above (roughly ±13-15pp half-width on the freshly
sampled channels) should be carried into any downstream volume claim, not dropped. A full census
of `dup_comment` in particular (2,242 pairs, only 60 sampled here) would tighten this
substantially before it's treated as a final number; not done in this pass, flagged as the
natural next increment if this pool becomes training-load-bearing.

## Finding 3 — k8s duplicate-label channel: checked, does not exist

Investigated whether k8s has a `dup_comment`-equivalent channel (the standout vscode lever): a
`*duplicate` label convention or an explicit "/duplicate #N" / "duplicate of #N" triage-bot
comment, mined the same way `scripts/phase2b_scrape_vscode_dups.py` mines vscode's.

- **Label census: confirmed absent, not just unsampled.** Scanned the full local corpus — all
  29,994 cached k8s raw issue JSONs, every label on every issue — for any label containing "dup".
  **Zero matches.** This is a full census over locally-cached data, not a guessed-name sweep — the
  negative is as solid as the local cache is complete.
- **Comment-text mining: genuinely untested, not ruled out.** Only 46% of local k8s issue JSONs
  (1,384/3,000 sampled) carry cached `comments_data` at all — the rest were never fetched with
  comments. Of the ones that are cached, only 9/1,384 (0.65%) contain any comment matching a
  duplicate-declaration pattern (`/duplicate`, "duplicate of #N", "dup of #N") — a much lower hit
  rate than vscode's dedicated `label:*duplicate` search converged on. This is consistent with
  k8s's Prow bot tooling not having an equivalent `/duplicate` triage convention (unlike vscode's
  purpose-built triage bot), so a comment-text scrape would be searching the general population of
  ~30,000 issues' comments for a rare, unlabeled signal — a full comment scrape (only half cached
  locally; the rest needs fresh API calls) for an apparent <1% hit rate is a real cost against a
  small, uncertain yield. **Not recommended as a near-term lever** — flagged as a checked,
  low-expected-value option, not an unexamined one.

## Recommendation (escalating before any build, per standing instruction)

1. **vscode: reclassify a portion of `dup_comment` from gate-only to product-task-eligible
   training data.** This is a methodology change to a deliberate prior design choice
   (`phase2b_merge_gold_v2.py`'s "GG-approved keep-but-stratify design" routes all `dup_comment`
   pairs to "gate"), not a mechanical action — flagging for explicit go-ahead before touching it.
   The channel is large (2,242 pairs), high-precision under the strict rubric (76.7%), and
   structurally the best training signal available in either repo. The existing `vscode_duplicate`
   held-out eval set (D1/D2, n=200) is drawn from this same channel — any pairs used for training
   must be asserted disjoint from that eval set's issue numbers before use (hard-fail check,
   scoped into the next step, not yet built).
2. **k8s: no new channel to add.** The duplicate-label lead is a confirmed dead end; comment-text
   mining is a low-yield, real-cost option not worth pursuing before the fine-tune retry. k8s's
   training pool stays bounded by its existing ~364-valid-pair ceiling across its regex-mined
   channels — smaller than vscode's, but roughly 40% larger than D1's original 264-pair pool, and
   for the first time filtered to the pairs a strict, retrieval-appropriate rubric actually
   endorses rather than D1's looser genuine/incidental bar.
3. **Apply a mechanical umbrella-issue pre-filter to any channel before training**, not just
   hand-labeling: exclude query issues with a large number of distinct outbound `#N` references
   (the exact fix ADR-0035 already recommended and never implemented) as a cheap, scalable proxy
   for the EXCLUDE_UMBRELLA failure mode found by hand in every channel sampled here. This doesn't
   replace hand-verification for the final training pool, but would cut obvious noise before
   spending labeling effort.

**Escalating before proceeding**, per this session's standing instruction: reclassifying
`dup_comment` (recommendation 1) changes a deliberate prior stratification design and needs an
explicit go before the training pool is built on top of it. If approved, next step is the
disjointness-hard-fail training-pool assembly (task #3), sized against these precision numbers
plus a full, not sampled, census of `dup_comment` for a tighter final count.
