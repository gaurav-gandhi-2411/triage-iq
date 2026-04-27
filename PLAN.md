# PLAN — Project 5: TriageIQ — Production-Grade GitHub Issue Intelligence

## Project Overview

**Tagline:** Production-grade GitHub issue intelligence — automatic classification, duplicate detection, resolution-time prediction, and LLM-powered triage — built on real OSS data with full evaluation, deployment, and observability.

**Goal:** Build a portfolio piece that demonstrates the full ML production lifecycle (data → models → API → deployment → monitoring → evaluation) on a real-world enterprise pattern. Specifically targets Microsoft and Google interview signal: Azure/GCP deployment, evaluation rigor, scale awareness, multiple ML patterns in one coherent system.

**Audience:** Senior Data Scientist / Applied ML Engineer / ML Engineer roles at Microsoft (Bangalore + Redmond) and Google (Bangalore + Mountain View). Strong fit for Copilot, Bing, Azure AI, GitHub (Microsoft-owned), Cloud AI, and any team that touches developer productivity tooling.

**Timeline:** 8-10 days of focused work. ~50-70 hours total.

**Compute strategy:**
- Local RTX 3070 8GB: development, training, prototyping
- GitHub free Codespace (60 hours/month): redundant dev environment if needed
- Cloud deployment: Azure Container Apps free tier (Microsoft alignment) — fallback to GCP Cloud Run free tier
- Vector DB: Qdrant Cloud free tier (1GB) for production duplicate detection

**Cost target:** Under $10 over the life of the project. Free tiers everywhere; small cost only if cloud egress exceeds free tier limits.

---

## Why This Project Wins Microsoft/Google Interviews

| Concern they ask about | Demonstrated by |
|---|---|
| "Have you worked with messy real-world data?" | GitHub Issue text is messy across 5 different OSS repos |
| "Can you choose the right ML for the task?" | TF-IDF → BERT → LLM comparison with measured trade-offs |
| "Do you understand production?" | Docker, FastAPI, Azure deploy, observability, CI/CD |
| "How rigorous is your evaluation?" | Confusion matrices, calibration plots, LLM-as-judge |
| "Do you think about scale?" | Cost analysis at 1k/100k/1M req/day, scaling roadmap |
| "What's your domain breadth?" | Classification + retrieval + regression + LLM in one project |
| "Microsoft alignment specifically" | Azure deployment, vscode repo as data source, Azure OpenAI optional |
| "Google alignment specifically" | Cloud Run alternative, Kubernetes repo as data source |
| "Can you make trade-off decisions?" | TF-IDF beats BERT when X; LLM beats both when Y; here's why |

---

## What Gets Built

### Four ML Systems

**System 1: Component Classifier**
- Predict which component label applies to a new issue (e.g., "kubelet", "scheduler", "controller-manager")
- Three model tiers compared: TF-IDF + Logistic Regression / DistilBERT classification head / Few-shot LLM (Llama 3.1)
- Eval: per-class precision/recall, macro F1, calibration

**System 2: Duplicate Detection**
- For a new issue, retrieve top-5 most similar past issues
- Sentence embeddings (BGE or MiniLM) + ANN search (FAISS local, Qdrant Cloud production)
- Eval: Recall@5 against curated duplicate pairs, per-repo accuracy

**System 3: Resolution Time Predictor**
- Predict days-to-close for new issues
- LightGBM regression with text features + metadata
- Eval: MAE, RMSE, calibration plot, bias analysis per component

**System 4: LLM Triage Assistant (capstone)**
- RAG-style integration combining outputs from Systems 1-3
- Plus retrieval over component documentation
- Generates structured triage plan: "Likely component X, similar to issue #1234, expected resolution Y days"
- Eval: LLM-as-judge against curated gold-standard triage plans

### Production Layer

- **API**: FastAPI with three live endpoints + interactive OpenAPI docs
- **Containerization**: Docker multi-stage builds, sub-200MB image
- **Cloud deployment**: Azure Container Apps (primary) with GCP Cloud Run alternative documented
- **Observability**: structured JSON logging, latency histograms, prediction distribution monitoring
- **CI/CD**: GitHub Actions running tests, linting, building container, deploying on tag
- **Cost analysis**: documented estimates at three scale points

### Evaluation Framework

A first-class module, testable and reproducible:
- Holdout test sets per system (no train-test leakage)
- Automated benchmark report runs on every commit
- Comparison tables: model × accuracy × latency × cost
- LLM-as-judge for System 4 outputs
- Performance regression detection in CI

---

## Repository Layout

```
triage-iq/
├── README.md                       # portfolio document
├── PLAN.md                         # this plan
├── pyproject.toml                  # poetry config
├── requirements.txt                # pinned versions
├── docker/
│   ├── Dockerfile.api              # multi-stage API image
│   └── docker-compose.yml          # local dev
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint + test
│       ├── eval.yml                # benchmark on PRs
│       └── deploy.yml              # cloud deploy on tags
├── data/
│   ├── raw/                        # gitignored, scraped Issues
│   ├── processed/                  # cleaned datasets
│   └── README.md                   # data sources, license, schema
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_component_classifier.ipynb
│   ├── 03_duplicate_detection.ipynb
│   ├── 04_resolution_time.ipynb
│   └── 05_llm_triage_assistant.ipynb
├── src/
│   └── triage_iq/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── github_scraper.py
│       │   ├── preprocess.py
│       │   └── splits.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── classifier.py        # System 1
│       │   ├── duplicates.py        # System 2
│       │   ├── resolution.py        # System 3
│       │   └── triage.py            # System 4
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── classifier_eval.py
│       │   ├── duplicates_eval.py
│       │   ├── resolution_eval.py
│       │   ├── triage_eval.py
│       │   └── benchmark.py         # unified runner
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py               # FastAPI app
│       │   ├── schemas.py           # Pydantic models
│       │   └── middleware.py        # logging, metrics
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── logging.py
│       │   └── metrics.py
│       └── utils/
│           └── ...
├── scripts/
│   ├── 01_scrape_issues.py
│   ├── 02_build_classifier_dataset.py
│   ├── 03_train_classifier.py
│   ├── 04_build_duplicate_index.py
│   ├── 05_train_resolution_predictor.py
│   ├── 06_evaluate_all.py
│   └── 07_deploy.sh
├── reports/
│   ├── 01_data_card.md
│   ├── 02_classifier_results.md
│   ├── 03_duplicate_results.md
│   ├── 04_resolution_results.md
│   ├── 05_triage_results.md
│   ├── 06_benchmark_summary.md
│   └── 07_scale_and_cost_analysis.md
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_classifier.py
│   ├── test_duplicates.py
│   ├── test_resolution.py
│   ├── test_triage.py
│   └── test_api.py
└── docs/
    ├── architecture.md
    ├── deployment.md
    └── screenshots/
```

---

## 10-Day Plan

### Day 1: Repo Setup + GitHub API Setup + Initial Scrape

**Morning (3 hrs):**
- Create the repo at `github.com/gaurav-gandhi-2411/triage-iq`
- Set up Poetry environment, all dependencies
- GitHub Personal Access Token configuration (one-time)
- Implement `src/triage_iq/data/github_scraper.py`:
  - Authenticated requests to GitHub Issues API
  - Pagination handling
  - Rate limit awareness (5000 req/hour authenticated)
  - Exponential backoff on 429s
  - Save raw JSON per issue to `data/raw/{repo}/{issue_id}.json`
- Run scrape on `microsoft/vscode` first (smaller, faster) — target ~5,000 issues

**Afternoon (3 hrs):**
- Continue scraping in background: kubernetes/kubernetes, tensorflow/tensorflow, pytorch/pytorch, apache/airflow
- Total target: ~50,000 issues across 5 repos
- Implement `src/triage_iq/data/preprocess.py`:
  - Clean text (remove markdown, code blocks if needed, strip noise)
  - Normalize labels (each repo has different label vocabulary)
  - Extract structured fields (created_at, closed_at, resolution_hours)
  - Save processed to parquet

**Deliverable:**
- ~50k issues scraped (or as many as 24-hour rate limit allows)
- Processed parquet ready for modeling
- Initial EDA notebook with basic statistics

**Commit:** `feat: GitHub Issue scraper with rate-limit handling and preprocessing`

---

### Day 2: Data Exploration + Train/Val/Test Splits + Data Card

**Full day (6 hrs):**

`notebooks/01_data_exploration.ipynb`:
- Per-repo issue counts, label distributions
- Resolution time distributions (often log-normal, will affect modeling)
- Text length distributions (truncation strategy)
- Label cardinality per repo (Kubernetes ~30, TensorFlow ~200+ — affects classifier difficulty)
- Cross-repo label overlap analysis (mostly disjoint)
- Open vs closed ratio per repo
- Comment count distributions
- Top contributors and assignees (potential signal for resolution time)

`src/triage_iq/data/splits.py`:
- Time-based split: train on issues closed before timestamp T, test on issues closed after
- Avoids data leakage common in random splits
- Per-repo splits since each system trains per-repo or per-domain
- Stratified sampling for component classifier training (avoid losing rare components)

Write `reports/01_data_card.md`:
- Data sources, GitHub API license terms
- Schema documentation
- Known biases (large repos overrepresented, certain components dominate)
- Cleaning steps applied
- Split methodology

**Deliverable:**
- Train/val/test splits saved as parquet
- Data card published
- Exploration notebook with 15+ figures

**Commit:** `feat: data exploration, train/val/test splits, data card`

---

### Day 3: System 1 — Component Classifier (Tier 1: TF-IDF Baseline)

**Morning (3 hrs):**

Implement TF-IDF + Logistic Regression baseline:
- Per-repo classifier (different label spaces)
- Pipeline: preprocess → TF-IDF (1-2gram, max 50k features) → LogisticRegression
- Class imbalance handling: class_weight='balanced'
- Cross-validation for hyperparameter tuning

**Afternoon (3 hrs):**

Comprehensive evaluation:
- Per-class precision/recall/F1
- Macro F1 (treats all classes equally — primary metric)
- Confusion matrix (top-20 most confused pairs)
- Calibration plot (predicted probability vs actual frequency)
- Latency benchmark (predict per second, p50/p95/p99)
- Save baseline model + eval results

Write `reports/02_classifier_results.md` (will append more tiers tomorrow).

**Deliverable:**
- Baseline TF-IDF classifier per repo
- Detailed eval per repo
- Latency benchmarks

**Commit:** `feat: System 1 TF-IDF baseline classifier with full evaluation`

---

### Day 4: System 1 — Tiers 2 + 3 (DistilBERT + LLM Few-shot)

**Morning (3 hrs):**

DistilBERT classification head:
- Use `distilbert-base-uncased`
- Add classification head sized to number of components per repo
- Fine-tune on RTX 3070 (3-5 epochs, AdamW, learning rate 5e-5)
- Save best checkpoint by validation F1
- Train per-repo (transfer doesn't help across repos with disjoint labels)

**Afternoon (3 hrs):**

LLM few-shot baseline:
- Use Llama 3.1 8B via Groq
- Prompt template: "Given this GitHub issue, classify it into one of these components: [list]. Examples: [3-5 few-shots from training set]. Issue: [text]. Component:"
- Evaluate on test set (subject to rate limits — sample 200 issues per repo)
- Compare to DistilBERT and TF-IDF

Append to `reports/02_classifier_results.md`:
- Three-way comparison table
- When does each tier win? (TF-IDF for very common labels, DistilBERT for moderate-frequency, LLM for rare/novel labels)
- Cost-per-prediction analysis
- Latency: TF-IDF (1ms) vs DistilBERT (15ms) vs LLM (1200ms)
- Recommendation: production stack should use TF-IDF as fast first pass, escalate uncertain predictions to DistilBERT, escalate further to LLM only for cold-start labels

**Deliverable:**
- Three classifier models per repo
- Comparison report
- Production recommendation documented

**Commit:** `feat: System 1 DistilBERT and LLM few-shot tiers + comparison report`

---

### Day 5: System 2 — Duplicate Detection

**Morning (3 hrs):**

Build duplicate detection pipeline:
- Sentence embeddings using `BAAI/bge-base-en-v1.5` (768-dim, strong on technical text)
- Embed all closed issues per repo
- Build FAISS index (IndexFlatIP for exact, IndexHNSWFlat for production scale)
- For new issue, retrieve top-5 by cosine similarity

**Afternoon (3 hrs):**

Curate duplicate pair gold standard:
- Many GitHub Issues are explicitly labeled as duplicates (closed with "duplicate" tag, or linked via "Closes #X")
- Extract these labels to create ~500 (issue, duplicate) gold pairs across repos
- Compute Recall@1, Recall@5, Recall@10 against this gold set
- Compare BGE vs MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
- Per-repo accuracy analysis

Write `reports/03_duplicate_results.md`:
- Recall@K curves
- Latency benchmark
- Index size on disk
- Production scaling: when to switch FAISS-Flat to FAISS-HNSW or external vector DB

**Deliverable:**
- Duplicate detection system per repo
- Gold-standard evaluation set
- Detailed report

**Commit:** `feat: System 2 duplicate detection with sentence embeddings + FAISS`

---

### Day 6: System 3 — Resolution Time Predictor

**Morning (3 hrs):**

Feature engineering for resolution prediction:
- Text features: TF-IDF or BGE embeddings (use BGE — already cached from System 2)
- Metadata: number of comments at open time (always 0 for prediction at create time, but training uses post-hoc), issue length, code block count, label assigned, repo, day of week opened, time of day
- Author features: author's prior issue count, author's prior resolution time average

**Afternoon (3 hrs):**

Train LightGBM regressor:
- Target: log(resolution_hours) — log-transform handles long-tail
- 80/10/10 train/val/test by time
- Hyperparameter tuning with Optuna (50 trials)
- Output: predicted hours + 80% confidence interval (quantile regression)

Eval:
- MAE, RMSE on log-scale and original-scale
- Calibration plot (predicted vs actual)
- Bias analysis: does model underpredict for some components?
- Confidence interval coverage

Write `reports/04_resolution_results.md`.

**Deliverable:**
- Resolution time model
- Calibration analysis
- Confidence intervals

**Commit:** `feat: System 3 resolution time predictor with LightGBM and confidence intervals`

---

### Day 7: System 4 — LLM Triage Assistant + Capstone Integration

**Morning (3 hrs):**

Triage assistant design:
- Input: new issue (title + body + repo)
- Workflow:
  1. Call System 1 → get predicted component (top-3 with probabilities)
  2. Call System 2 → get top-5 similar past issues
  3. Call System 3 → get predicted resolution time + confidence
  4. Retrieve component-specific docs/READMEs (one-time scrape from repo)
  5. Pass everything as context to LLM
  6. LLM generates structured triage plan: predicted component, recommended assignee class (based on similar past issues' assignees), priority guess, expected resolution, suggested next steps

LLM prompt is structured JSON output, validated with Pydantic.

**Afternoon (3 hrs):**

Curate gold-standard triage plans:
- Take 30 closed issues
- Manually write what an ideal triage plan would have been at issue creation
- Include: correct component, who actually fixed it, time it took, what next steps were
- This becomes the eval set

LLM-as-judge methodology:
- Use Claude or GPT (or Llama 3.1 70B via Groq) to grade Triage Assistant outputs against gold
- Rubric: component_correct, similar_issues_relevant, resolution_estimate_reasonable, next_steps_actionable
- Score each output 1-5 per dimension

Write `reports/05_triage_results.md`.

**Deliverable:**
- Triage assistant integrating Systems 1-3
- LLM-as-judge eval
- Sample triage plans included in report

**Commit:** `feat: System 4 LLM triage assistant with capstone integration`

---

### Day 8: API + Containerization + Local Deployment

**Morning (3 hrs):**

FastAPI service:
- `POST /classify` — System 1 inference
- `POST /duplicates` — System 2 inference
- `POST /resolution` — System 3 inference
- `POST /triage` — System 4 full pipeline
- Pydantic request/response schemas
- OpenAPI docs auto-generated
- Health check endpoint
- Request ID tracking
- Structured logging middleware

**Afternoon (3 hrs):**

Containerization:
- Multi-stage Dockerfile (build stage + slim runtime)
- Target image size under 200MB (use python:3.11-slim base)
- Mount volume for model artifacts (don't bake into image — reload pattern)
- Test locally with docker-compose
- Smoke tests via pytest against running container

Write `docs/deployment.md` covering local development workflow.

**Deliverable:**
- Working API locally
- Docker image building and running
- All endpoints tested

**Commit:** `feat: FastAPI service + Docker containerization`

---

### Day 9: Cloud Deployment (Azure Container Apps) + Observability + CI/CD

**Morning (3 hrs):**

Azure Container Apps deployment (interview signal for Microsoft):
- Create Azure account if needed (use free tier credit)
- Set up Container Registry, push image
- Deploy as serverless container with min-replicas=0 (free when idle)
- Configure auto-scaling rules
- Set up custom domain (optional)
- Deploy and verify endpoints from public URL

Document GCP Cloud Run as alternative deployment in `docs/deployment.md` (don't actually deploy unless time permits — Azure is the priority for Microsoft signal).

**Afternoon (3 hrs):**

Observability:
- Structured JSON logging with request IDs
- Per-endpoint latency histograms (in-app middleware)
- Prediction distribution tracking (alert on drift)
- Optional: connect to Grafana Cloud free tier for dashboards
- Optional: integrate Application Insights (Azure-native)

CI/CD via GitHub Actions:
- `.github/workflows/ci.yml`: lint (ruff), type check (mypy), test (pytest) on every push
- `.github/workflows/eval.yml`: run benchmark suite on every PR, post results as comment
- `.github/workflows/deploy.yml`: build + push container + deploy to Azure on git tag

Write `docs/architecture.md` with deployment diagram.

**Deliverable:**
- Live deployed API at public URL
- Observability dashboards
- Working CI/CD pipelines

**Commit:** `feat: Azure deployment, observability, CI/CD pipelines`

---

### Day 10: Evaluation Framework + Scale Analysis + README + Polish

**Morning (3 hrs):**

Unified evaluation framework `src/triage_iq/evaluation/benchmark.py`:
- Single command runs all four systems' evaluations
- Generates consolidated report with all metrics
- Output: `reports/06_benchmark_summary.md` regenerable
- Integrated into CI: regressions fail the build
- Dashboard format: model × accuracy × latency × cost matrix

Scale analysis report `reports/07_scale_and_cost_analysis.md`:
- Cost per request at three scale points (1k/100k/1M req/day)
- Bottleneck identification per system
- What changes at each scale (e.g., FAISS → external vector DB at 100k+, model serving → batch inference at 1M+)
- Optimization roadmap

**Afternoon (3 hrs):**

README polish — portfolio-grade:
- Title + tagline + badges (CI status, deployment status, license)
- One-sentence pitch
- Architecture diagram (deployment + ML pipeline)
- Demo: animated GIF or screenshots of live API in use
- Four-system summary with key metrics
- Tech stack table
- Project structure
- Running locally + deploying to cloud
- Evaluation methodology
- Scale and cost analysis link
- Honest known limitations
- "What I learned" — 5 specific bullets with concrete examples
- Future work section (lists what would come next in production)

Final cleanup: type hints, docstrings, remove dead code.

**Deliverable:**
- Production-grade README
- Unified eval framework
- Scale analysis published
- Repo ready for portfolio review

**Commit:** `docs: production README, unified evaluation framework, scale analysis`

---

## Stretch Goals (if extra time)

### Stretch 1: GCP Cloud Run dual-deployment
- Deploy same image to both Azure and GCP
- Document differences (Azure-specific features used, vendor lock-in concerns)
- Interview talking point about cloud-agnostic design

### Stretch 2: Active learning loop
- Endpoint: `POST /feedback` lets users mark predictions as correct/incorrect
- Periodically retrain on accumulated feedback
- Demonstrates MLOps thinking

### Stretch 3: A/B testing infrastructure
- Route X% of traffic to a new model version
- Track per-version metrics
- Interview-grade production engineering signal

### Stretch 4: Public demo Streamlit app
- Beautiful UI on top of the API
- Hosted on HuggingFace Spaces (uses your existing pattern)
- Demoable in interviews

---

## Success Criteria

By Day 10, the project demonstrates:

1. **All four ML systems shipped and evaluated** — concrete metrics, not "it works"
2. **Deployed to public cloud** — Azure URL works, anyone can call the API
3. **Production observability** — logs, metrics, monitoring in place
4. **CI/CD automated** — tests run on every commit, benchmarks on every PR
5. **Honest evaluation framework** — comparison tables across model tiers, calibration plots, latency budgets
6. **Scale awareness documented** — cost at 1k/100k/1M req/day, scaling roadmap
7. **Portfolio README** — recruiters can understand value in 30 seconds, deep readers find depth

**Stretch goals:**
- Top 100 stars on GitHub (unlikely but possible if shared on Twitter/Reddit)
- Featured in a relevant newsletter or blog post
- Live demo URL bookmarked by interviewers during hiring loop

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| GitHub API rate limit hits during scrape | Authenticated tokens (5000 req/hr), exponential backoff, gradual ingestion over 24-48 hrs |
| Cloud deployment costs exceed free tier | Azure Container Apps min-replicas=0, monitor cost dashboard daily, hard $10 budget limit |
| LLM-as-judge unreliable | Use Llama 3.1 70B with structured rubric, hand-validate 20 grading decisions before automating |
| LightGBM resolution model underperforms | Fall back to "naive 7-day" baseline; document why text alone is insufficient |
| DistilBERT training OOM on 8GB | Use gradient accumulation, smaller batch size, or fall back to RoBERTa-base if needed |
| Component labels too sparse on some repos | Drop repos with < 1000 labeled issues; document and move on |
| Time overrun (typical) | Day 10 has slack; cut Stretch goals; acceptable to ship 3 systems instead of 4 if needed |

---

## Interview Talking Points (capture as you build)

Throughout development, document specific moments. These become interview gold:

1. **TF-IDF vs BERT vs LLM trade-off** — "TF-IDF achieves 81% F1 at 1ms latency. DistilBERT 87% F1 at 15ms. LLM 91% F1 at 1200ms. The right choice depends on the SLA — for a real-time triage UI, TF-IDF is correct. For batch overnight processing, LLM wins. We documented this decision in the production recommendation."

2. **Calibration matters more than accuracy** — "The classifier had 87% accuracy but was overconfident on rare components. Adding temperature scaling improved calibration by 30%, which matters when the API consumer uses confidence to decide whether to escalate to human review."

3. **Resolution time follows log-normal** — "The naive linear regression had MAE of 23 days. Log-transforming the target reduced this to 8 days. The distribution of issue resolution times is heavy-tailed because most issues close quickly but some drag for months."

4. **Duplicate detection sensitive to embedding model** — "BGE-base outperformed MiniLM by 12 percentage points on Recall@5. This is consistent with BGE's training on technical text. The trade-off is 3x larger model size and 4x slower inference."

5. **Production deployment gotchas** — "First-time Azure deployment took 6 hours. Docker layer caching saved 4 minutes per redeploy. Min-replicas=0 keeps cost under $1/month for development but adds 8s cold start. We documented both options in the deployment guide."

6. **Evaluation framework saves debugging time** — "When the second-tier classifier regressed by 3% F1, the CI flagged it within 2 minutes. Manual debugging would have caught this days later. The eval framework is now the most-used part of the codebase."

7. **LLM-as-judge has its own biases** — "Initial LLM-as-judge scoring favored verbose triage plans even when they were less accurate. We added a length penalty to the rubric. This is a real production challenge with LLM evaluation."

---

## What This Demonstrates to Interviewers

A specific 30-second pitch you can give in interviews:

> "I built TriageIQ, a production-grade GitHub issue intelligence system. Four ML systems — classification, retrieval, regression, and LLM-powered triage — running over 50,000 real issues from major OSS repos. Deployed to Azure Container Apps with full observability and CI/CD. The interesting trade-offs I documented: TF-IDF beats BERT for high-frequency labels, BGE embeddings substantially outperform MiniLM on technical text, log-transformation is essential for resolution-time prediction. The eval framework runs on every commit and blocks regressions. Total cost under $10. I can talk through any of these decisions in depth."

This pitch hits every Microsoft/Google interviewer signal:
- Specific scope ("four ML systems", not "an ML system")
- Real data ("50,000 real issues", not "a dataset")
- Production thinking ("Azure Container Apps", "observability", "CI/CD")
- Measured trade-offs ("TF-IDF beats BERT for X", not "I tried different models")
- Engineering rigor ("eval framework runs on every commit")
- Cost awareness ("under $10")

---

## What I Need From You Before Starting

**1. GitHub Personal Access Token**

Required for the scraper. Without authenticated requests, you're stuck at 60 req/hour (impossible). With a token, 5000 req/hour.

Create one:
- Go to https://github.com/settings/tokens
- Click "Generate new token (classic)"
- Scopes: `public_repo` (read-only public repos is enough)
- Expiration: 90 days
- Copy the token (starts with `ghp_...`)

Save to `.env` in the new repo: `GITHUB_TOKEN=ghp_...`

**2. Azure account confirmed**

You'll need:
- Azure account (free, requires phone verification, no credit card to sign up but credit card needed for free tier services)
- Or alternatively, GCP account (similar process)
- Confirm which one you'll use

If neither — we can defer cloud deployment to Day 9 and decide then.

**3. Repo name and visibility**

Confirm:
- Name: `triage-iq` (suggestion — concise, professional, available)
- Visibility: public (portfolio piece)
- Path on disk: `C:\Users\gaura\ml-projects\triage-iq`

**4. Time commitment**

Confirm: ~6 hours/day for 10 consecutive days. If you have any planned interruptions (interviews, travel), tell me now so we can compress the plan.

---

## Ready to Start?

When you have:
- GitHub token created
- Repo name confirmed
- Azure or GCP account ready

Reply with the token in `.env`, the repo created at GitHub, and I'll write the Day 1 kickoff message for Claude Code.

Until then, this PLAN.md should be saved at `C:\Users\gaura\ml-projects\triage-iq\PLAN.md` as the source of truth for the project.
