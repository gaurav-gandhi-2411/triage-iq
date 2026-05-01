# Data Sources

## Raw Issues (`data/raw/`)

Scraped from GitHub Issues API (authenticated, public repos only).

| Directory | Source | License |
|---|---|---|
| `microsoft_vscode/` | github.com/microsoft/vscode | MIT |
| `kubernetes_kubernetes/` | github.com/kubernetes/kubernetes | Apache-2.0 |
| `tensorflow_tensorflow/` | github.com/tensorflow/tensorflow | Apache-2.0 |
| `pytorch_pytorch/` | github.com/pytorch/pytorch | BSD-style |
| `apache_airflow/` | github.com/apache/airflow | Apache-2.0 |

GitHub Issues are public data under each repo's open-source license. API usage governed by [GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service).

## Schema

Each raw JSON file (`{issue_number}.json`) contains the GitHub Issues API response plus an added `comments_data` key:

```
id, number, title, body, labels[], state, created_at, closed_at,
user.login, assignees[], comments (count), comments_data[]
```

## Processed Parquet (`data/processed/`)

Files: `issues_{repo}.parquet`

Columns:
- `id`, `number`, `title` — identifiers
- `body_clean` — cleaned text (HTML comments removed, whitespace normalized, max 10k chars)
- `code_blocks` — extracted fenced code blocks (max 5k chars)
- `labels_raw` — list of label strings
- `component`, `type`, `priority` — normalized label facets
- `state` — open / closed
- `created_at`, `closed_at` — UTC timestamps
- `resolution_hours` — float, NaN for open issues
- `author` — GitHub login
- `num_comments`, `num_assignees` — integers

## Known Biases

- Large, high-activity repos overrepresented (vscode, kubernetes have 50k+ issues)
- Issues without labels cannot be used for component classification training
- Resolution time only computable for closed issues (~60-70% depending on repo)
- Bots (e.g., k8s-ci-robot) appear as authors — not filtered at this stage
