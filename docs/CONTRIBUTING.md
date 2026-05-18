# Contributing to TriageIQ

## Change Documentation

Every substantive PR must include at least one of the following:

### (a) Update `CHANGELOG.md`

Add a line under `## [Unreleased]` in the appropriate section (`Added`, `Changed`,
`Fixed`, `Security`, etc.). One line per logical change. State *what* changed and,
if non-obvious, *why*.

```markdown
## [Unreleased]

### Fixed
- Resolution predictor now handles issues with null `created_at`; previously raised
  `KeyError` at feature-engineering time (`scripts/09_train_resolution.py:142`).
```

### (b) Add or update an ADR

If the PR makes a substantive **technical decision** — choosing between approaches,
adopting a new tool, changing an architectural boundary, or reversing a prior decision —
create or update an ADR in `docs/architecture/adr/`. Copy `0000-template.md`, assign
the next sequential ID, fill in Context / Decision / Consequences / Alternatives.

Examples of decisions that warrant an ADR:

- Switching from TF-IDF to a fine-tuned encoder as System 1.
- Adding a new supported repository (e.g., tensorflow/tensorflow).
- Changing the LLM provider from Groq to another.
- Introducing openapi-typescript codegen for UI type sync.
- Adding or removing a Prometheus metric.

### (c) Both

Schema changes, new ML systems, and deploy-target changes typically require both a
CHANGELOG entry and an ADR.

---

## Exemptions

The following PR types are exempt from change documentation:

- Dependency version bumps (Dependabot PRs).
- Typo or formatting fixes in existing docs.
- Ruff / mypy lint-only fixes with no behavior change.
- `.gitignore` additions.

---

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): message`. Types: `feat`, `fix`, `docs`, `chore`, `style`, `test`,
`refactor`, `perf`. Include the PR number in merge commit messages.
