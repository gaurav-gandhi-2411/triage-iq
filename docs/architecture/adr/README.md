# Architecture Decision Records

This directory holds Architecture Decision Records (ADRs) for TriageIQ.

## Convention

TriageIQ uses a MADR-lite template. One ADR per substantive technical decision.
Trivial implementation choices (library version bumps, minor refactors, formatting) do not warrant an ADR.

## Template

```markdown
# ADR-NNNN: <title>

Status: Proposed | Accepted | Superseded by ADR-XXXX
Date: YYYY-MM-DD

## Context

Why are we deciding this? What forces are in tension?

## Decision

What did we decide?

## Consequences

What changes as a result? What becomes harder? What becomes easier?

## Alternatives considered

- **Alternative A** — rejected because …
- **Alternative B** — rejected because …
```

## Index

| ID | Title | Status | Date |
|---|---|---|---|
| [0000](0000-template.md) | Template | — | — |
| [0001](0001-monorepo-vs-split.md) | Monorepo vs split repos | Accepted | 2026-05-18 |

## Process

1. Copy `0000-template.md` to `NNNN-kebab-case-title.md` with the next sequential number.
2. Fill in Context, Decision, Consequences, Alternatives.
3. Set Status to `Proposed`.
4. Submit via PR. Status moves to `Accepted` when merged.
5. To supersede: set old ADR Status to `Superseded by ADR-XXXX`, create the new ADR.
