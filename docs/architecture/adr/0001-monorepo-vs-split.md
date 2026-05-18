# ADR-0001: Keep triage-iq and triage-iq-ui as separate repositories

Status: Accepted
Date: 2026-05-18

## Context

TriageIQ is served by two repositories:

- `triage-iq` — Python FastAPI backend, ML models, Cloud Run deployment.
- `triage-iq-ui` — React 19 / Vite 8 SPA, Vercel deployment.

The 2026-05-18 baseline audit (`docs/audit/2026-05-18-baseline.md`, section B) examined
whether these should be merged into a monorepo. The audit found no documented decision
behind the split; it arose organically because the backend was scaffolded three weeks before
the UI (first backend commit 2026-04-27, first UI commit 2026-05-07) and the two halves were
built with unrelated toolchains.

At audit time, one concrete cost of the split was identified: the `TriagePlan` and
`SimilarIssue` types are hand-duplicated in both repos with no automated sync mechanism
(`src/triage_iq/api/schemas.py` + `models/triage.py` in Python vs `src/App.tsx:58–80`
in TypeScript). This will drift on any schema change.

## Decision

Keep the two repositories separate.

The type-drift problem is addressed separately via OpenAPI → TypeScript code generation
(tracked as a future ADR when implemented), not by merging the repos.

## Consequences

- **What changes:** None immediately. Both repos continue on their current CI/CD paths.
- **What becomes harder:** Schema changes in the backend must be accompanied by a UI
  update. Until openapi-typescript codegen is wired up, this is a manual step and a drift
  risk.
- **What becomes easier:** Backend and UI can be deployed, tested, and iterated on
  independently. Vercel's native CI for the SPA and GitHub Actions → Cloud Run for the API
  remain decoupled.

## Alternatives considered

- **Merge into a Turborepo monorepo** — rejected because the tooling overhead (nx/Turborepo
  config, retooling Vercel to point at a sub-directory, adjusting GitHub Actions paths) exceeds
  the benefit for a solo project at current scale. The two halves have different runtimes,
  different dependency graphs, and different deploy targets that are already working cleanly.

- **Shared `packages/shared-types` sub-package** — rejected for the same overhead reason;
  adds a build step and package linking that openapi-typescript codegen achieves without
  coupling the repos.

- **Generate TypeScript types from the FastAPI OpenAPI spec (`openapi-typescript`)** — this
  is the accepted mitigation for type drift. It does not require merging repos. It will be
  implemented in the UI repo via a CI pre-step and tracked in a separate ADR when done.
