# Contribution Guide

SKaiNET uses the Gitflow branching model described in
[GITFLOW.adoc](GITFLOW.adoc). Keep ordinary fixes small, focused, and easy to
review.

## Two Processes: DARC and SKEEP

Non-trivial work in SKaiNET goes through one of two processes — sometimes
both. They answer different questions:

| | DARC | SKEEP |
|---|---|---|
| **Question** | Is *this feature* the right thing to build, and is it built to the documented design? | What is the durable *shape of the codebase* going forward? |
| **Unit of work** | One operator, metric, layer, format reader, kernel strategy | One architectural decision: public API, DSL syntax, storage model, runtime/compiler integration, compatibility policy |
| **Phases / states** | Document → Assess → Research → Code (cyclical) | Draft → Accepted → Implemented (or Superseded / Rejected) |
| **Artifact** | Feature issue (`.github/ISSUE_TEMPLATE/darc_feature_request.md`) decomposed into lane sub-issues; for operators, a doc partial + `@DarcValidated` | `docs/modules/skeep/pages/NNN-short-title.adoc` + a tracking issue (`.github/ISSUE_TEMPLATE/skeep_tracking.md`) |
| **Sign-off** | A reviewer who is not the implementer | A maintainer, by moving the `Status:` field |

**Which one?** If the change trips a SKEEP trigger (next section), write the
SKEEP first and have the DARC feature issue link to it. If it doesn't, but a
maintainer six months from now would want to know *why* the change is shaped
the way it is, it's DARC. Typos, obvious one-line fixes, dependency bumps and
test-only changes need neither.

Full text: [Getting started as a contributor](https://skainet-developers.github.io/SKaiNET/skainet/contributing/getting-started.html)
and [DARC: advanced contribution workflow](https://skainet-developers.github.io/SKaiNET/skainet/contributing/darc-workflow.html)
(sources under `docs/modules/ROOT/pages/contributing/`).

## Finding Work: Issue Taxonomy

A DARC feature is one parent issue (`tracking`, `darc`) plus one native
sub-issue per *lane* — numerics research, Kotlin core, per-platform
verification, ground-truth/CI, docs, review. Every sub-issue carries exactly
one `skill:*` label (`numerics`, `kotlin-core`, `android`, `ios`, `native`,
`js`, `docs`, `review`, `design`), one `size:*` label (`xs` < 1 h, `s` a few
hours, `m` 1–2 days, `l` 3+ days), and a DARC phase label (`documentation`,
`assessment`, `research`, `coding`). `good first issue` means no prior
SKaiNET codebase knowledge is assumed.

Useful searches: `is:open label:"good first issue"`,
`is:open label:skill:android label:size:xs`, `is:open label:tracking label:darc`.

The label set is declared in `.github/labels.txt` and applied with
`.github/scripts/sync-labels.sh`; the full reference is
[Issue taxonomy](https://skainet-developers.github.io/SKaiNET/skainet/contributing/issue-taxonomy.html).

## When to Write an SKEEP

SKEEP stands for SKaiNET Evolution and Enhancement Process. It is the
project's KEEP-style process for changes that need a durable design record
before or alongside implementation.

Write an SKEEP when a change affects:

* public Kotlin APIs;
* DSL syntax or semantics;
* tensor dtype, shape, storage, or execution behavior;
* compiler, graph export, or runtime integration;
* compatibility or migration policy;
* documentation structure for a long-lived feature area.

You usually do not need an SKEEP for local bug fixes, internal refactors,
dependency bumps, test-only changes, typo fixes, or implementation details that
do not affect user-visible behavior. Additive features that sit behind an
existing interface (a new metric implementing `Metric`, a new op on an existing
backend) are DARC features, not SKEEPs — unless building them forces one of the
triggers above.

## SKEEP Procedure

1. Create a feature branch. Prefer names like
   `feature/skeep-001-tensor-collection-literals` or a similarly focused
   feature name.
2. Pick the next available three-digit number under
   `docs/modules/skeep/pages/`. Do not reuse numbers, even if a proposal is
   later rejected or superseded.
3. Create the proposal as
   `docs/modules/skeep/pages/NNN-short-title.adoc`.
4. Add the proposal to `docs/modules/skeep/nav.adoc`.
5. Add the proposal to the "Current Proposals" table in
   `docs/modules/skeep/pages/index.adoc`.
6. Start new proposals with `Status: Draft`. Use `Accepted`, `Implemented`,
   `Superseded`, or `Rejected` only when maintainers have made that decision.
   The PR that ships the implementation must also flip the status to
   `Implemented` — a proposal whose code has shipped but whose status still
   says `Draft` is worse than no status field at all.
7. Open a tracking issue from `.github/ISSUE_TEMPLATE/skeep_tracking.md`
   (title `[SKEEP-NNN]: …`, labels `skeep`, `tracking`) and put its link in
   the proposal's `Tracking issue:` header. The proposal is the durable
   record; the issue is where day-to-day coordination happens.
8. Include the standard sections: summary, motivation, proposed design,
   compatibility and migration notes, rollout plan, acceptance criteria, risks,
   open questions, and references.
9. If the proposal depends on external language or platform features, link the
   relevant upstream documents and call out stability or compiler-flag
   requirements.
10. Keep implementation PRs connected to the SKEEP. The proposal explains the
   shape of the decision; code changes prove and ship it.

SKEEP files are part of the Antora docs component. The module is registered in
`docs/antora.yml` and lives separately from the normal contributor docs so that
design proposals can evolve as their own track.

## General Contribution Expectations

We plan to expand this guide over time. For now:

* keep changes clear and well-scoped;
* prefer project-local patterns over new abstractions;
* add tests when behavior changes;
* update docs when user-facing behavior changes;
* keep generated or unrelated churn out of focused PRs.

If you are unsure whether a change needs an SKEEP, open a small draft proposal
or ask in the issue or PR before implementing the whole feature.
