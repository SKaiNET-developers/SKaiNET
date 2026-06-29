# Contribution Guide

SKaiNET uses the Gitflow branching model described in
[GITFLOW.adoc](GITFLOW.adoc). Keep ordinary fixes small, focused, and easy to
review.

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
do not affect user-visible behavior.

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
7. Include the standard sections: summary, motivation, proposed design,
   compatibility and migration notes, rollout plan, acceptance criteria, risks,
   open questions, and references.
8. If the proposal depends on external language or platform features, link the
   relevant upstream documents and call out stability or compiler-flag
   requirements.
9. Keep implementation PRs connected to the SKEEP. The proposal explains the
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
