# ADR 0001, Record architecture decisions

**Status:** Accepted · **Date:** 2025-09-10

## Context

This project has two halves that were built weeks apart: a batch trainer and
a streaming scorer. Several decisions taken during the batch work: the
imputation strategy, the shape of the feature contract, the choice of
evaluation metric, only reveal their consequences once the streaming half
tries to load the resulting model.

By the time the streaming job was written, the reasoning behind those
earlier choices had already faded, and at least one of them was
re-litigated from scratch and got a different answer, which broke the model
load.

## Decision

Record every non-obvious architectural decision as a short, numbered,
immutable document in `docs/adr/`. Each records the context that forced the
choice, the decision, and the consequences accepted alongside it.

An ADR is never edited after acceptance. If a decision is reversed, a new
ADR supersedes it and the old one is marked as such: the history of what
was believed and when is the point.

## Note on dates

ADRs 0002–0007 carry the date the *decision* was made, which is standard ADR
practice. Several were written up retrospectively during the refactor that
produced `src/gridsmart/`, reconstructed from the code and from notes made at
the time. Where a decision's rationale was recovered rather than recorded
live, it is stated as such.

## Consequences

- A reader can reconstruct *why* the pipeline looks the way it does without
  reading every line of it.
- Reversing a decision requires writing down what changed, which is a useful
  friction.
- Small overhead per decision. Only genuinely architectural choices are
  recorded; routine implementation details stay in code comments.
