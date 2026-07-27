# Contributing to hologram-ai

## Ground rules

1. Read `AGENTS.md` and `docs/rewrite-plan.md` first. The ownership
   boundaries (ADR-0001) are mandatory: compiler/runtime algorithms belong
   to `uor-r4`, container/layer machinery to `hologram`.
2. Work on a topic branch off `main` (or the active rewrite branch);
   conventional commits; keep diffs scoped.
3. Before committing, run the gates from `AGENTS.md` (fmt, clippy
   `-D warnings`, workspace tests, wasm32 portable-core checks).
4. Public API changes need: doc comments, an ADR update if semantics change,
   and error-category stability (codes are append-only ABI).

## Upstream changes

If you need an API that `uor-r4` or `hologram` does not provide, do not
copy the algorithm into this repo. Prepare a minimal PR-ready change in the
relevant sibling worktree (`../uor-r4-facade`, `../hologram-im`) and record
the contract in `docs/upstream/`. Both upstream repos have their own
contribution rules (uor-r4: no direct pushes to `main`, PR + CI gates;
hologram: house rules in its README, format changes are append-only).

## Testing expectations

- Unit tests next to each crate; hermetic (no network) by default.
- Live Hugging Face tests are opt-in (`--ignored`), never in default CI.
- Bundle/archive parsers must have hostile-input tests (truncation, hostile
  lengths, tampering, duplicates).
- Inference changes must keep the allocation census and determinism tests
  green.
