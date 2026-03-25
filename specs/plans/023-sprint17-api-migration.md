# Plan 023: Adapt to hologram Sprint 17 API Removals

## Context

hologram Sprint 17 (Plans 014 + 015) removed the deprecated `KvExecutor`
dispatch path, all `execute_plan*` functions, and intermediate capture. The
canonical execution API is now tape-only (`build_tape_from_plan` +
`execute_tape`). hologram-ai was already 95% migrated (Plan 022 moved HoloRunner
to tape), but 3 remaining call sites referenced removed APIs.

## Changes (4 files)


### 1. `run_with_shape_context()` — migrated to tape API
- **File:** `crates/hologram-ai/src/compiler.rs`
- Replaced `hologram::execute_plan(&plan, inputs)` with
  `build_tape_from_plan(&plan)` + `execute_tape(&tape, &plan, inputs)`
- Removed `#[allow(deprecated)]` and stale doc comments about legacy path
- Dynamic shapes handled by tape executor's `resolve_size()` + `infer_matmul_k()`

### 2. Deleted `tinyllama_node_divergence_finder()`
- **File:** `crates/hologram-ai-conformance/tests/exec_conformance.rs`
- Was `#[test] #[ignore] #[cfg(feature = "profile")]`
- Depended on `hologram::execute_plan_with_intermediates` (removed)
- Node-level debugging now requires adding probe output nodes to graph

### 3. Deleted `tinyllama_node_inspector()`
- **File:** `crates/hologram-ai-conformance/tests/exec_conformance.rs`
- Was `#[test] #[ignore] #[cfg(feature = "profile")]`
- Depended on `hologram::execute_plan_with_intermediates_and_shape_hints` (removed)

### 4. Fixed `mini_transformer_variable_seq_len_runs` test
- **File:** `crates/hologram-ai/tests/mini_fixture.rs`
- Was compiling once and running at multiple seq_len via `run_with_shape_context`
  (relied on `execute_plan` dynamic shape resolution)
- Now compiles separately for each seq_len with `seq_len_override`, matching
  the tape executor's compile-time shape baking
- Gated `workspace_path` and `cosine_similarity` helpers behind `#[cfg(feature = "e2e")]`
  to eliminate dead-code warnings

### 5. Updated SPRINT.md
- Marked intermediate capture API as removed in Sprint 17
- Added migration entries under P2c Integration

## Not Needed

- **HoloRunner** — already fully on tape API (Plan 022)
- **CustomOpRegistry** — hologram-ai lowers all ops to native `GraphOp`
- **KvExecutor / execute_bytes / execute_file** — not referenced
- **shape_propagate / dirty_bits / IntermediateCapture** — not imported
