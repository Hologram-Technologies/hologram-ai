# AGENTS.md

This document provides guidance for automated agents operating in **`hologram-ai`**.

---

## Repository Purpose

`hologram-ai` is a **library** repository in the ecosystem.

Standards version: `2026.03`

---

## Repository Structure

```
specs/
  docs/         — project documentation
  adrs/         — architecture decision records
models/         — test models for development (TinyLlama ONNX, etc.)
```

> **Models directory**: Test models live in a sibling directory `../hologram-ai/models` (i.e., this repo's `models/` subdirectory). Do not search for models in the repository root or elsewhere.

---

## Rules for Agents

1. **SOLVE PROBLEMS IN THE MOST PRODUCTION-READY WAY POSSIBLE.** Always take a project-wide perspective — solutions must be robust, correct, and ready to ship. No hacks, no shortcuts, no "good enough for now." Every change should be something you'd confidently deploy to production.
2. **ZERO RUNTIME PERFORMANCE PENALTIES.** Never introduce unnecessary allocations, copies, indirections, or overhead in hot paths. Prefer zero-cost abstractions, compile-time evaluation, and in-place operations. If a solution has a runtime cost, justify it explicitly and minimize it. Profile-guided decisions over guesswork.
3. Follow the architecture standards defined in the architecture repo
4. Do not modify files outside this repository unless explicitly instructed
5. Run `cargo clippy -- -D warnings` before committing Rust changes
6. Use a consistent naming prefix for all crate names
7. **ALWAYS solve problems holistically** — see the Problem-Solving Philosophy section below.
8. **Prefer simpler code and smaller functions.** Functions should be short, focused, and easily testable. If a function is getting large, break it into smaller well-named helpers. Avoid complex nested logic when a flatter structure is clearer.
9. **Never commit test modules or scratch files to this repo.** Use `/tmp` for any throwaway test scripts, one-off experiments, or scratch files. Do not leave `test_*.rs`, `scratch_*.rs`, or similar files in the source tree.
10. **Clean up after yourself.** Before finishing a task, remove any scratch files, temporary debug files, and `dbg!`/`eprintln!` debugging output that were added during investigation and are not part of the final solution.
11. **Never leave TODOs, placeholders, or stubs.** Do not commit `TODO`, `FIXME`, `HACK`, placeholder implementations, or stub functions. Either implement the feature fully or do not add it. If work is deferred, track it in `specs/SPRINT.md` — not in source code comments.
12. **Never use `.unwrap()` in production code.** Use `.expect("descriptive message")` instead so that failures produce actionable context. This applies to all code — library and binary crates alike. The only exception is throwaway scratch files in `/tmp`.
13. **Keep site docs up to date.** When making structural changes, adding new features, changing APIs, or modifying the compilation pipeline, update the relevant documentation in `specs/docs/` (e.g., `architecture.md`, `lowering.md`, `data-model.md`, `cli.md`). Docs must reflect the current state of the code — stale documentation is worse than no documentation.
14. **Never suppress `clippy::too_many_arguments`.** If a function triggers this lint, refactor it to accept a params/config struct using the builder pattern. Do not add `#[allow(clippy::too_many_arguments)]`.

---

## Problem-Solving Philosophy

**Think like a principal systems architect, not a patch author.** When encountering bugs, build failures, or design issues, do not apply narrow band-aid fixes that address only the immediate symptom. Instead:

1. **Diagnose the root cause.** Before writing any fix, understand *why* the problem exists. Trace it back to the underlying design decision, missing abstraction, or architectural gap that allowed it to surface.

2. **Assess the blast radius.** Ask: is this a one-off mistake, or a symptom of a systemic pattern? If the same class of bug could occur elsewhere, the fix must address the class, not just the instance.

3. **Propose a production-ready solution.** Design the fix as if this code will run in production under load, across tenants, for years. Consider concurrency, error propagation, backward compatibility, and operational debuggability. A correct fix that is fragile or hard to reason about is not production-ready.

4. **Refactor when the problem is structural.** If the root cause is that a function does too much, a type doesn't enforce its invariants, or responsibilities are in the wrong module — fix the structure. Moving code, splitting types, introducing a new abstraction, or changing an API boundary are all valid (and often necessary) responses to a bug.

5. **Never play whack-a-mole.** If you find yourself fixing the same kind of issue in multiple places with small, repetitive patches, stop. That pattern means the underlying design needs to change. Propose the holistic fix, not N localized patches.

6. **Validate the fix is complete.** After implementing a solution, check whether the same class of issue exists anywhere else in the codebase. A fix that leaves known instances of the same bug untouched is incomplete.

### Anti-patterns (never do these)

- **Band-aid fixes.** Patching the symptom where it surfaces without understanding
  why it happens. If you find yourself adding a special case, a hardcoded fallback,
  or a "just check for this" guard — stop. You are treating symptoms, not causes.
- **Whack-a-mole.** Fixing one instance of a problem only to have the same class
  of problem appear elsewhere. If the fix is not general, it is not a fix.
- **Solving at the wrong layer.** Adding complexity to a downstream consumer when
  the real gap is in an upstream pass, abstraction, or data structure.
- **Accumulating small patches.** Ten small "just this one thing" changes that
  together create an unmaintainable mess. One correct change at the right
  abstraction layer is always better than many scattered patches.

### Required methodology

For every non-trivial problem, follow this sequence:

1. **Step back.** Do not immediately jump to the code where the error occurs.
   Understand the broader system: what pipeline stage are we in? What are the
   inputs, outputs, and invariants of this layer?
2. **Trace the root cause.** Follow the data flow upstream. Where did the bad
   state originate? Often the symptom is 3–4 layers removed from the cause.
3. **Identify the right abstraction layer.** The fix belongs where the invariant
   should be established — not where it is consumed. If a shape is wrong at
   lowering time, the fix is probably in shape propagation, not in the lowering
   code.
4. **Design a general solution.** Will this fix handle all instances of this
   problem class, not just the one you're looking at? If not, generalize.
5. **Implement once, correctly.** A single well-placed change that solves the
   class of problems. No scattered guards, no defensive checks at every
   call site, no "just in case" fallbacks.
6. **Verify end-to-end.** Confirm the fix works through the full pipeline, not
   just in isolation. Check that it doesn't break other paths.

### This applies to everything

This philosophy is not just for bugs. It applies equally to:

- **New features**: Design them to fit cleanly into the existing architecture.
  Don't bolt things on the side.
- **Refactors**: Improve the system's structural integrity, don't just shuffle
  code around.
- **Performance work**: Understand the actual bottleneck before optimizing. Never
  optimize what you haven't measured.

---

## Conformance Testing Mandate

**Any runtime bug or connected-op bug MUST have a conformance test before the fix is applied.**

The `hologram-ai-conformance` crate (`crates/hologram-ai-conformance/`) provides the testing harness:
- `ort_runner::onnx_builder` — build multi-node ONNX models programmatically (matmul, concat, gemm, softmax, rms_norm, etc.)
- `ort_runner::runner::run_onnx_all_outputs` — run a model through ORT and collect all outputs
- `exec_conformance.rs` — integration tests that compile ONNX → hologram, execute, and compare against ORT

### When you find a runtime bug

1. **Reproduce it in `exec_conformance.rs`** — add a test that calls `compile_and_execute()` and compares against `run_onnx_all_outputs()`. The test should FAIL until the bug is fixed.
2. **If the ONNX builder lacks the required graph type** — add a builder function to `ort_runner/onnx_builder.rs` (e.g., `concat_along_axis`, `batched_matmul_4d`, `scaled_dot_product_attention`).
3. **Never fix the bug without a test** — a fix without a failing test is a guess.
4. **Do NOT write one-off dispatch patches or runtime guards** — trace the root cause to the compiler (shape propagation, lowering strategy, concat axis calculation) and fix the general case there.

This applies to:
- Shape tracking bugs (wrong dimensions after transpose, reshape, concat)
- Connected-op bugs (where output of one op feeds into another with wrong shape)
- Execution dispatching bugs (wrong batched matmul, wrong broadcast expansion)

The `shape_chain.rs` tests in hologram base (`hologram/crates/hologram-exec/tests/shape_chain.rs`) cover dispatch-level correctness for individual ops. The `exec_conformance.rs` tests cover full ONNX pipeline correctness end-to-end (compile + execute).

## Fixture-Driven Regression Testing

When a full model (TinyLlama, etc.) produces incorrect output, use this
workflow to isolate the bug into a minimal, reproducible fixture.

### Workflow: Model failure → Minimal fixture → Conformance test → Fix

#### Phase 1: Identify the failing subgraph
1. Compile the full model and run it (`cargo test -p hologram-ai --features e2e`)
2. Examine diagnostics: shape warnings, NaN detector, output quality
3. Identify the suspect op or op-chain (e.g., GQA kernel, SwiGLU, RoPE)
4. Determine the path: **ONNX-decomposed** (individual ops) or **fused kernel** (GGUF `AiOp`)

#### Phase 2: Create a minimal fixture file
Generate a `.onnx` fixture in `crates/hologram-ai-conformance/fixtures/`:

- Add a Python generator function to `fixtures/generate.py` that builds the
  minimal ONNX subgraph (typically 2–20 nodes) reproducing the failure
- Use small dimensions (seq=4–8, hidden=16–64) for fast CI execution
- Run `python3 crates/hologram-ai-conformance/fixtures/generate.py` to create the file
- The fixture file serves as the ORT reference — loaded via `fixtures::load("name")`

#### Phase 3: Write the conformance test

**For ONNX-path bugs (decomposed ops):**
```rust
let onnx_bytes = fixtures::load("my_fixture").expect("run generate.py");
let ort_out = run_onnx_all_outputs(&onnx_bytes, inputs)?;
let holo_out = compile_and_execute(&onnx_bytes, &inputs);
assert!(compare_outputs(&holo_out, &ort_out[0].data, tol).passed);
```

**For fused-kernel bugs (GGUF path):**
Build an `AiGraph` with the fused `AiOp` and compile via `ModelSource::AiGraph`.
Load the decomposed ONNX fixture as the ORT reference:
```rust
// ORT reference from decomposed fixture
let onnx_bytes = fixtures::load("gqa_fused_reference").expect("run generate.py");
let ort_out = run_onnx_all_outputs(&onnx_bytes, inputs)?;

// Fused kernel via AiGraph
let graph = build_gqa_aigraph(n_q, n_kv, head_dim, seq, causal);
let holo_out = compile_and_execute_aigraph(graph, &inputs);
assert!(compare_outputs(&holo_out, &ort_out[0].data, tol).passed);
```

#### Phase 4: Fix and verify
Fix the root cause following the Problem-Solving Philosophy. The test MUST pass.

### Available infrastructure

| Tool | Purpose | Location |
|------|---------|----------|
| `fixtures::load(name)` | Load `.onnx` fixture from file | `ort_runner.rs` |
| `generate.py` | Generate fixture `.onnx` files | `fixtures/generate.py` |
| `onnx_builder::*()` | Build ONNX models in Rust (for dynamic or parameterized tests) | `ort_runner.rs` |
| `run_onnx_all_outputs()` | Run ONNX through ORT, collect outputs | `ort_runner.rs` |
| `compile_and_execute()` | Compile ONNX → hologram, execute | `exec_conformance.rs` |
| `compile_and_execute_aigraph()` | Compile `AiGraph` → hologram, execute (fused kernel path) | `exec_conformance.rs` |
| `ModelSource::AiGraph` | Compile directly from `AiGraph` (skips ONNX import) | `compiler.rs` |
| `compare_outputs()` | Compare f32 slices with tolerance | `tolerance.rs` |

### Fixture conventions

- Fixtures live in `crates/hologram-ai-conformance/fixtures/*.onnx`
- Each fixture has a generator in `generate.py` and at least one test in `exec_conformance.rs`
- Use `_fused_reference` suffix for fixtures that serve as ORT reference for fused kernel tests
- Keep dimensions small (< 100 KB per fixture) for fast CI

<!-- ARCHON:MANAGED:BEGIN -->
## Ecosystem Rules

These rules apply to all repositories in the Hologram ecosystem.

### Naming
- Use the `hologram-` prefix for all crate names (never `holo-`)
- Follow kebab-case for crate and repo names

### Code Quality
- Run `cargo clippy -- -D warnings` before committing Rust changes
- Run `cargo fmt --check` before committing Rust changes
- All public APIs must have documentation comments
- No `unwrap()` in library code — use proper error handling
- Use traits at API boundaries; use macros to eliminate boilerplate
- Functions with >3 parameters must use the builder pattern
- **Never use `#[allow(clippy::too_many_arguments)]`** — if clippy complains about too many arguments, refactor the function to accept a struct with the builder pattern instead of suppressing the lint
- Use `thiserror` for library errors; `anyhow` only in binaries
- See ADR-0007 for the full set of Rust development standards

### Architecture
- Follow ADR decisions from `hologram-architecture`
- Declare contracts in `hologram.repo.yaml`
- Do not introduce cross-repo dependencies without an ADR

### Documentation
- Keep `specs/docs/architecture.md` up to date with structural changes
- Update `AGENTS.md` when adding new conventions or rules
<!-- ARCHON:MANAGED:END -->

## Shape System Strategy

The hologram-ai compiler must resolve all tensor shapes to concrete values before
lowering to hologram's byte-domain graph. The shape pipeline is:

```
ONNX/GGUF symbolic dims
  → AiGraph (DimExpr: Var, Dynamic, Concrete)
  → ShapePropagation (forward inference from input shapes)
  → DataPropagation (evaluate shape-computation subgraphs)
  → ShapePropagation (second pass: use known_i64_values for Reshape/Expand)
  → concretize_all_dims (Var → upper bounds, Dynamic → 1)
  → ShapeHealing (infer remaining empty shapes from op semantics)
  → lower (emit full tensor shapes into compiled graph)
```

### Key principles

1. **Full shapes on every compiled node.** Every node in the compiled
   hologram::Graph must have a correct multi-dim shape in the shape_map.
   The runtime uses these for batched matmul dispatch and output allocation.
2. **Fail loud at compile time, not silently at runtime.** If a MatMul
   dimension can't be determined, the compiler should error — never emit
   a fallback like m=1 that will crash at runtime.
3. **Shape healing as a safety net.** After concretization, a final pass
   infers any remaining empty shapes from op semantics, element count
   conservation, and input shapes. This is the last resort before lowering.
4. **Don't fix individual ops in isolation.** When a new shape bug surfaces,
   first check: (a) does ShapePropagation handle this op? (b) does
   DataPropagation track its values? (c) does ShapeHealing cover it?
   Fix the gap in the appropriate pass, not in the lowering code.
5. **Prefer simple implementations over complex ones.** Solve problems at
   the right abstraction layer with the minimum code needed. Avoid building
   elaborate inference machinery when a simpler approach (e.g., re-running
   an existing pass after concretization) achieves the same result.

### Milestone: TinyLlama end-to-end

The defined goal is to compile TinyLlama-1.1B (ONNX) to a `.holo` archive
and run it with a joke prompt to produce coherent English text. This validates
the full pipeline: import → optimize → concretize → lower → execute.

Higher-level goal: support ANY ONNX or GGUF model.

**Priority: ONNX-first, GGUF second.** The ONNX pipeline is the primary focus —
all shape propagation, lowering, and conformance testing should target ONNX models
first. GGUF support is secondary and should not drive architectural decisions or
block ONNX progress.

### What the runtime needs from compiled shapes

- `FloatOp::MatMul { m, k, n }`: Only last-2-dim hints. The runtime uses
  `input_shapes` from the compiled graph to dispatch batched matmul for ≥3D
  tensors. **Correct shapes on MatMul inputs are more important than m/k/n.**
- `FloatOp::Softmax/RmsNorm/etc { size }`: Last-dim size. Runtime resolves
  size=0 from actual input shape.
- Reshape/Transpose/Identity: Passthrough — runtime just copies bytes.

### Holistic compilation strategy

The compiler must solve two systemic problems:
1. **Shape resolution**: all tensor shapes must be concrete before lowering
2. **Runtime capability gaps**: hologram's runtime supports only 1-D broadcasting
   (element-wise with repeat), NOT N-D tensor broadcasting that ONNX models
   rely on for causal masks, attention scores, RoPE, etc.

**The core principle: evaluate everything possible at compile time.**
Instead of fixing individual ops (whack-a-mole), the compiler uses a
layered pipeline that progressively eliminates runtime work:

#### Phase 1: Shape resolution (pre-concretization)
ShapeProp → DataProp → ShapeProp.
Works with symbolic dims. Gets as far as possible.

#### Phase 2: Concretization
Var → upper bounds, Dynamic → 1. Clear stale intermediate values.
Re-run: AggressiveShapeProp → DataProp → AggressiveShapeProp → ConstFold → DeadNode.

#### Phase 3: Compile-time tensor evaluation (post-concretization)
**This is the key phase.** After concretization, many subgraphs become
fully constant (all inputs are materialized AiParam constants). The
**ConstantEvaluation pass** evaluates these nodes at compile time:

- Element-wise arithmetic (Add, Sub, Mul, Div) with N-D broadcast
- Comparisons (LessOrEqual, Less, Greater, Equal) with N-D broadcast
- Logical ops (And, Or, Not) with N-D broadcast
- Expand (broadcast to target shape)
- Where (conditional selection with N-D broadcast)
- Cast (dtype conversion)
- Reshape, Transpose, Concat, etc.

The evaluator uses actual tensor data with proper N-D broadcasting
(numpy-style). Results are stored as AiParam::Inline constants.
ConstantFolding then removes the redundant nodes.

This eliminates entire subgraphs like:
- **Causal mask**: Range → Unsqueeze → LessOrEqual → And → Expand → Where
- **Position embeddings**: Gather from constant frequency tables
- **Shape computation**: Shape → Gather → Concat chains

**Rule: when a runtime op fails, first check if it could have been
evaluated at compile time. If ALL inputs are constants, add the op
to ConstantEvaluation. Only add runtime support as a last resort.**

#### Phase 4: Lowering validation
Before emitting a node, verify:
- Element counts match between input and output for reshape-like ops
- The runtime op (FloatOp) can handle the actual input shapes
- All required shape parameters are non-zero

#### Bug investigation protocol
When a new runtime failure occurs:
1. **Trace the AiGraph node** → identify the ONNX op and its inputs
2. **Check if inputs are constants** → if yes, add to ConstantEvaluation
3. **Check if the dispatch is correct** → e.g., Expand ≠ Reshape
4. **Check lowering strategy** → verify shape parameters are correct
5. **Only then** consider runtime changes (in hologram base crate)

<!-- ARCHON:CONTEXT:BEGIN -->
## Ecosystem Context (auto-generated by archon)

See [`.archon/context.md`](.archon/context.md) for full dependency graph, public API surface, and contract details for this repo.
<!-- ARCHON:CONTEXT:END -->
