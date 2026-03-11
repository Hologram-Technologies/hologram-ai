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
7. **ALWAYS solve bugs holistically** — take a project-wide perspective instead of patching symptoms locally. Fix the root cause in the appropriate pass or abstraction layer.
8. **Prefer simpler code and smaller functions.** Functions should be short, focused, and easily testable. If a function is getting large, break it into smaller well-named helpers. Avoid complex nested logic when a flatter structure is clearer.
9. **Never commit test modules or scratch files to this repo.** Use `/tmp` for any throwaway test scripts, one-off experiments, or scratch files. Do not leave `test_*.rs`, `scratch_*.rs`, or similar files in the source tree.
10. **Clean up after yourself.** Before finishing a task, remove any scratch files, temporary debug files, and `dbg!`/`eprintln!` debugging output that were added during investigation and are not part of the final solution.

---

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

Higher-level goal: support ANY ONNX or GGUF model (focusing on ONNX first).

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
