# Plan 006: Execution Conformance Testing

## Context

Plan 005 validated **individual kernels** — `dispatch_float()` against reference
implementations and ORT. All 187+ tests pass. But running TinyLlama end-to-end
crashes because of bugs in the **compile → lower → execute** pipeline:

1. **Shape propagation errors**: `MatMul { m:2, k:4096, n:2048 }` baked into
   the compiled graph, but the actual weight has `4096*1024` elements — the
   compiler inferred wrong dimensions.
2. **Dynamic dim sentinel failures**: `seq_len=0` sentinels not properly
   resolved at runtime.
3. **Dtype confusion**: i64 Concat bytes interpreted as f32 in diagnostics,
   hinting at real dtype issues in the execution pipeline.

Kernel tests can't catch these — they test `dispatch_float(op, inputs)` in
isolation. The bugs live between compilation and execution: wrong shapes baked
in, wrong node wiring, wrong sentinel resolution.

## Architecture: Two New Layers

### Layer E — Compile-Time Shape Consistency (cheap, no ORT)

A validation pass after concretization that catches shape errors **before** they
reach the executor. Zero runtime cost — runs only during `hologram-ai compile`.

**Checks:**
1. Weight tensor byte size matches `product(shape) * dtype.byte_size()`
2. MatMul/Gemm inner dimensions match: `lhs_shape[last] == rhs_shape[-2]`
3. Op output shapes consistent with op semantics + input shapes
4. No remaining `Dynamic` dims after concretization
5. No zero-product shapes (except intentional sentinels)

**Files:**
- `hologram-ai-common/src/opt/shape_consistency.rs` — NEW: validation pass
- `hologram-ai/src/compiler.rs` — integrate after concretization, before lowering

### Layer D — Node-by-Node Execution Validation (ORT comparison)

Full execution conformance: run the same ONNX model through both ORT and
hologram, compare **every intermediate tensor**. Zero runtime cost — lives
entirely in `hologram-ai-conformance` behind feature gate.

**Flow:**
1. Parse ONNX model, add all intermediate tensors as outputs
2. Run modified model through ORT → `HashMap<name, (shape, data)>`
3. Compile original model via hologram, capturing ONNX-name → NodeId mapping
4. Run compiled model through hologram executor, capturing all intermediate buffers
5. Map NodeIds back to ONNX names, compare node-by-node with tolerances
6. Report first divergence + summary

**Changes needed:**

In hologram base (`hologram-exec`):
- `buffer/arena.rs` — add `snapshot()` method (non-destructive clone, test-only)
- `eval/executor.rs` — add `KvExecutor::execute_with_intermediates()`, feature-gated

In hologram-ai:
- `compiler.rs` — add `compile_with_debug_info()` returning archive + name map
- `validate.rs` — extend with execution validation reporting

In hologram-ai-conformance:
- `ort_runner.rs` — add `run_onnx_with_intermediates()` via protobuf manipulation
- `exec_comparator.rs` — NEW: node-by-node comparison with tolerances
- `tests/exec_conformance.rs` — NEW: integration tests with multi-node ONNX models

## Runtime Performance Guarantees

- **Zero production overhead**: All new code lives in test-only methods and
  compile-time passes. The hot execution path is never touched.
- **O(1) lookup preserved**: `BufferArena` remains `HashMap<NodeId, Cow<'a, [u8]>>`
- **Zero-copy preserved**: `Cow::Borrowed` for weight slices — unchanged
- **Feature-gated**: `execute_with_intermediates()` behind `profile` feature
- **Archive unchanged**: `compile_with_debug_info()` returns debug map alongside
  the archive — the archive itself is byte-identical to `compile()` output

## Implementation Sequence

### Phase 1: Layer E (hologram-ai only, no cross-repo changes)

1. Create `shape_consistency.rs` pass (implements `Pass` trait)
2. Integrate into `compiler.rs` after concretization, before lowering
3. Add tests that compile known-bad graphs and verify errors are caught

### Phase 2: Layer D executor changes (hologram base repo)

4. Add `BufferArena::snapshot()` in `arena.rs`
5. Add `KvExecutor::execute_with_intermediates()` in `executor.rs`

### Phase 3: Layer D conformance harness (hologram-ai repo)

6. Add `compile_with_debug_info()` to `ModelCompiler`
7. Add ORT intermediate capture in `ort_runner.rs`
8. Add `exec_comparator.rs` — node-by-node comparison
9. Add integration tests in `exec_conformance.rs`

## Verification

1. `cargo test -p hologram-ai` — shape consistency catches known-bad shapes
2. `cargo test -p hologram-ai-conformance --features=conformance` — execution conformance passes
3. Compile TinyLlama ONNX with shape consistency — identifies the specific bugs
4. Fix shape bugs, re-run — TinyLlama executes without panics
5. `cargo clippy -- -D warnings` clean in both repos

## Design Decisions

- **Separate method** rather than flag on hot path — zero overhead in production
- **Debug map alongside archive** rather than modifying format — production builds unaffected
- **ORT intermediate capture via protobuf modification** — standard approach, no ORT API extensions
- **Layer E first** — catches most bugs cheaply without cross-repo changes, immediate value
