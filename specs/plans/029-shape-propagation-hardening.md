# Plan 029: Compiler Shape Propagation + Runtime Meta Computation

## Context

SD UNet compiles (1634 nodes, 3.4 GB) but fails to execute because MatMul ops have wrong baked dimensions (`k=59136` instead of `768`). The root cause: the compiler's shape propagation pipeline doesn't fully resolve all intermediate tensor shapes for non-LLM models, causing `matmul_recipes()` to extract wrong k/m/n from empty or flattened shapes.

The runtime `shape_resolve` infrastructure (Plan 028) is in place and working, but it can only help when input metas are correct — which requires either correct compiled shapes OR runtime meta propagation.

**Two complementary fixes needed:**
1. **Compiler (hologram-ai)**: Strengthen shape propagation so all shapes are correct at lowering time
2. **Runtime (hologram base)**: Compute output metas from actual dispatch results so correct shapes propagate even when compiled shapes are imperfect

---

## Phase 1: Strengthen Compiler Shape Propagation (hologram-ai)

### 1.1 ShapeHealing for MatMul/Conv/Transpose

File: [shape_heal.rs](crates/hologram-ai-common/src/opt/shape_heal.rs)

ShapeHealing currently handles Reshape, Squeeze, elementwise, and identity ops. Add inference for:
- **MatMul**: output = `[batch..., M, N]` from input shapes `[..., M, K]` x `[..., K, N]`
- **Conv2d**: output = `[N, Co, Ho, Wo]` from convolution arithmetic
- **Transpose**: output = permute(input_shape, perm)
- **Concat**: sum along axis

### 1.2 Extra shape propagation after ConstantEvaluation

File: [compiler.rs](crates/hologram-ai/src/compiler.rs) (`post_concretization_repair`)

After `ForceConcretize + ConstantEvaluation + ConstantFolding`, run one more `AggressiveShapePropagation`. ConstantEvaluation may fold Shape nodes into concrete values that enable further shape inference.

### 1.3 Validate MatMul dimensions at lowering

File: [strategy.rs](crates/hologram-ai-common/src/lower/strategy.rs)

In `matmul_recipes`, after computing m/k/n, cross-check:
- k must divide input[1]'s param byte-size (when available)
- k should be reasonable (< 16384 for typical models)
- Log warning when k/m/n look suspicious

### Verification
- `cargo test` — all existing tests pass
- Compile SD UNet, check that MatMul k values are now correct (768 instead of 59136)

---

## Phase 2: Runtime Meta Computation (hologram base)

### 2.1 Seed constant metas into arena

File: `hologram-exec/src/mmap/mod.rs`

When seeding constants, set `TensorMeta` from `SerializedGraph.node_shapes` (already available). This gives weight tensors correct N-D metas so `resolve_matmul_dims` can use B's shape to determine k/n.

### 2.2 MatMul returns computed output meta

File: `hologram-exec/src/tape.rs`

After `dispatch_matmul_into`, compute output meta from resolved dims and return via `DispatchResult::InOutBufWithMeta(out_meta)`. This ensures downstream ops get correct metas.

### 2.3 Norm/Softmax return input meta as output meta

For element-preserving ops, return input meta as output meta via `InOutBufWithMeta`.

### Verification
- All hologram-exec tests pass
- TinyLlama generates coherent text
- `tape_softmax` regression guard

---

## Phase 3: SD UNet E2E Validation

- Execute SD UNet with dummy inputs, verify output `[1, 4, 64, 64]` all-finite
- Update `sd_unet_e2e.rs` to assert execution
- Verify no regression on TinyLlama, BERT, ResNet

---

## Critical Files

| File | Repo | Change |
|------|------|--------|
| `hologram-ai-common/src/opt/shape_heal.rs` | hologram-ai | MatMul/Conv/Transpose shape inference |
| `hologram-ai/src/compiler.rs` | hologram-ai | Extra AggressiveProp after ConstEval |
| `hologram-ai-common/src/lower/strategy.rs` | hologram-ai | MatMul dimension validation |
| `hologram-exec/src/mmap/mod.rs` | hologram | Constant TensorMeta seeding |
| `hologram-exec/src/tape.rs` | hologram | MatMul/norm output meta computation |
