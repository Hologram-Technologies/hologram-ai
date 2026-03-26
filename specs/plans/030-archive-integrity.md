# Plan 030: End-to-End Archive Integrity for Non-LLM Models

## Context

SD UNet compiles (1634 nodes, 3.4 GB) but fails at runtime with a Gemm reading 825 floats instead of 409600. Investigation reveals:

1. No ONNX initializer has 825 elements — this is a runtime buffer, not a weight
2. The Gemm's weight IS a direct initializer (no intermediate ops)
3. The compiler's shape/constant diagnostics show no mismatches

This means the **compiled archive maps the Gemm's weight input to the wrong arena slot at runtime**. The constant-to-node mapping is corrupted somewhere in the lowering→archive→tape builder→arena chain.

## Root Issue

The pipeline from AiGraph to archive to runtime has no **end-to-end integrity verification**. Individual passes (lowering, archive writing, tape building, arena seeding) each assume the previous stage's output is correct. There's no checkpoint that validates:

- Every node's input indices map to the correct tensors
- Every constant's byte offset and size match the weight blob
- Every node's compiled shape matches its actual tensor dimensions
- The tape instruction order matches the topological execution order

## Design: Archive Validation Pass

Instead of fixing individual symptoms, add a **validation pass** that runs after compilation and before archive writing. This pass cross-checks the entire data flow.

### Phase 1: Compile-Time Validation

**File:** New `crates/hologram-ai/src/validate_archive.rs`

After `lower()` produces the `SerializedGraph` and before `PipelineWriter` assembles the archive:

```rust
pub fn validate_lowered_graph(
    sg: &SerializedGraph,
    ai_graph: &AiGraph,
    weight_bytes: &[u8],
) -> Vec<ValidationError>
```

Checks:
1. **Constant byte ranges**: For every `Deferred { source_id, byte_size }` constant, verify `source_id + byte_size <= weight_bytes.len()`
2. **Node input validity**: Every node's input index references a valid node (< total_nodes)
3. **Shape consistency**: For every node with a shape in `node_shapes`, verify `shape_product * elem_size == expected_byte_count`
4. **Gemm/MatMul weight validation**: For Gemm/MatMul nodes, verify input[1] (weight) references a Constant node (not a compute node), and its byte_size matches `k * n * 4`
5. **No dangling references**: Every constant and input is referenced by at least one compute node

### Phase 2: Runtime Validation (Debug Mode)

**File:** `hologram-exec/src/tape.rs` (behind `#[cfg(debug_assertions)]`)

Before executing the first instruction:
1. Verify all input_indices are populated in the arena
2. For `InlineGemm`/`InlineMatMul`, verify input[1] size >= `k * n * 4`

### Phase 3: Trace the Specific Bug

With the validation pass in place, recompile SD UNet and check which validation fails first. This tells us exactly where the corruption happens — lowering, archive writing, or tape building.

### Phase 4: Fix the Root Cause

Based on Phase 3 findings, fix the specific stage where the corruption occurs. Likely candidates:
- Builder input index resolution (`tid_to_idx` mapping)
- Constant ID assignment when inline+mmap params coexist
- Tape builder input slot resolution for multi-input ops (Gemm with bias)

## Implementation Order

Phase 1 → Phase 3 → Phase 4 → Phase 2

Phase 1 is the key — it gives us the diagnostic to find the bug. Phase 3 uses it. Phase 4 fixes what Phase 3 reveals. Phase 2 is a permanent safety net.

## Critical Files

| File | Change |
|------|--------|
| `crates/hologram-ai/src/validate_archive.rs` | NEW — validation pass |
| `crates/hologram-ai/src/compiler.rs` | Wire validation after lowering |
| `hologram-exec/src/tape.rs` | Debug-mode runtime validation |
