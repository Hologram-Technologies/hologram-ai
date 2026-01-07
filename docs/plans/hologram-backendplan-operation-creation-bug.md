# Critical Bug: BackendPlan Operations Have No Input Connections

## Status

**CRITICAL**: All operations in compiled .holo files have empty `input_indices` and incorrect `output_indices`, making execution impossible.

## Problem

When compiling ONNX models (e.g., T5 encoder with 527 operations), all PlanOps in the BackendPlan have:
- `input_indices = []` (empty - no inputs!)
- `output_indices = [0]` (all writing to buffer 0)
- `kernel_id = KernelId(512)` (same ID for all operations)

Example from T5 encoder:
```
Op 0: kernel_id=KernelId(512), input_indices=[], output_indices=[0]
Op 1: kernel_id=KernelId(512), input_indices=[], output_indices=[0]
Op 2: kernel_id=KernelId(512), input_indices=[], output_indices=[0]
...
```

This causes execution to fail because:
1. GEMM kernels require 2 inputs, but `input_indices=[]` means 0 inputs are passed
2. All operations overwrite buffer 0, destroying intermediate results
3. No data flow between operations

## Root Cause Analysis

The issue is in hologram's compilation pipeline where `CompileGraph` is converted to `BackendPlan`. The operations are being created, but their buffer connections are not being set up.

### Successful Parts

✅ Input/Output metadata is correct:
- `layout_metadata.num_inputs = 2`
- `layout_metadata.num_outputs = 1`
- Input/output IDs are correctly assigned (0, 1, ...)

### Broken Part

❌ Individual PlanOp creation doesn't assign buffer indices:
- No analysis of data dependencies
- No buffer allocation for intermediate values
- No connection between operations

## Expected Behavior

For a simple model like: `input_ids -> Gather -> MatMul -> Add -> output`

The BackendPlan should have something like:
```rust
Op 0 (Gather):
  input_indices: [0]        // Read from input buffer 0
  output_indices: [2]       // Write to intermediate buffer 2

Op 1 (MatMul):
  input_indices: [2, 1]     // Read from intermediate buffer 2 and weight buffer 1
  output_indices: [3]       // Write to intermediate buffer 3

Op 2 (Add):
  input_indices: [3, weight_buffer]
  output_indices: [4]       // Final output buffer

...
```

## Where to Fix

The issue is likely in one of these locations in hologram:

### `/hologram/crates/compiler/src/pipeline.rs` (or similar)

Where `CompileGraph` is lowered to `BackendPlan`:

```rust
// CURRENT (BROKEN):
for compile_node in compile_graph.nodes() {
    let plan_op = PlanOp {
        kernel_id: select_kernel(&compile_node),
        input_indices: vec![],  // ❌ Empty!
        output_indices: vec![0], // ❌ Hardcoded!
        ...
    };
    plan.ops.push(plan_op);
}
```

**Should be:**
```rust
// Build buffer allocation map
let buffer_map = allocate_buffers(&compile_graph);

for compile_node in compile_graph.nodes() {
    // Map compile graph edges to buffer indices
    let input_indices: Vec<usize> = compile_graph
        .predecessors(compile_node)
        .map(|pred| buffer_map[&pred])
        .collect();

    let output_buffer = buffer_map[&compile_node];

    let plan_op = PlanOp {
        kernel_id: select_kernel(&compile_node),
        input_indices,
        output_indices: vec![output_buffer],
        ...
    };
    plan.ops.push(plan_op);
}
```

## Buffer Allocation Strategy

The compiler needs to:

1. **Identify all values** that need storage:
   - Graph inputs (buffers 0..num_inputs-1)
   - Intermediate values (one per operation)
   - Graph outputs (last buffers)

2. **Allocate buffer indices**:
   - Could use simple sequential allocation
   - Or optimize with liveness analysis to reuse buffers

3. **Set input_indices for each operation**:
   - Look at incoming edges in CompileGraph
   - Map source nodes to their buffer indices
   - Handle constants/weights specially

4. **Set output_indices for each operation**:
   - Most ops write to one buffer
   - Some ops (like Split) write to multiple buffers

## Example: Simple MatMul Chain

ONNX: `A (input 0) * B (input 1) = C`

CompileGraph:
```
Input "A" (node 0) -> MatMul (node 2) -> Output "C" (node 3)
Input "B" (node 1) ----^
```

BackendPlan should be:
```rust
// Buffer allocation:
// 0 = input A
// 1 = input B
// 2 = output C

layout_metadata: {
    num_inputs: 2,
    num_outputs: 1,
}

ops: [
    PlanOp { // MatMul
        kernel_id: KernelId::Gemm,
        input_indices: [0, 1],  // Read A and B
        output_indices: [2],     // Write C
    }
]
```

## Testing

After fixing, recompile T5 encoder:
```bash
cd /workspace
cargo run --release -- compile models/t5-small/encoder_model.onnx -o encoder.holo
```

Then check operation structure:
```bash
RUST_LOG=info cargo run --release -- run --config configs/test-encoder.toml
```

Should see:
```
Op 0: kernel_id=KernelId(X), input_indices=[...], output_indices=[...]
Op 1: kernel_id=KernelId(Y), input_indices=[...], output_indices=[...]
```

With non-empty input_indices and varied output_indices.

## Related Files

- `/hologram/crates/compiler/src/pipeline.rs` - Main compilation pipeline
- `/hologram/crates/compiler/src/graph/mod.rs` - CompileGraph structure
- `/hologram/crates/backend/src/plan.rs` - BackendPlan and PlanOp definitions
- `/hologram/crates/backend/src/executor.rs` - PlanExecutor (consumer of these indices)

## Success Criteria

- [ ] All PlanOps have non-empty input_indices (except pure sources like constants)
- [ ] PlanOps have varied output_indices (not all writing to buffer 0)
- [ ] PlanOps have different kernel_ids based on operation type
- [ ] T5 encoder execution succeeds without "GEMM kernel requires 2 inputs" error
- [ ] Model produces correct output tensors

## Priority

**CRITICAL** - Without this fix, no ONNX model can execute through the hologram-onnx runtime. This blocks all functionality.
