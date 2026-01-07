# Fix BackendPlan Input/Output Metadata from OperationGraph

## Problem

When compiling ONNX models via hologram-onnx, the T5 encoder (which has 2 inputs and 1 output) produces a .holo file with incorrect `layout_metadata`:
- `num_inputs: 1` (should be 2)
- `num_outputs: 1` (correct)
- `output_sizes: [1024]` (should be much larger for T5 hidden states)

This causes execution to fail with: "Expected 1 inputs, got 2"

## Root Cause

In `/hologram/crates/compiler/src/from_ir.rs`, the `convert_from_ir()` function loses input/output information when converting from `OperationGraph` to `CompileGraph`:

**OperationGraph has:**
```rust
pub struct OperationGraph {
    graph: DiGraph<Node, Edge>,
    pub inputs: FxHashMap<String, NodeIndex>,   // ✅ Named inputs
    pub outputs: FxHashMap<String, NodeIndex>,  // ✅ Named outputs
    // ...
}
```

**CompileGraph doesn't preserve this:**
```rust
pub struct CompileGraph {
    graph: DiGraph<OpNode, EdgeType>,
    node_names: HashMap<String, NodeIndex>,
    subgraph_deps: Vec<&'static str>,
    weights: HashMap<NodeIndex, WeightMetadata>,
    // ❌ No inputs/outputs fields!
}
```

When `BackendPlan` is created, it can't determine:
- How many input buffers are needed
- How many output buffers are needed
- The size of each buffer
- The shape of each tensor

## Solution

### Step 1: Add Input/Output Tracking to CompileGraph

**File:** `/hologram/crates/compiler/src/graph/mod.rs`

Add fields to `CompileGraph`:
```rust
pub struct CompileGraph {
    graph: DiGraph<OpNode, EdgeType>,
    node_names: HashMap<String, NodeIndex>,
    subgraph_deps: Vec<&'static str>,
    weights: HashMap<NodeIndex, WeightMetadata>,

    // NEW: Track graph inputs and outputs
    /// Named input nodes (from OperationGraph)
    pub inputs: HashMap<String, NodeIndex>,
    /// Named output nodes (from OperationGraph)
    pub outputs: HashMap<String, NodeIndex>,
}
```

Update `CompileGraph::new()`:
```rust
pub fn new() -> Self {
    CompileGraph {
        graph: DiGraph::new(),
        node_names: HashMap::new(),
        subgraph_deps: Vec::new(),
        weights: HashMap::new(),
        inputs: HashMap::new(),   // NEW
        outputs: HashMap::new(),  // NEW
    }
}
```

Add helper methods:
```rust
impl CompileGraph {
    // ... existing methods ...

    /// Register an input node.
    pub fn add_input(&mut self, name: String, node: NodeIndex) {
        self.inputs.insert(name, node);
    }

    /// Register an output node.
    pub fn add_output(&mut self, name: String, node: NodeIndex) {
        self.outputs.insert(name, node);
    }

    /// Get all input nodes in a consistent order (alphabetical by name).
    pub fn get_ordered_inputs(&self) -> Vec<(String, NodeIndex)> {
        let mut inputs: Vec<_> = self.inputs.iter()
            .map(|(name, idx)| (name.clone(), *idx))
            .collect();
        inputs.sort_by(|a, b| a.0.cmp(&b.0));
        inputs
    }

    /// Get all output nodes in a consistent order (alphabetical by name).
    pub fn get_ordered_outputs(&self) -> Vec<(String, NodeIndex)> {
        let mut outputs: Vec<_> = self.outputs.iter()
            .map(|(name, idx)| (name.clone(), *idx))
            .collect();
        outputs.sort_by(|a, b| a.0.cmp(&b.0));
        outputs
    }
}
```

### Step 2: Preserve Inputs/Outputs in convert_from_ir()

**File:** `/hologram/crates/compiler/src/from_ir.rs`

Update `convert_from_ir()` to copy input/output mappings:

```rust
pub fn convert_from_ir(ir_graph: &OperationGraph) -> ConversionResult<CompileGraph> {
    let mut compile_graph = CompileGraph::new();
    let mut node_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    // Get topological order for conversion
    let order = ir_graph
        .topological_order()
        .map_err(|e| ConversionError::new(format!("failed to get topological order: {}", e)))?;

    // Convert nodes in topological order
    for ir_idx in &order {
        let ir_node = ir_graph
            .node(*ir_idx)
            .ok_or_else(|| ConversionError::new("missing node in graph"))?;

        // Convert the operation
        let (op_node, weight_opt) = convert_node_op(&ir_node.op, &ir_node.shape)?;
        let compile_idx = compile_graph.add_op(op_node);
        node_map.insert(*ir_idx, compile_idx);

        // Attach weight if present
        if let Some(weight) = weight_opt {
            compile_graph.attach_weight(compile_idx, weight);
        }
    }

    // Convert edges
    for ir_idx in &order {
        let compile_idx = node_map[ir_idx];
        for pred_ir_idx in ir_graph.predecessors(*ir_idx) {
            if let Some(&pred_compile_idx) = node_map.get(&pred_ir_idx) {
                compile_graph.connect(pred_compile_idx, compile_idx);
            }
        }
    }

    // NEW: Copy input/output mappings
    for (name, ir_idx) in &ir_graph.inputs {
        if let Some(&compile_idx) = node_map.get(ir_idx) {
            compile_graph.add_input(name.clone(), compile_idx);
        }
    }

    for (name, ir_idx) in &ir_graph.outputs {
        if let Some(&compile_idx) = node_map.get(ir_idx) {
            compile_graph.add_output(name.clone(), compile_idx);
        }
    }

    Ok(compile_graph)
}
```

### Step 3: Populate layout_metadata in CompilationPipeline

**File:** `/hologram/crates/compiler/src/pipeline.rs` (or wherever `BackendPlan` is created)

When creating `BackendPlan`, extract input/output information from `CompileGraph`:

```rust
// Inside the compilation pipeline where BackendPlan is created:

fn create_backend_plan(compile_graph: &CompileGraph, /* ... */) -> BackendPlan {
    // ... existing plan creation code ...

    // NEW: Extract input/output metadata
    let ordered_inputs = compile_graph.get_ordered_inputs();
    let ordered_outputs = compile_graph.get_ordered_outputs();

    let num_inputs = ordered_inputs.len();
    let num_outputs = ordered_outputs.len();

    // Calculate input sizes and shapes
    let mut input_sizes = Vec::with_capacity(num_inputs);
    let mut input_shapes = Vec::with_capacity(num_inputs);

    for (_name, node_idx) in &ordered_inputs {
        if let Some(node) = compile_graph.node(*node_idx) {
            // Extract shape from the node (you'll need to access shape metadata)
            // For now, use placeholder logic - adjust based on actual node structure
            let shape = extract_shape_from_node(node); // Helper function needed
            let size_bytes = calculate_buffer_size(&shape); // Helper function needed

            input_shapes.push(shape);
            input_sizes.push(size_bytes);
        }
    }

    // Calculate output sizes and shapes
    let mut output_sizes = Vec::with_capacity(num_outputs);
    let mut output_shapes = Vec::with_capacity(num_outputs);

    for (_name, node_idx) in &ordered_outputs {
        if let Some(node) = compile_graph.node(*node_idx) {
            let shape = extract_shape_from_node(node);
            let size_bytes = calculate_buffer_size(&shape);

            output_shapes.push(shape);
            output_sizes.push(size_bytes);
        }
    }

    let layout_metadata = LayoutMetadata {
        num_inputs,
        num_outputs,
        input_sizes,
        output_sizes,
        input_shapes,
        output_shapes,
    };

    BackendPlan {
        // ... existing fields ...
        layout_metadata,
        // ... rest of fields ...
    }
}

// Helper function to extract shape from OpNode
fn extract_shape_from_node(node: &OpNode) -> [usize; 4] {
    // You'll need to determine how shapes are stored in OpNode
    // This might require adding shape metadata to OpNode or accessing it from elsewhere
    // For now, placeholder:
    match node {
        OpNode::Input { size_hint, .. } => {
            // Convert size_hint to shape
            // This is model-specific - for T5 encoder:
            // - input_ids: [batch_size, seq_len] → needs actual values
            // - attention_mask: [batch_size, seq_len]
            [1, 128, 1, 1] // Placeholder
        }
        OpNode::Output { .. } => {
            // T5 encoder output: [batch_size, seq_len, hidden_size]
            [1, 128, 512, 1] // Placeholder for T5-small
        }
        _ => [1, 1, 1, 1]
    }
}

// Helper function to calculate buffer size from shape
fn calculate_buffer_size(shape: &[usize; 4]) -> usize {
    let num_elements: usize = shape.iter().product();
    num_elements * 4 // Assuming f32 (4 bytes per element)
}
```

**Important Note:** The shape extraction logic above is simplified. You'll need to:
1. Either store shape information in `OpNode` (requires modifying the `OpNode` enum)
2. Or maintain a separate shape tracking map during compilation
3. Or extract shapes from the original OperationGraph nodes during conversion

The cleanest approach is probably to add shape metadata to `CompileGraph` similar to how weights are stored:
```rust
pub struct CompileGraph {
    // ... existing fields ...
    /// Shape metadata for nodes (from OperationGraph)
    shapes: HashMap<NodeIndex, Vec<usize>>,
}
```

And update `convert_from_ir()` to copy shapes:
```rust
// In convert_from_ir():
for ir_idx in &order {
    let ir_node = ir_graph.node(*ir_idx).unwrap();
    // ... conversion logic ...

    // Copy shape information
    if let Some(static_shape) = ir_node.shape.static_dims() {
        compile_graph.set_shape(compile_idx, static_shape);
    }
}
```

## Expected Result

After these changes:
- T5 encoder .holo files will have `layout_metadata.num_inputs = 2`
- `layout_metadata.num_outputs = 1`
- `layout_metadata.input_sizes` will contain correct buffer sizes
- `layout_metadata.output_sizes` will contain correct output buffer size (much larger than 1024)
- hologram-onnx executor will be able to allocate correct buffers and execute successfully

## Testing

After implementing, recompile the T5 encoder:
```bash
cd /workspace
cargo run --release -- compile models/t5-small/encoder_model.onnx -o models/t5-small/compiled/encoder.holo
```

Then test execution:
```bash
cargo run --release -- run --config configs/test-encoder.toml
```

Should see:
```
Plan requires 2 inputs, 1 outputs
```

Instead of:
```
Plan requires 1 inputs, 1 outputs
```

## Files to Modify

1. `/hologram/crates/compiler/src/graph/mod.rs` - Add inputs/outputs fields to CompileGraph
2. `/hologram/crates/compiler/src/from_ir.rs` - Preserve inputs/outputs in convert_from_ir()
3. `/hologram/crates/compiler/src/pipeline.rs` (or similar) - Populate layout_metadata when creating BackendPlan
4. Possibly add shape tracking to CompileGraph for accurate buffer size calculations

## Notes

- The alphabetical ordering convention is important - both hologram and hologram-onnx need to use the same ordering
- Shape information needs to flow from OperationGraph → CompileGraph → BackendPlan
- Buffer sizes should account for data type (f32 = 4 bytes, f16 = 2 bytes, etc.)
