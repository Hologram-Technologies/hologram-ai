# Hologram Integration Guide

**Version:** 1.0
**Date:** 2026-01-27
**Audience:** Any project using Hologram as a compute backend

---

## Table of Contents

1. [Overview](#1-overview)
2. [Getting Started](#2-getting-started)
3. [.holo File Formats](#3-holo-file-formats)
4. [Operation Discovery](#4-operation-discovery)
5. [Operation Invocation](#5-operation-invocation)
6. [Weight Handling Strategies](#6-weight-handling-strategies)
7. [Optimization Features](#7-optimization-features)
8. [Integration Checklist](#8-integration-checklist)
9. [API Reference](#9-api-reference)
10. [Error Handling](#10-error-handling)
11. [Performance Guidelines](#11-performance-guidelines)

---

## 1. Overview

### 1.1 What is Hologram?

Hologram is a high-performance tensor compute engine designed for:

- **Zero-copy execution** via rkyv serialization
- **Hardware memory bandwidth optimization** (~50GB/s throughput)
- **Compile-time optimizations** (Winograd transforms, epilogue fusion, SIMD dispatch)
- **Flexible weight management** (embedded, external, memory-mapped)
- **Dynamic shape support** with runtime dimension resolution

### 1.2 Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **30+ Pre-compiled Operations** | MatMul, Conv2D, binary ops, reductions, categorical algebra | Comprehensive operation coverage |
| **Automatic SIMD Dispatch** | AVX2, AVX-512, NEON detection | No manual optimization needed |
| **Epilogue Fusion** | Fuse bias + activation into weighted ops | Fewer kernel calls |
| **Winograd Convolutions** | 3x3 and 5x5 optimizations | 2-3x faster convolutions |
| **Memory-Mapped Weights** | Lazy OS-level loading | Low memory footprint for large models |
| **rkyv Serialization** | Zero-copy deserialization | ~10x faster than JSON |

### 1.3 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ├─ Build IR Graph (GraphBuilder)
                      │
                      ├─ Compile to BackendPlan (compile_to_plan)
                      │
                      ├─ Create Executor (PlanExecutor)
                      │
                      └─ Execute (O(1) dispatch)
                           │
                           ├─ GEMM Kernels
                           ├─ Convolution Kernels (Winograd)
                           ├─ Binary/Unary Ops (SIMD)
                           ├─ Reduction Ops
                           └─ Categorical Ops (ℤ₉₆ resonance ring)
```

---

## 2. Getting Started

### 2.1 Dependencies

**Cargo.toml:**

```toml
[dependencies]
hologram = "0.1"
hologram-compiler = "0.1"
hologram-backend = "0.1"

# Optional: FFI bindings for other languages
hologram-ffi = "0.1"
```

### 2.2 Basic Example

```rust
use hologram::compiler::{GraphBuilder, OpNode};
use hologram::backend::{create_best_backend, PlanExecutor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build IR graph
    let mut builder = GraphBuilder::new();
    let input = builder.add_input("x", vec![1, 784]);
    let weights = builder.add_constant("W", vec![784, 10], weight_data);
    let matmul = builder.add_op(
        OpNode::MatMul(MatMul::new(1, 784, 10)),
        vec![input, weights],
    );
    builder.add_output("y", matmul);

    // 2. Compile to plan
    let graph = builder.build()?;
    let backend = create_best_backend()?;
    let plan = graph.compile_to_plan(&backend)?;

    // 3. Execute
    let mut executor = PlanExecutor::new(plan, &*backend)?;
    executor.execute(&[input_tensor], &mut [output_buffer])?;

    Ok(())
}
```

### 2.3 Key Concepts

- **IR Graph**: High-level operation graph with nodes and edges
- **BackendPlan**: Compiled, immutable execution plan with kernel selection
- **Executor**: Manages workspace and executes plans with O(1) dispatch
- **Input Ordering**: Operations specify how inputs should be ordered (DataFirst, Ordered, etc.)
- **Epilogue Fusion**: Weighted operations can fuse bias + activation

---

## 3. .holo File Formats

### 3.1 Format Detection

Hologram supports three formats, detected via magic bytes:

| Format | Magic Bytes | Extension | Use Case |
|--------|-------------|-----------|----------|
| **SingleGraph** | None (rkyv) | `.holo` | Simple single-layer models |
| **Archive** | `0x1f 0x8b` (gzip) | `.holo` | Multi-layer models with dependencies |
| **BackendPlan** | `'HOLP'` (`0x484F4C50`) | `.holp` | Pre-compiled execution plans |

```rust
use hologram::compiler::{detect_holo_format, HoloFormat};

let format = detect_holo_format(Path::new("model.holo"))?;
match format {
    HoloFormat::SingleGraph => println!("rkyv-serialized graph"),
    HoloFormat::Archive => println!("tar.gz archive"),
    HoloFormat::BackendPlan => println!("Compiled plan"),
}
```

### 3.2 SingleGraph Format

**Structure:**

```rust
pub struct OperationGraphData {
    pub version: u32,                    // Format version
    pub nodes: Vec<Node>,                // Graph nodes
    pub edges: Vec<(u32, u32)>,         // (source, target) pairs
    pub constants: Vec<ConstantData>,   // Embedded constants
    pub inputs: HashMap<String, u32>,   // Named inputs
    pub outputs: HashMap<String, u32>,  // Named outputs
}
```

**Loading:**

```rust
let bytes = std::fs::read("model.holo")?;
let graph = OperationGraphData::from_bytes(&bytes)?;

// Zero-copy access (no deserialization)
let archived = OperationGraphData::archived_ref(&bytes)?;
println!("Nodes: {}", archived.nodes.len());
```

### 3.3 Archive Format

**Structure:**

```
model.holo (tar.gz)
├── manifest.toml              # Model metadata
├── layers/
│   ├── embedding.holo         # Layer 1
│   ├── transformer.holo       # Layer 2
│   └── head.holo              # Layer 3
└── weights/                   # Optional weight files
    ├── embedding.weights
    └── transformer.weights
```

**Manifest (manifest.toml):**

```toml
version = "1.0.0"
min_version = "1.0.0"

[metadata]
name = "bert-base"
description = "BERT base model"
created_at = "2025-01-02T00:00:00Z"

[[layers]]
id = "abc123..."                    # SHA-256 content hash
name = "embedding"
holo_path = "layers/embedding.holo"
weights_path = "weights/embedding.weights"

[layers.imports]
input_ids = { dtype = "i64", shape = [null, null] }

[layers.exports]
embeddings = { dtype = "f32", shape = [null, 768] }
```

**Loading:**

```rust
use hologram::compiler::ArchiveReader;

let reader = ArchiveReader::from_reader(File::open("model.holo")?)?;
let manifest = reader.manifest();

// Verify checksums
if !reader.all_checksums_valid() {
    return Err("Integrity check failed");
}

// Load layer + weights
let graph = reader.get_layer("transformer")?;
let weights = reader.get_weights("transformer");
```

### 3.4 BackendPlan Format

Pre-compiled execution plan with:
- Selected kernels (GEMM, Conv, Binary, etc.)
- Buffer references (input, constant, workspace)
- Workspace size requirements
- Dimension expressions for dynamic shapes

```rust
let plan = BackendPlan::from_bytes(&std::fs::read("model.holp")?)?;
let executor = PlanExecutor::new(plan, &*backend)?;
```

**Critical files:**
- [specs/holo-format.md](../holo-format.md) - Complete format specification
- [crates/compiler/src/format/archive.rs](../../crates/compiler/src/format/archive.rs)
- [crates/compiler/src/format/single.rs](../../crates/compiler/src/format/single.rs)

---

## 4. Operation Discovery

### 4.1 Available Operations

| Category | Operations | Weighted? |
|----------|-----------|-----------|
| **GEMM** | MatMul, Gemm | Yes |
| **Convolution** | Conv2D, ConvTranspose2D, DepthwiseConv | Yes |
| **Binary** | Add, Sub, Mul, Div, Max, Min, Pow, Mod | No |
| **Unary** | Relu, Sigmoid, Tanh, Gelu, SiLU, Exp, Log, Sqrt, Sin, Cos | No |
| **Reduction** | ReduceSum, ReduceMax, ReduceMin, ReduceMean, ArgMax, ArgMin | No |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm, Softmax | Some (BatchNorm) |
| **Pooling** | MaxPool2D, AvgPool2D, GlobalAvgPool | No |
| **Tensor Ops** | Gather, Concat, Split, Slice, Reshape, Transpose, Squeeze, Unsqueeze | No |
| **Categorical** | CharProduct, OrbitClassify, Lift, MultInverse, MonsterChar | No |
| **Bitwise** | BitAnd, BitOr, BitXor, BitNot, Shl, Shr, Rotl, Rotr | No |

### 4.2 From SingleGraph

```rust
let graph = OperationGraphData::from_bytes(&bytes)?;

for node in &graph.nodes {
    match node.op.kind {
        OpKind::MatMul => {
            println!("MatMul({}x{}x{})",
                node.op.params.dims[0],  // M
                node.op.params.dims[1],  // K
                node.op.params.dims[2]); // N
        }
        OpKind::Add => {
            println!("Add(size={})", node.op.params.dims[0]);
        }
        OpKind::ReduceSum => {
            println!("ReduceSum(axis={})", node.op.params.extra[0]);
        }
        _ => println!("Operation: {:?}", node.op.kind),
    }
}

// Named inputs/outputs
for (name, node_id) in &graph.inputs {
    println!("Input '{}' -> node {}", name, node_id);
}
```

### 4.3 From Archive

```rust
let reader = ArchiveReader::from_reader(File::open("model.holo")?)?;

for layer in &reader.manifest().layers {
    println!("Layer: {}", layer.name);
    println!("  Inputs: {:?}", layer.imports.keys());
    println!("  Outputs: {:?}", layer.exports.keys());
    println!("  Has weights: {}", layer.weights_path.is_some());

    let graph = reader.get_layer(&layer.name)?;
    // ... enumerate operations as above
}
```

### 4.4 From BackendPlan

```rust
let plan = BackendPlan::from_bytes(&bytes)?;

for op in &plan.ops {
    match op.kernel_id {
        KernelId::GEMM_STANDARD => println!("GEMM"),
        KernelId::CONV_WINOGRAD_F23 => println!("Conv2D (Winograd F(2,3))"),
        KernelId::ELEM_ADD => println!("Add"),
        KernelId::REDUCE_SUM => println!("ReduceSum"),
        _ => println!("Kernel: {:?}", op.kernel_id),
    }

    // Inspect input types
    for (idx, buf_ref) in op.input_refs.iter().enumerate() {
        match buf_ref {
            BufferRef::Input(i) => {
                println!("  Input[{}] <- runtime input {}", idx, i);
            }
            BufferRef::Constant { offset, size } => {
                println!("  Input[{}] <- constant @{} ({}B)", idx, offset, size);
            }
            BufferRef::ExternalConstant { path, offset, size } => {
                println!("  Input[{}] <- '{}' @{} ({}B)", idx, path, offset, size);
            }
            _ => {}
        }
    }
}
```

---

## 5. Operation Invocation

### 5.1 Input Ordering Contract

Operations declare input ordering via `InputOrdering` enum:

```rust
pub enum InputOrdering {
    Default,               // No special ordering
    DataFirst,            // [data, weights, bias] - weighted ops
    DataBeforeConstants,  // [data, constants] - commutative ops
    Ordered,              // Strict ordering - non-commutative ops (Sub, Div)
    InputThenShape,       // [input, shape_tensor] - dynamic shapes
}
```

**Examples:**

```rust
// MatMul: DataFirst (weighted)
let matmul = MatMul::new(m, k, n);
assert_eq!(matmul.input_ordering(), InputOrdering::DataFirst);
// Inputs MUST be: [activation_data, weight_matrix, optional_bias]

// Add: DataBeforeConstants (commutative)
let add = Add::new(size);
assert_eq!(add.input_ordering(), InputOrdering::DataBeforeConstants);
// Inputs can be: [data1, data2] or [data, constant]

// Sub: Ordered (non-commutative)
let sub = Sub::new(size);
assert_eq!(sub.input_ordering(), InputOrdering::Ordered);
// Inputs MUST preserve order: [a, b] -> a - b
```

**Important:** Query `input_ordering()` and sort inputs accordingly before constructing your IR graph.

### 5.2 Execution Pattern 1: Build IR → Compile → Execute

```rust
use hologram::compiler::{GraphBuilder, OpNode, MatMul, Add};
use hologram::backend::{create_best_backend, PlanExecutor};

// Build graph
let mut builder = GraphBuilder::new();
let x = builder.add_input("x", vec![1, 784]);
let w = builder.add_constant("W", vec![784, 10], weight_data);
let b = builder.add_constant("bias", vec![10], bias_data);

// MatMul (weighted): DataFirst ordering
let matmul = builder.add_op(
    OpNode::MatMul(MatMul::new(1, 784, 10)),
    vec![x, w],  // Data first, weights second
);

// Add (unweighted): DataBeforeConstants ordering
let add = builder.add_op(
    OpNode::Add(Add::new(10)),
    vec![matmul, b],
);

builder.add_output("y", add);

// Compile to plan
let graph = builder.build()?;
let backend = create_best_backend()?;
let plan = graph.compile_to_plan(&backend)?;

// Execute
let mut executor = PlanExecutor::new(plan, &*backend)?;
executor.execute(&[input_data], &mut [output_buffer])?;
```

### 5.3 Execution Pattern 2: Load Pre-compiled Plan

```rust
// Load cached .holp file (much faster)
let plan = BackendPlan::from_bytes(&std::fs::read("model.holp")?)?;
let backend = create_best_backend()?;
let mut executor = PlanExecutor::new(plan, &*backend)?;

// Execute immediately
executor.execute(&inputs, &mut outputs)?;
```

### 5.4 Execution Pattern 3: Load Archive with External Weights

```rust
let reader = ArchiveReader::from_reader(File::open("model.holo")?)?;

// Verify integrity
if !reader.all_checksums_valid() {
    return Err("Checksum validation failed");
}

// Load layer + weights
let graph = reader.get_layer("transformer")?;
let weights = reader.get_weights("transformer")
    .ok_or("Weights not found")?;

// Compile with weights
let backend = create_best_backend()?;
let plan = graph.compile_to_plan_with_weights(&backend, weights)?;
let executor = PlanExecutor::new(plan, &*backend)?;
```

---

## 6. Weight Handling Strategies

### 6.1 Embedded Weights (Models < 100MB)

**Use case:** Small models, fast startup required, single-file deployment

```rust
// Embed weights in constant_data section
let mut builder = GraphBuilder::new();
let w = builder.add_constant("W", shape, weight_data);  // Embedded

// Compile and save
let plan = graph.compile_to_plan(&backend)?;
std::fs::write("model.holp", plan.to_bytes()?)?;

// Load (all weights in memory immediately)
let plan = BackendPlan::from_bytes(&std::fs::read("model.holp")?)?;
let executor = PlanExecutor::new(plan, &*backend)?;
```

**Pros:**
- ✅ Fast startup (no mmap overhead)
- ✅ Single file deployment
- ✅ Simple management

**Cons:**
- ❌ High memory usage (all weights loaded)
- ❌ File size = weights + graph

### 6.2 Memory-Mapped Weights (Models > 1GB)

**Use case:** Large models, memory-constrained environments, low startup latency

```rust
// Compile graph (no weights embedded)
let plan = graph.compile_to_plan(&backend)?;

// Save plan + weights separately
std::fs::write("model.holp", plan.to_bytes()?)?;
std::fs::write("model.weights", raw_weight_data)?;

// Load with lazy loading
let plan = BackendPlan::from_bytes(&std::fs::read("model.holp")?)?;
let executor = PlanExecutor::with_external_constants(
    plan,
    &*backend,
    Path::new("model.weights"),
)?;

// Set access pattern
executor.set_const_access_pattern(ConstantAccessPattern::Random);

// RSS stays low - OS pages in data on demand
executor.execute(&inputs, &mut outputs)?;
```

**Pros:**
- ✅ Low memory footprint (lazy loading)
- ✅ Fast startup (no upfront loading)
- ✅ Multiple processes can share mmap
- ✅ Works with models larger than RAM

**Cons:**
- ❌ Two-file deployment
- ❌ First access has page fault overhead

### 6.3 Hybrid Strategy (100MB - 1GB)

**Use case:** Balance between startup speed and memory usage

```rust
// Embed critical small layers, mmap large layers
if layer_size < 100_000_000 {
    builder.add_constant("small_W", shape, data);  // Embed
} else {
    builder.add_external_constant(
        "large_W",
        shape,
        "large_weights.bin",
        offset,
        size
    );  // Mmap
}
```

### 6.4 Weight Location Enum

```rust
pub enum WeightLocation {
    Embedded { offset: u64, size: u64 },              // In .holo file
    External { path: String, offset: u64, size: u64 },  // Separate file
    MemoryMapped { path: String },                    // Lazy-loaded
    Inline(Vec<u8>),                                  // Small constants
}
```

---

## 7. Optimization Features

### 7.1 Epilogue Fusion

Weighted operations can fuse bias addition + activation into a single kernel:

```rust
// Check if operation supports fusion
let matmul = MatMul::new(64, 512, 256);
if let Some(epilogue) = matmul.epilogue_info() {
    println!("Channel dim: {}", epilogue.channel_dim);  // 256
    println!("Fusion kind: {}", epilogue.kind);         // "matmul"
    // Compiler can fuse: matmul + bias_add + relu
}
```

**Without fusion (3 kernel calls):**

```rust
let mm = builder.add_op(OpNode::MatMul(...), vec![x, w]);
let add = builder.add_op(OpNode::Add(...), vec![mm, bias]);
let relu = builder.add_op(OpNode::Relu(...), vec![add]);
```

**With fusion (1 kernel call):**

Compiler automatically detects the pattern and generates a fused kernel:
`matmul_fused_bias_relu(x, w, bias) -> output`

**Supported operations:**
- MatMul
- Gemm
- Conv2D
- LayerNorm
- RMSNorm
- BatchNorm

**Action required:** None - fusion is automatic when patterns are detected.

### 7.2 Winograd-Transformed Convolutions

3x3 and 5x5 convolutions automatically use Winograd algorithm:

```rust
let conv = Conv2D::new(56, 56, 3, 3, 64, 128, 1);  // 3x3 kernel
let kernel_info = conv.select_kernel(TensorDtype::Float32);

// Automatically selects Winograd F(2,3)
assert_eq!(kernel_info.algorithm, AlgorithmVariant::WinogradF23);
assert_eq!(kernel_info.kernel_id, KernelId::CONV_WINOGRAD_F23);
```

**Weight transformation happens at compile-time:**

```rust
// Original weights: [3, 3, 64, 128]
// Transformed weights: [4, 4, 64, 128] (Winograd domain)

// Compiler writes transformed weights to constant_data
// Runtime kernel uses pre-transformed weights directly
```

**Performance improvement:** 2-3x faster than direct convolution for 3x3 kernels.

**Selection criteria:**
- **Winograd F(2,3)**: 3x3 kernels, input size ≥ 56x56
- **Winograd F(4,3)**: 5x5 kernels
- **Im2Col**: General fallback
- **Direct**: Small inputs or non-standard kernels

**Action required:** None - algorithm selection is automatic.

### 7.3 SIMD Dispatch

Operations automatically dispatch to the best SIMD variant:

```rust
use hologram::lookup::{detect_simd, SimdLevel};

let simd = detect_simd();
match simd {
    SimdLevel::Avx512 => println!("Using AVX-512 kernels"),
    SimdLevel::Avx2 => println!("Using AVX2 kernels"),
    SimdLevel::Neon => println!("Using NEON kernels (ARM)"),
    SimdLevel::Scalar => println!("Using scalar fallback"),
}

// Binary operations automatically use SIMD
let add = Add::new(1_000_000);
executor.execute(...)?;  // Dispatches to avx2_add or neon_add
```

**Supported operations:** All binary ops (Add, Mul, etc.), unary ops, reductions

**Action required:** None - dispatch is automatic based on CPU capabilities.

### 7.4 Dynamic Shape Support

Use `0` for dimensions that will be resolved at runtime:

```rust
// Build graph with dynamic batch size
let mut builder = GraphBuilder::new();
let x = builder.add_input("x", vec![0, 784]);  // batch=0 (dynamic)
let matmul = builder.add_op(
    OpNode::MatMul(MatMul::new(0, 784, 10)),  // m=0 (dynamic)
    vec![x, w],
);
builder.add_output("y", matmul);

// Compile to plan
let plan = graph.compile_to_plan(&backend)?;
let mut executor = PlanExecutor::new(plan, &*backend)?;

// Register actual shape at runtime
executor.register_input_shape(0, &[32, 784])?;  // batch=32
executor.execute(&inputs_32, &mut outputs_32)?;

// Can change batch size
executor.register_input_shape(0, &[64, 784])?;  // batch=64
executor.execute(&inputs_64, &mut outputs_64)?;
```

**DimExpr resolution:**

```rust
pub enum DimExpr {
    Static(usize),                           // Constant value
    InputRef { input_id, dim_index },        // Read from input shape
    PredecessorElements { slot, divisor },   // Compute from predecessor
    // ... arithmetic operations
}
```

**Workspace allocation:**
- **Static shapes**: Allocated once at executor creation
- **Dynamic shapes**: Reallocated when shape changes

---

## 8. Integration Checklist

### 8.1 Graph Builder

- [ ] Create IR graphs using `GraphBuilder`
- [ ] Query `input_ordering()` for each operation
- [ ] Sort inputs according to ordering contract
- [ ] Add inputs via `add_input(name, shape)`
- [ ] Add constants via `add_constant()` or `add_external_constant()`
- [ ] Add operations via `add_op(OpNode, inputs)`
- [ ] Add outputs via `add_output(name, node_id)`

### 8.2 Weight Manager

- [ ] Decide strategy per layer: embedded vs external vs mmap
- [ ] For models < 100MB: use embedded weights
- [ ] For models > 1GB: use memory-mapped weights
- [ ] For hybrid: embed small layers, mmap large layers
- [ ] Implement weight loading/streaming as needed

### 8.3 Compiler Integration

- [ ] Compile IR to `BackendPlan` via `compile_to_plan()`
- [ ] Cache compiled .holp files for production
- [ ] Serialize plans with `plan.to_bytes()`
- [ ] Deserialize plans with `BackendPlan::from_bytes()`
- [ ] Handle compilation errors gracefully

### 8.4 Executor Management

- [ ] Create executor with `PlanExecutor::new()` or `with_external_constants()`
- [ ] Handle dynamic shapes via `register_input_shape()`
- [ ] Allocate buffers for inputs/outputs
- [ ] Reuse executor across multiple forward passes
- [ ] Clean up executor when done

### 8.5 Error Handling

- [ ] Catch `CompileError` during compilation
- [ ] Catch `BackendError` during execution
- [ ] Validate graph before compilation via `graph.validate()`
- [ ] Validate buffer sizes match plan requirements
- [ ] Validate shapes match plan's input specification

### 8.6 Testing

- [ ] Test operation discovery from .holo files
- [ ] Test weighted vs unweighted execution
- [ ] Test all three weight loading strategies
- [ ] Test dynamic shape execution
- [ ] Benchmark compilation overhead
- [ ] Verify memory footprint matches expectations
- [ ] Test FFI bindings if using other languages

---

## 9. API Reference

### 9.1 Core Types

**GraphBuilder:**

```rust
pub struct GraphBuilder {
    // ...
}

impl GraphBuilder {
    pub fn new() -> Self;
    pub fn add_input(&mut self, name: &str, shape: Vec<usize>) -> NodeId;
    pub fn add_constant(&mut self, name: &str, shape: Vec<usize>, data: Vec<u8>) -> NodeId;
    pub fn add_external_constant(&mut self, name: &str, shape: Vec<usize>, path: &str, offset: u64, size: u64) -> NodeId;
    pub fn add_op(&mut self, op: OpNode, inputs: Vec<NodeId>) -> NodeId;
    pub fn add_output(&mut self, name: &str, node_id: NodeId);
    pub fn build(self) -> Result<CompileGraph, CompileError>;
}
```

**BackendPlan:**

```rust
pub struct BackendPlan {
    pub ops: Vec<KernelOp>,
    pub constant_data: Vec<u8>,
    pub workspace_size: usize,
    // ...
}

impl BackendPlan {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error>;
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error>;
}
```

**PlanExecutor:**

```rust
pub struct PlanExecutor {
    // ...
}

impl PlanExecutor {
    pub fn new(plan: BackendPlan, backend: &dyn ProgramBackend) -> Result<Self, BackendError>;
    pub fn with_external_constants(plan: BackendPlan, backend: &dyn ProgramBackend, weights_path: &Path) -> Result<Self, BackendError>;
    pub fn register_input_shape(&mut self, input_id: usize, shape: &[usize]) -> Result<(), BackendError>;
    pub fn execute(&mut self, inputs: &[&[u8]], outputs: &mut [&mut [u8]]) -> Result<(), BackendError>;
    pub fn set_const_access_pattern(&mut self, pattern: ConstantAccessPattern);
}
```

### 9.2 Operation Traits

**Operation:**

```rust
pub trait Operation {
    fn name(&self) -> &'static str;
    fn input_ordering(&self) -> InputOrdering { InputOrdering::Default }
    fn epilogue_info(&self) -> Option<EpilogueInfo> { None }
}
```

**WorkspaceRequirements:**

```rust
pub trait WorkspaceRequirements {
    fn workspace_size(&self) -> usize { 0 }
}
```

**KernelSelectable:**

```rust
pub trait KernelSelectable {
    fn select_kernel(&self, dtype: TensorDtype) -> KernelInfo;
}
```

### 9.3 Key Enums

**InputOrdering:**

```rust
pub enum InputOrdering {
    Default,
    DataFirst,
    DataBeforeConstants,
    Ordered,
    InputThenShape,
}
```

**ConstantAccessPattern:**

```rust
pub enum ConstantAccessPattern {
    Random,                            // Disable readahead
    Sequential { readahead_bytes },    // Prefetch window
}
```

**BufferRef:**

```rust
pub enum BufferRef {
    Input(usize),
    Output(usize),
    Workspace(usize),
    Constant { offset: u64, size: u64 },
    ExternalConstant { path: String, offset: u64, size: u64 },
}
```

**Critical files:**
- [crates/compiler/src/lib.rs](../../crates/compiler/src/lib.rs) - Public API exports
- [crates/backend/src/lib.rs](../../crates/backend/src/lib.rs) - Backend API exports
- [crates/compiler/src/graph/ops/traits.rs](../../crates/compiler/src/graph/ops/traits.rs) - Operation traits

---

## 10. Error Handling

### 10.1 Error Types

**CompileError:**

```rust
pub enum CompileError {
    ShapeInference(String),       // Cannot infer shape
    UnsupportedOp(String),        // Operation not supported
    InvalidGraph(String),         // Cyclic graph, orphaned nodes, etc.
    KernelSelection(String),      // No kernel available
    // ...
}
```

**BackendError:**

```rust
pub enum BackendError {
    Buffer(String),               // Buffer size mismatch
    Kernel(String),               // Kernel execution failed
    Shape(String),                // Shape mismatch
    InvalidPlan(String),          // Plan validation failed
    ConstantNotFound(String),     // Weight not found
    // ...
}
```

### 10.2 Error Handling Flow

```rust
// 1. Validate graph
graph.validate()?;
// Checks: cycles, orphaned nodes, shape compatibility

// 2. Compile (performs additional validation)
let plan = graph.compile_to_plan(&backend)?;
// Checks: kernel selection, shape inference, workspace requirements

// 3. Create executor (validates constants)
let executor = PlanExecutor::new(plan, &*backend)?;
// Checks: constant availability, weight file exists

// 4. Register shapes if dynamic
if plan.has_dynamic_dims() {
    executor.register_input_shape(0, &shape)?;
    // Checks: dimension compatibility, workspace size
}

// 5. Execute
executor.execute(&inputs, &mut outputs)?;
// Checks: buffer sizes, pointer validity
```

### 10.3 Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ShapeInference("Cannot infer output shape")` | Missing shape information | Provide explicit shapes for all inputs |
| `Buffer("Input buffer too small")` | Buffer size mismatch | Allocate buffers matching plan's requirements |
| `ConstantNotFound("model.weights")` | Weight file missing | Ensure weight file exists at specified path |
| `Shape("Dynamic dimension mismatch")` | Wrong shape registered | Register correct shape via `register_input_shape()` |
| `InvalidGraph("Cycle detected")` | Cyclic dependency | Ensure graph is acyclic |

### 10.4 Debugging Tips

```rust
// Enable logging
env_logger::init();

// Validate graph before compilation
match graph.validate() {
    Ok(()) => println!("Graph valid"),
    Err(e) => eprintln!("Validation error: {}", e),
}

// Check plan structure
println!("Ops: {}", plan.ops.len());
println!("Workspace: {} bytes", plan.workspace_size);
println!("Constants: {} bytes", plan.constant_data.len());

// Inspect operation inputs
for (i, op) in plan.ops.iter().enumerate() {
    println!("Op {}: kernel={:?}", i, op.kernel_id);
    for (j, buf_ref) in op.input_refs.iter().enumerate() {
        println!("  Input {}: {:?}", j, buf_ref);
    }
}
```

---

## 11. Performance Guidelines

### 11.1 Compilation Overhead

```rust
// Cold start: ~10-100ms compilation + 1-10ms executor setup
let start = Instant::now();
let plan = graph.compile_to_plan(&backend)?;      // Expensive
let executor = PlanExecutor::new(plan, &*backend)?;  // Fast
println!("Cold start: {:?}", start.elapsed());

// Warm start: ~1ms load + 1ms executor setup
let start = Instant::now();
let plan = BackendPlan::from_bytes(&cached_bytes)?;  // Fast
let executor = PlanExecutor::new(plan, &*backend)?;
println!("Warm start: {:?}", start.elapsed());
```

**Recommendation:** Pre-compile and cache .holp files for production.

### 11.2 Memory Footprint Strategy

| Model Size | Strategy | Startup RSS | Steady-state RSS | Use Case |
|------------|----------|-------------|------------------|----------|
| < 100 MB | Embedded | Full size | Full size | Edge devices, fast startup |
| 100 MB - 1 GB | Hybrid | Critical layers | + active pages | Balanced |
| > 1 GB | Memory-mapped | ~0 MB | Active pages only | Large models, low memory |

**Example selection logic:**

```rust
fn select_weight_strategy(model_size_mb: usize) -> WeightStrategy {
    if model_size_mb < 100 {
        WeightStrategy::Embedded
    } else if model_size_mb < 1000 {
        WeightStrategy::Hybrid
    } else {
        WeightStrategy::MemoryMapped
    }
}
```

### 11.3 Workspace Reuse

```rust
// Static shapes: allocate once, reuse forever
let executor = PlanExecutor::new(plan, &*backend)?;
for i in 0..1000 {
    executor.execute(&inputs[i], &mut outputs[i])?;
    // Workspace reused every iteration
}

// Dynamic shapes: reallocate on shape change
let mut executor = PlanExecutor::new(plan, &*backend)?;

executor.register_input_shape(0, &[32, 512])?;  // Allocates for batch=32
for i in 0..100 {
    executor.execute(&inputs_32[i], &mut outputs_32[i])?;
}

executor.register_input_shape(0, &[64, 512])?;  // Reallocates for batch=64
for i in 0..100 {
    executor.execute(&inputs_64[i], &mut outputs_64[i])?;
}
```

**Recommendation:** Batch inputs with the same shape to minimize workspace reallocations.

### 11.4 Benchmarking

```rust
use std::time::Instant;

// Benchmark compilation
let start = Instant::now();
let plan = graph.compile_to_plan(&backend)?;
println!("Compilation: {:?}", start.elapsed());

// Benchmark executor creation
let start = Instant::now();
let executor = PlanExecutor::new(plan, &*backend)?;
println!("Executor setup: {:?}", start.elapsed());

// Benchmark execution (warm up first)
for _ in 0..10 {
    executor.execute(&inputs, &mut outputs)?;
}

let iterations = 1000;
let start = Instant::now();
for _ in 0..iterations {
    executor.execute(&inputs, &mut outputs)?;
}
let elapsed = start.elapsed();
println!("Avg execution: {:?}", elapsed / iterations);
```

### 11.5 Memory Profiling

**Linux:**

```bash
/usr/bin/time -v ./your_program 2>&1 | grep "Maximum resident"
```

**macOS:**

```bash
/usr/bin/time -l ./your_program 2>&1 | grep "maximum resident"
```

**Rust (via custom allocator):**

```rust
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    let _profiler = dhat::Profiler::new_heap();
    // ... run your code
}
```

---

## Appendix: Complete Example

### Simple Neural Network

```rust
use hologram::compiler::{GraphBuilder, OpNode, MatMul, Add};
use hologram::backend::{create_best_backend, PlanExecutor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load weights (from file, network, etc.)
    let w1_data = load_weights("layer1.bin")?;
    let b1_data = load_weights("bias1.bin")?;
    let w2_data = load_weights("layer2.bin")?;
    let b2_data = load_weights("bias2.bin")?;

    // Build 2-layer network: x @ W1 + b1 -> ReLU -> @ W2 + b2
    let mut builder = GraphBuilder::new();

    // Layer 1
    let x = builder.add_input("x", vec![1, 784]);
    let w1 = builder.add_constant("W1", vec![784, 512], w1_data);
    let b1 = builder.add_constant("b1", vec![512], b1_data);
    let mm1 = builder.add_op(
        OpNode::MatMul(MatMul::new(1, 784, 512)),
        vec![x, w1],
    );
    let add1 = builder.add_op(
        OpNode::Add(Add::new(512)),
        vec![mm1, b1],
    );
    let relu = builder.add_op(
        OpNode::FusedActivation(FusedActivation::relu()),
        vec![add1],
    );

    // Layer 2
    let w2 = builder.add_constant("W2", vec![512, 10], w2_data);
    let b2 = builder.add_constant("b2", vec![10], b2_data);
    let mm2 = builder.add_op(
        OpNode::MatMul(MatMul::new(1, 512, 10)),
        vec![relu, w2],
    );
    let add2 = builder.add_op(
        OpNode::Add(Add::new(10)),
        vec![mm2, b2],
    );

    builder.add_output("logits", add2);

    // Compile and cache
    let graph = builder.build()?;
    let backend = create_best_backend()?;
    let plan = graph.compile_to_plan(&backend)?;

    // Save compiled plan
    std::fs::write("model.holp", plan.to_bytes()?)?;
    println!("Compiled plan saved to model.holp");

    // Create executor
    let mut executor = PlanExecutor::new(plan, &*backend)?;

    // Run inference
    let input_data = vec![0.0f32; 784];
    let mut output_data = vec![0.0f32; 10];

    let input_bytes = unsafe {
        std::slice::from_raw_parts(
            input_data.as_ptr() as *const u8,
            input_data.len() * 4,
        )
    };
    let output_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * 4,
        )
    };

    executor.execute(&[input_bytes], &mut [output_bytes])?;

    println!("Output: {:?}", output_data);

    Ok(())
}

fn load_weights(path: &str) -> Result<Vec<u8>, std::io::Error> {
    std::fs::read(path)
}
```

---

## Additional Resources

- **Format Specification:** [specs/holo-format.md](../holo-format.md)
- **Operation Reference:** [crates/compiler/src/graph/ops/](../../crates/compiler/src/graph/ops/)
- **Backend Documentation:** [crates/backend/README.md](../../crates/backend/README.md)
- **FFI Bindings:** [crates/ffi/README.md](../../crates/ffi/README.md)
- **Examples:** [examples/](../../examples/)

---

**Questions or Issues?** Open an issue at: https://github.com/your-org/hologram/issues
