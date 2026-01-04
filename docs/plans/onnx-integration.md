# hologram-onnx Restructuring Guide

Restructure `hologram-onnx` to integrate with the new `hologram` crate architecture (at `/hologram`).

---

## Current hologram-onnx Analysis

### Existing Crates (5 total, ~25k lines)

| Crate                  | Lines | Status                                     | Keep          |
| ---------------------- | ----- | ------------------------------------------ | ------------- |
| `hologram-onnx-config` | 5,717 | Valuable - unified config, handlers, tests | ✅ Keep as-is |
| `hologram-onnx-core`   | 7,030 | Parser, shapes, weights, IR conversion     | Flatten       |
| `hologram-onnx-ops`    | 8,691 | 80 op translators, decomposition           | Flatten       |
| `hologram-onnx-spec`   | ~100  | ONNX protobuf bindings (prost)             | Flatten       |
| `hologram-onnx-cli`    | 3,476 | CLI commands                               | Flatten       |

### What's Worth Keeping

**hologram-onnx-config (Keep separate):**

- Unified TOML config format (minimal 4-line to complex multi-model)
- Multi-modal output handlers (image/audio/text with SIMD)
- 1,151 lines of integration tests
- Dynamic shape expressions
- Feature-gated optional handlers

**hologram-onnx-ops (Valuable, needs update):**

- 80 operation translators across 12 categories
- Decomposition passes (Conv2D→Im2col+GEMM)
- Well-organized by op type

**hologram-onnx-core (Partially reusable):**

- Parser (keep prost-based parsing)
- Weights (dedup, streaming - keep)
- Shapes (replace with hologram-ir)
- IR conversion (update to use hologram-ir)

**hologram-onnx-spec (Keep):**

- onnx.proto3 + prost-build setup
- Generated protobuf types

---

## Recommended Structure: 2 Crates

```
hologram-onnx/
├── Cargo.toml                    # Workspace with 2 members
├── crates/
│   ├── hologram-onnx-config/     # KEEP AS-IS (5,717 lines)
│   │   ├── src/
│   │   │   ├── config.rs         # TOML parsing
│   │   │   ├── unified.rs        # Unified config format
│   │   │   ├── conversion.rs     # Config translation
│   │   │   ├── output_handlers/  # Image/audio/text handlers
│   │   │   └── ...
│   │   └── tests/
│   │
│   └── hologram-onnx/            # FLATTEN core+ops+spec+cli
│       ├── build.rs              # prost-build for onnx.proto3
│       ├── proto/
│       │   └── onnx.proto3
│       └── src/
│           ├── lib.rs            # Public API
│           ├── proto.rs          # Generated protobuf types
│           ├── error.rs
│           ├── parser.rs         # From core
│           ├── weights.rs        # From core (keep dedup)
│           ├── converter.rs      # Updated to use hologram-ir
│           ├── context.rs
│           ├── ops/              # From ops crate
│           │   ├── mod.rs
│           │   ├── core.rs
│           │   ├── activation.rs
│           │   ├── shape.rs
│           │   ├── conv.rs
│           │   ├── norm.rs
│           │   ├── pool.rs
│           │   ├── reduction.rs
│           │   ├── advanced.rs   # LSTM, GRU, Attention
│           │   └── logical.rs
│           └── cli/              # From cli crate (optional)
│               ├── mod.rs
│               ├── compile.rs
│               ├── run.rs
│               └── ...
└── tests/
    └── models/                   # Test ONNX files
```

---

## Dependencies

```toml
# hologram-onnx/crates/hologram-onnx/Cargo.toml
[package]
name = "hologram-onnx"
version = "0.1.0"
edition = "2024"

[dependencies]
# New hologram crates (at /hologram)
hologram-ir = { path = "/hologram/crates/ir" }
hologram-compiler = { path = "/hologram/crates/compiler" }

# Sibling config crate
hologram-onnx-config = { path = "../hologram-onnx-config" }

# Protobuf
prost = "0.13"
bytes = "1.0"

# Utils (keep from existing)
thiserror = "1.0"
ahash = "0.8"
bytemuck = "1.0"

# CLI (optional)
clap = { version = "4.0", optional = true }

[build-dependencies]
prost-build = "0.13"

[features]
default = []
cli = ["dep:clap"]
```

---

## Key Changes from Old to New

### 1. Shape System

**Old (hologram-onnx-core/shapes.rs):**

```rust
use hologram_compiler::shapes::{Dim, Shape};

pub enum Dim {
    Concrete(usize),
    Var(String),
    Expr(DimExpr),
}
```

**New (use hologram-ir directly):**

```rust
use hologram_ir::{Shape, Dim};

// Dim::Static(n) replaces Dim::Concrete(n)
// Dim::Symbolic(s) replaces Dim::Var(s)
// Dim::Dynamic for unknown
```

### 2. Graph Building

**Old (hologram-onnx-core/ir_to_graph.rs):**

```rust
use hologram_compiler::ir::{Function, OpBuilder};

fn convert_to_ir(graph: &GraphProto) -> Function {
    let mut builder = OpBuilder::new();
    // ...
}
```

**New (use hologram-ir):**

```rust
use hologram_ir::{GraphBuilder, OperationGraph, NodeIndex};

fn convert_to_ir(graph: &GraphProto) -> Result<OperationGraph, OnnxError> {
    let mut builder = GraphBuilder::new();
    // ...
    builder.build()
}
```

### 3. Operation Translators

**Old (hologram-onnx-ops/activation.rs):**

```rust
pub fn translate_relu(ctx: &mut Context, node: &NodeProto) -> Result<()> {
    let input = ctx.get_input(&node.input[0])?;
    let output = ctx.op_builder.relu(input)?;
    ctx.set_output(&node.output[0], output);
    Ok(())
}
```

**New (minimal change - update builder type):**

```rust
pub fn translate_relu(ctx: &mut ConversionContext, node: &NodeProto) -> Result<(), OnnxError> {
    let input = ctx.get_input(&node.input[0])?;
    let output = ctx.builder.relu(input)?;  // builder is now GraphBuilder
    ctx.set_output(&node.output[0], output);
    Ok(())
}
```

### 4. Compilation

**Old:**

```rust
use hologram_compiler::compile;

let ir = convert_to_ir(&onnx_graph)?;
let holo = compile(ir)?;
```

**New:**

```rust
use hologram_compiler::compile_graph;

let graph = convert_to_ir(&onnx_graph)?;
let holo = compile_graph(&graph)?;
```

---

## Flattening Steps

### Phase 1: Create New Structure

```bash
cd hologram-onnx

# Keep config crate unchanged
# Create flattened main crate
mkdir -p crates/hologram-onnx/src/ops
mkdir -p crates/hologram-onnx/src/cli
mkdir -p crates/hologram-onnx/proto
```

### Phase 2: Move Files

```bash
# From spec
cp crates/hologram-onnx-spec/proto/onnx.proto3 crates/hologram-onnx/proto/
cp crates/hologram-onnx-spec/build.rs crates/hologram-onnx/

# From core
cp crates/hologram-onnx-core/src/parser.rs crates/hologram-onnx/src/
cp crates/hologram-onnx-core/src/weights.rs crates/hologram-onnx/src/
cp crates/hologram-onnx-core/src/error.rs crates/hologram-onnx/src/

# From ops (all translators)
cp crates/hologram-onnx-ops/src/*.rs crates/hologram-onnx/src/ops/

# From cli
cp crates/hologram-onnx-cli/src/*.rs crates/hologram-onnx/src/cli/
```

### Phase 3: Update Imports

Find and replace across all moved files:

| Old Import                     | New Import                                    |
| ------------------------------ | --------------------------------------------- |
| `hologram_compiler::shapes::*` | `hologram_ir::{Shape, Dim}`                   |
| `hologram_compiler::ir::*`     | `hologram_ir::{GraphBuilder, OperationGraph}` |
| `hologram_onnx_core::*`        | `crate::*`                                    |
| `hologram_onnx_ops::*`         | `crate::ops::*`                               |
| `hologram_onnx_spec::*`        | `crate::proto::*`                             |

### Phase 4: Create New lib.rs

```rust
// crates/hologram-onnx/src/lib.rs
mod proto;
mod error;
mod parser;
mod weights;
mod converter;
mod context;
pub mod ops;

#[cfg(feature = "cli")]
pub mod cli;

pub use error::OnnxError;
pub use hologram_ir::OperationGraph;

use std::path::Path;

pub struct OnnxModel {
    pub graph: OperationGraph,
    pub opset_version: i64,
    pub producer: String,
}

pub fn load_onnx(path: impl AsRef<Path>) -> Result<OnnxModel, OnnxError> {
    let bytes = std::fs::read(path)?;
    load_onnx_bytes(&bytes)
}

pub fn load_onnx_bytes(bytes: &[u8]) -> Result<OnnxModel, OnnxError> {
    let model = parser::parse(bytes)?;
    converter::convert_model(&model)
}

impl OnnxModel {
    pub fn compile(&self) -> Result<Vec<u8>, OnnxError> {
        hologram_compiler::compile_graph(&self.graph)
            .map_err(OnnxError::from)
    }
}
```

### Phase 5: Update Workspace Cargo.toml

```toml
# hologram-onnx/Cargo.toml
[workspace]
members = [
    "crates/hologram-onnx",
    "crates/hologram-onnx-config",
]
resolver = "2"

[workspace.dependencies]
hologram-ir = { path = "/hologram/crates/ir" }
hologram-compiler = { path = "/hologram/crates/compiler" }
prost = "0.13"
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
```

### Phase 6: Remove Old Crates

```bash
rm -rf crates/hologram-onnx-core
rm -rf crates/hologram-onnx-ops
rm -rf crates/hologram-onnx-spec
rm -rf crates/hologram-onnx-cli
```

---

## What hologram-ir Must Provide

The 80 op translators use these builder methods. Mark which exist vs need adding:

### Already in hologram-ir

- `input()`, `output()`, `constant_tensor()`
- `add()`, `sub()`, `mul()`, `div()`, `pow()`
- `matmul()`, `gemm()`
- `relu()`, `sigmoid()`, `tanh()`, `gelu()`
- `reshape()`, `transpose()`, `squeeze()`, `unsqueeze()`
- `concat()`, `split()`, `slice()`
- `reduce_sum()`, `reduce_mean()`, `reduce_max()`, `reduce_min()`
- `conv2d()`, `batch_norm()`, `layer_norm()`
- `max_pool2d()`, `avg_pool2d()`, `global_avg_pool()`

### To Add (P0 - Transformers)

- `gather(data, indices, axis)`
- `softmax(input, axis)`
- `where_select(cond, x, y)`
- `cast(input, dtype)`
- `clip(input, min, max)`
- `erf(input)`

### To Add (P1 - CNNs)

- `pad(input, pads, mode, value)`
- `resize(input, scales, sizes, mode)`
- `conv2d` needs `groups` parameter

---

## Testing After Restructure

```bash
# Build workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Test specific model
cargo test --package hologram-onnx -- test_mnist

# Run with config crate features
cargo test --workspace --features image-output,audio-output
```

---

## Summary

| Before                     | After                         |
| -------------------------- | ----------------------------- |
| 5 crates                   | 2 crates                      |
| ~25k lines scattered       | Config (5.7k) + Runtime (19k) |
| Old hologram-compiler deps | New hologram-ir deps          |
| Complex workspace          | Simple workspace              |

**What we keep:**

- ✅ Config crate (5,717 lines) - untouched
- ✅ 80 op translators - updated imports
- ✅ Protobuf handling - moved
- ✅ Weight deduplication - moved
- ✅ CLI commands - moved

**What we update:**

- Shape system → `hologram_ir::{Shape, Dim}`
- Graph building → `hologram_ir::GraphBuilder`
- Compilation → `hologram_compiler::compile_graph`
