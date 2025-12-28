# hologram-onnx

An ONNX runtime implementation using Hologram as the execution backend.

## Overview

This project provides a modular ONNX runtime that separates specification from implementation, enabling easy operator development and composition.

## Architecture

```
hologram-onnx/
├── crates/
│   ├── hologram-onnx-spec/     # ONNX protobuf definitions (this crate)
│   └── hologram-onnx-runtime/  # Operator implementations (future)
```

### Crates

| Crate | Description |
|-------|-------------|
| `hologram-onnx-spec` | Pure ONNX protobuf definitions compiled from the official specification |

## Quick Start

### Requirements

- Rust 2024 edition
- `protoc` (Protocol Buffers compiler)

### Building

```bash
cargo build
```

### Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
hologram-onnx-spec = { path = "crates/hologram-onnx-spec" }
```

Access ONNX types:

```rust
use hologram_onnx_spec::{ModelProto, GraphProto, NodeProto, TensorProto};

// Load an ONNX model
let model: ModelProto = /* ... */;

// Iterate through graph nodes
for node in &model.graph.unwrap().node {
    println!("Op: {}", node.op_type);
}
```

## Development

```bash
# Build all crates
cargo build

# Run tests
cargo test

# Check for issues
cargo clippy

# Generate documentation
cargo doc --open
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
