# Agent Workflows for hologram-onnx

This document describes workflows for AI agents working on the hologram-onnx project.

## Project Context

hologram-onnx is a Rust workspace building an ONNX runtime with Hologram as the execution backend. The architecture separates:

- **Spec crate**: Pure ONNX protobuf definitions (`hologram-onnx-spec`)
- **Runtime crate**: Operator implementations (future)

## Exploration Workflows

### Understanding the Codebase

1. Start with workspace structure:
   ```
   crates/
   └── hologram-onnx-spec/
       ├── Cargo.toml      # Dependencies
       ├── build.rs        # Proto compilation
       ├── src/lib.rs      # Entry point
       └── proto/onnx.proto3
   ```

2. Key entry points:
   - Root `Cargo.toml` - workspace configuration
   - `crates/*/Cargo.toml` - per-crate configuration
   - `crates/*/src/lib.rs` - public API

### Finding ONNX Types

The ONNX proto defines many types. Common ones:

| Type | Description |
|------|-------------|
| `ModelProto` | Top-level model container |
| `GraphProto` | Computation graph (nodes, inputs, outputs) |
| `NodeProto` | Individual operation |
| `TensorProto` | Tensor data and metadata |
| `AttributeProto` | Node attribute (weights, params) |
| `ValueInfoProto` | Type and shape info for values |

Search in generated code:
```bash
# Find in build output
cargo build 2>&1 | head -1  # Get OUT_DIR path
# Or check target/debug/build/hologram-onnx-spec-*/out/onnx.rs
```

## Implementation Patterns

### Adding a New Crate

1. Create structure:
   ```bash
   mkdir -p crates/<name>/src
   ```

2. Add `Cargo.toml` with workspace inheritance:
   ```toml
   [package]
   name = "<name>"
   version.workspace = true
   edition.workspace = true

   [dependencies]
   hologram-onnx-spec = { path = "../hologram-onnx-spec" }
   ```

3. Create `src/lib.rs`

### Adding Proto Files

1. Place `.proto3` file in `crates/<crate>/proto/`
2. Add `build.rs`:
   ```rust
   fn main() -> std::io::Result<()> {
       prost_build::Config::new()
           .compile_protos(&["proto/file.proto3"], &["proto/"])?;
       Ok(())
   }
   ```
3. Include in `lib.rs`:
   ```rust
   include!(concat!(env!("OUT_DIR"), "/package.rs"));
   ```

## Testing Requirements

### Before Submitting Changes

1. Build succeeds: `cargo build`
2. Tests pass: `cargo test`
3. No clippy warnings: `cargo clippy --all-targets`
4. Docs generate: `cargo doc --no-deps`

### Writing Tests

- Unit tests go in `src/*.rs` files
- Integration tests go in `tests/` directories
- Use `#[cfg(test)]` modules for unit tests

## Common Tasks

### Update ONNX Proto

```bash
curl -sL https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3 \
  -o crates/hologram-onnx-spec/proto/onnx.proto3
cargo build  # Verify compilation
```

### Check Generated Types

```bash
cargo doc --no-deps -p hologram-onnx-spec --open
```

### Debug Build Issues

Proto compilation errors:
1. Check `protoc` is installed
2. Verify proto file syntax
3. Check `build.rs` paths are correct

Type errors in generated code:
1. Check prost version compatibility
2. Verify clippy allows are in place
