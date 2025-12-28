# Claude Instructions for hologram-onnx

## Project Overview

This is a Cargo workspace for an ONNX runtime using Hologram as the execution backend. The project separates ONNX protobuf definitions from operator implementations.

## Workspace Structure

```
hologram-onnx/
├── Cargo.toml              # Workspace manifest
├── crates/
│   └── hologram-onnx-spec/ # ONNX protobuf definitions
│       ├── Cargo.toml
│       ├── build.rs        # prost-build compilation
│       ├── src/lib.rs      # Re-exports generated types
│       └── proto/
│           └── onnx.proto3 # Vendored from official ONNX repo
```

## Build Commands

```bash
# Build everything
cargo build

# Build specific crate
cargo build -p hologram-onnx-spec

# Run tests
cargo test

# Check with clippy
cargo clippy --all-targets

# Generate docs
cargo doc --no-deps --open
```

## Key Files

| File | Purpose |
|------|---------|
| `crates/hologram-onnx-spec/proto/onnx.proto3` | Vendored ONNX protobuf definition |
| `crates/hologram-onnx-spec/build.rs` | Compiles proto to Rust at build time |
| `crates/hologram-onnx-spec/src/lib.rs` | Includes generated code with clippy suppression |

## Proto Compilation

The `hologram-onnx-spec` crate uses `prost-build` to compile `onnx.proto3` at build time:

1. `build.rs` runs during `cargo build`
2. prost generates `$OUT_DIR/onnx.rs`
3. `lib.rs` includes the generated file with `include!`

The generated code contains clippy warnings, so `#![allow(clippy::doc_overindented_list_items)]` is applied.

## Adding New Crates

1. Create directory: `crates/<crate-name>/`
2. Add `Cargo.toml` using workspace inheritance:
   ```toml
   [package]
   name = "<crate-name>"
   version.workspace = true
   edition.workspace = true
   ```
3. The `[workspace] members = ["crates/*"]` pattern auto-discovers crates

## Conventions

- Edition: 2024
- Use workspace dependencies where possible
- Proto files go in `proto/` subdirectories
- Build scripts handle proto compilation

## Dependencies

Workspace-level dependencies (defined in root `Cargo.toml`):
- `prost = "0.13"` - Protocol Buffers runtime
- `prost-types = "0.13"` - Well-known proto types
- `bytes = "1"` - Byte buffer handling

Build dependency (per-crate):
- `prost-build = "0.13"` - Proto compilation

## ONNX Proto Source

The `onnx.proto3` file is vendored from:
https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3

To update:
```bash
curl -sL https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto3 \
  -o crates/hologram-onnx-spec/proto/onnx.proto3
```
