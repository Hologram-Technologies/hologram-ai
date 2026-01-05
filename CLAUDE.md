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

## Documentation Guidelines

### Working Documents Location
- **ALL working markdown files MUST go in `/workspace/docs/working/`** unless otherwise specified
- Keep `/workspace/docs/working/implementation.md` as the active TODO tracker
- Example config files go in `/workspace/configs/examples/`
- Planning documents should reference implementation docs from `docs/working/`

### Code Quality Standards

**CRITICAL: These standards are MANDATORY and NON-NEGOTIABLE**

1. **NO TODOs, Placeholders, or Stubs**
   - Every function MUST be fully implemented
   - No `unimplemented!()` macros
   - No `todo!()` macros
   - No `panic!("not implemented")` or similar
   - All edge cases must be handled

2. **Complete Implementations**
   - Functions must do what they claim to do
   - No shortcuts or partial implementations
   - All error paths must be handled
   - No temporary workarounds

3. **Tests Required**
   - Write tests for EVERY module and function
   - Unit tests in module files or `tests/` subdirectory
   - Integration tests in top-level `tests/` directory
   - Include edge cases and error conditions
   - Test symbolic shapes with variable dimensions

4. **Documentation**
   - All public APIs MUST have rustdoc comments
   - Include examples in rustdoc for non-trivial functions
   - Document panics, errors, and safety considerations
   - Explain symbolic shape handling where applicable

5. **Error Handling**
   - Use proper error types (thiserror, anyhow)
   - No `unwrap()` in production code (use `?` or proper error handling)
   - No `expect()` unless truly impossible conditions
   - Provide helpful error messages

### Testing Requirements

**Unit Tests**:
- For every module in `src/`
- Test all public functions
- Test error conditions
- Test edge cases (empty inputs, large inputs, etc.)
- Test with symbolic shapes (variable batch, seq_len)

**Integration Tests**:
- In `tests/` directory for each crate
- Test full compilation pipelines
- Test multi-operation graphs
- Test with real ONNX models (MNIST, ResNet, etc.)

**Symbolic Shape Tests**:
- CRITICAL: Validate variable batch sizes
- CRITICAL: Validate variable sequence lengths
- Test shape inference propagation
- Test dimension expressions (Conv output dims)

**Memory Tests**:
- Ensure no OOM with large models
- Profile memory usage during compilation
- Test graph partitioning with 3000+ node graphs

**E2E Tests**:
- Full workflow: ONNX → .holo → execution
- Compile with hologram-onnx CLI
- Run with hologram CLI
- Verify output correctness

### ONNX Operation Implementation Requirements

**CRITICAL: When implementing or updating ONNX operations, you MUST:**

1. **Write comprehensive tests** for every operation translator:
   - Test normal cases with various input shapes and data types
   - Test edge cases (empty inputs, scalars, large tensors)
   - Test error conditions (wrong input count, invalid attributes)
   - Test constant folding paths (if applicable)
   - Tests should go in the `#[cfg(test)] mod tests {}` section of the same file

2. **Implement constant folding** when inputs are constants:
   - Many ONNX operations (Shape, Gather, Cast, Concat, Unsqueeze, etc.) should perform constant folding
   - If all inputs are `NodeOp::Constant`, compute the result at compile time and return a new `Constant`
   - This enables shape inference chains to collapse: Shape → Gather → Cast → Range → all Constants
   - Constant folding is essential for handling dynamic shape computations in models like T5

3. **Example operation implementations with tests:**
   - `src/ops/advanced.rs` - Range, Cast (with constant folding + tests)
   - `src/ops/shape.rs` - Unsqueeze, Concat, Reshape (with constant folding + tests)
   - `src/ops/constant.rs` - Shape, ConstantOfShape (with constant folding + tests)
   - `src/ops/indexing.rs` - Gather (with constant folding)

4. **Verify constant folding works**:
   - Create a test with constant inputs
   - Assert the result is a `NodeOp::Constant`
   - Verify the constant data matches expected output

### Implementation Workflow

1. **Read existing code first** - Never modify without understanding
2. **Write tests first** - TDD approach preferred
3. **Implement fully** - No TODOs or stubs
4. **Verify with tests** - All tests must pass
5. **Document public APIs** - Rustdoc for all public items
6. **Fix all warnings and errors** - Before completing any task:
   - Run `cargo check --all` to verify no compilation errors
   - Run `cargo clippy --all-targets` to fix all warnings
   - No unused imports, variables, or dead code without explicit `#[allow(...)]`
   - All warnings must be addressed before marking a task complete
7. **Update TODO tracker** - Mark items complete in `docs/working/implementation.md`
