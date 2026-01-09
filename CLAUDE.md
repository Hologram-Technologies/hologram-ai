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

## Pure Hologram Architecture Principle

**CRITICAL: This is the foundational principle of hologram-onnx**

### Core Philosophy

**Everything runs through hologram.** The entire point of hologram is to be a unified computational compiler and runtime. This principle is non-negotiable.

### What This Means

1. **No External Runtime Dependencies for Core Functionality**
   - Do NOT add dependencies like `tokenizers`, `ndarray`, `candle`, etc. for runtime execution
   - All computational operations must compile to hologram IR
   - All execution must go through hologram backend
   - External crates are acceptable ONLY for:
     - Build-time tools (prost-build, etc.)
     - Development utilities (testing, benchmarking)
     - Data loading/parsing (serde, image loading, etc.)

2. **Compilation Target: .holo Files**
   - Tokenizers compile to .holo → execute on hologram backend
   - Models compile to .holo → execute on hologram backend
   - Post-processing compiles to .holo → execute on hologram backend
   - Everything is a computational graph executed by hologram

3. **Temporary Pure Rust Implementations**
   - When hologram_ir lacks necessary operations (Gather, String ops, etc.):
     - Implement algorithms in **pure Rust** (std library only)
     - Document as bridge until hologram_ir gains operations
     - Plan migration path to full hologram_ir implementation
   - Example: SentencePiece tokenizer implemented in pure Rust until hologram_ir supports string operations

4. **The Vision**
   ```
   Everything is a .holo file:
   ├── tokenizer.holo       (text → tokens)   Future: Full hologram_ir
   ├── encoder.holo         (tokens → hidden)  ✅ Working now
   ├── decoder.holo         (hidden → logits)  ✅ Working now
   └── post_process.holo    (logits → output)  Future: Full hologram_ir

   All execute on hologram backend.
   All benefit from hologram optimizations.
   All are config-driven and cacheable.
   ```

5. **Implementation Guidelines**
   - When implementing new functionality (tokenizers, custom ops, etc.):
     - First: Check if hologram_ir operations exist
     - If YES: Implement via hologram IR compilation
     - If NO: Implement in pure Rust (std only), document as bridge
     - Never: Add external runtime dependencies
   - Create issues/plans for hologram_ir enhancements needed
   - Maintain compilation to .holo even for bridge implementations

### Why This Matters

- **Unified optimization**: All operations benefit from hologram's SIMD kernels
- **Zero-copy execution**: Hologram workspace management
- **Consistent architecture**: One backend, one format, one execution model
- **Future-proof**: When hologram_ir gains operations, migrate seamlessly

### Examples

**✅ CORRECT - Pure Rust Implementation**:
```rust
// Implement SentencePiece unigram algorithm in pure Rust
// Uses only std::collections, no external tokenizer crates
impl SentencePieceTokenizer {
    fn tokenize_unigram(&self, text: &str) -> Vec<u32> {
        // Full Viterbi implementation in pure Rust
        // ...
    }
}
```

**❌ WRONG - External Runtime Dependency**:
```rust
use tokenizers::Tokenizer;  // NO! External runtime dep

impl SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.hf_tokenizer.encode(text, false)  // NO!
    }
}
```

## Documentation Guidelines

### Working Documents Location
- **ALL working markdown files MUST go in `/workspace/docs/working/`** unless otherwise specified
- Keep `/workspace/docs/working/implementation.md` as the active TODO tracker
- Example config files go in `/workspace/configs/examples/`
- Planning documents should reference implementation docs from `docs/working/`

### Code Quality Standards

**CRITICAL: These standards are MANDATORY and NON-NEGOTIABLE**

#### Production-Ready Code ONLY

**ABSOLUTE REQUIREMENT: Every piece of code in this project MUST be production-ready.**

- **NO stubs** - Period. Nothing is a stub.
- **NO TODOs** - Every function is complete.
- **NO placeholders** - All code is real, working code.
- **NO "simplistic" implementations** - Full, proper implementations only.
- **NO "in a real implementation" comments** - This IS the real implementation.
- **NO shortcuts** - Do it right or don't do it.

Any code that contains phrases like "in production you would...", "a real implementation would...", "simplified for demonstration", or similar disclaimers is **UNACCEPTABLE**. If you're writing it, write it properly. If a feature isn't ready, don't include it at all.

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

3. **Tests Required - Maximum Coverage**
   - **Write tests for ALL methods and functions** - aim for the highest test coverage possible
   - Every public function MUST have at least one test
   - Every private function with non-trivial logic MUST have tests
   - Unit tests in module files or `tests/` subdirectory
   - Integration tests in top-level `tests/` directory
   - Include edge cases and error conditions
   - Test symbolic shapes with variable dimensions
   - Test all code paths, including error paths
   - No code should be merged without corresponding tests

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
