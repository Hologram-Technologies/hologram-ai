# Agent Workflows for hologram-onnx

This document describes workflows for AI agents working on the hologram-onnx project.

## Project Context

hologram-onnx is a Rust workspace building an ONNX runtime with Hologram as the execution backend. The architecture separates:

- **Spec crate**: Pure ONNX protobuf definitions (`hologram-onnx-spec`)
- **Runtime crate**: Operator implementations (future)

## Pure Hologram Architecture Principle

**CRITICAL: This is the foundational principle of hologram-onnx**

### Core Philosophy

**Everything runs through hologram.** The entire point of hologram is to be a unified computational compiler and runtime. This principle is non-negotiable.

### Implementation Rules

1. **No External Runtime Dependencies**
   - Do NOT add dependencies like `tokenizers`, `ndarray`, `candle` for runtime execution
   - All computational operations compile to hologram IR → .holo files
   - All execution goes through hologram backend

2. **When hologram_ir Lacks Operations**
   - Implement algorithms in **pure Rust** (std library only)
   - Document as bridge until hologram_ir gains operations
   - Example: SentencePiece tokenizer in pure Rust until hologram_ir supports string ops

3. **The Vision**
   ```
   Everything is a .holo file executed by hologram:
   ├── tokenizer.holo    ✅ Compiles now (stub IR, pure Rust runtime bridge)
   ├── encoder.holo      ✅ Full hologram execution
   ├── decoder.holo      ✅ Full hologram execution
   └── post_process.holo 🔄 Future
   ```

4. **Always Maintain**
   - Compilation to .holo format (even for bridges)
   - Path to full hologram_ir implementation
   - No external runtime dependencies for core functionality

See [CLAUDE.md](CLAUDE.md#pure-hologram-architecture-principle) for detailed guidelines.

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

6. **Function Signatures - Use Structs for Multiple Parameters**
   - When a function has more than 3-4 parameters, use a struct instead
   - Implement the **builder pattern** for configuration structs
   - This improves readability, maintainability, and allows for optional parameters
   - Example:

     ```rust
     // ❌ WRONG - Too many parameters
     fn compile_model(
         model_path: &Path,
         output_path: &Path,
         optimize: bool,
         target_device: Device,
         batch_size: usize,
         precision: Precision,
     ) -> Result<()> { ... }

     // ✅ CORRECT - Use a config struct with builder pattern
     pub struct CompileConfig {
         model_path: PathBuf,
         output_path: PathBuf,
         optimize: bool,
         target_device: Device,
         batch_size: usize,
         precision: Precision,
     }

     impl CompileConfig {
         pub fn builder() -> CompileConfigBuilder { ... }
     }

     fn compile_model(config: &CompileConfig) -> Result<()> { ... }
     ```

7. **Function Length - Keep Functions Under 50 Lines**
   - No function should exceed 50 lines of code
   - If a function is too long, break it down:
     - Extract helper functions for logical sub-tasks
     - Consider using traits to define composable behavior
     - Each extracted function should be independently testable
   - Benefits:
     - Easier to test individual pieces
     - Improved readability and maintenance
     - Better separation of concerns
   - Example:

     ```rust
     // ❌ WRONG - Monolithic function
     fn process_graph(graph: &Graph) -> Result<Output> {
         // 100+ lines of mixed concerns...
     }

     // ✅ CORRECT - Trait-based decomposition
     trait GraphProcessor {
         fn validate(&self, graph: &Graph) -> Result<()>;
         fn optimize(&self, graph: &Graph) -> Result<Graph>;
         fn compile(&self, graph: &Graph) -> Result<Output>;
     }

     impl GraphProcessor for MyProcessor {
         fn validate(&self, graph: &Graph) -> Result<()> { ... }  // ~15 lines
         fn optimize(&self, graph: &Graph) -> Result<Graph> { ... }  // ~20 lines
         fn compile(&self, graph: &Graph) -> Result<Output> { ... }  // ~25 lines
     }
     ```

8. **Documentation in README.md**
   - Every crate MUST have a README.md documenting:
     - Purpose and overview of the crate
     - Public API functions and their usage
     - Examples for common use cases
   - Update README.md when adding or modifying public functions
   - Keep documentation in sync with code changes

9. **Rust Best Practices**
   - **Prefer `&str` over `String`** in function parameters for flexibility
   - **Use `impl Trait`** for return types when the concrete type is an implementation detail
   - **Prefer iterators** over manual loops - use `.iter()`, `.map()`, `.filter()`, `.collect()`
   - **Use `?` operator** for error propagation instead of manual `match` or `unwrap()`
   - **Derive common traits** appropriately: `Debug`, `Clone`, `PartialEq`, `Eq`, `Hash`
   - **Use newtypes** for type safety (e.g., `struct UserId(u64)` instead of raw `u64`)
   - **Prefer `Cow<str>`** when a function may or may not need to allocate
   - **Use `#[must_use]`** on functions where ignoring the return value is likely a bug
   - **Avoid `clone()` unless necessary** - prefer borrowing and lifetimes
   - **Use `Default` trait** for types with sensible defaults
   - **Prefer composition over inheritance** - use traits for shared behavior
   - **Use `From`/`Into` traits** for type conversions instead of custom methods
   - Example patterns:

     ```rust
     // ✅ Prefer &str over String in parameters
     fn process_name(name: &str) -> String { ... }

     // ✅ Use impl Trait for return types
     fn get_items(&self) -> impl Iterator<Item = &Item> { ... }

     // ✅ Use newtypes for type safety
     pub struct BatchSize(pub usize);
     pub struct SequenceLength(pub usize);

     // ✅ Implement From for conversions
     impl From<OnnxModel> for HoloGraph {
         fn from(model: OnnxModel) -> Self { ... }
     }

     // ✅ Use Default for sensible defaults
     #[derive(Default)]
     pub struct CompileOptions {
         optimize: bool,
         debug_info: bool,
     }
     ```

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

## Tasks

Updates to every task must be documented with their status before completion of the task. Update the plan document with a table and each task with their completed state between:

- Not started
- In-Progress
- Pending implemention (no tests written)
- Completed

Tasks are only complete if all the workspace tests pass. You can run them with `just test`. A task is not complete until all of the tests pass.

As you're working through tasks, ensure you have a plan in `specs/plans` and keep it up to date throughout the feature build.