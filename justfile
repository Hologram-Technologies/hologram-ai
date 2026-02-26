# hologram-onnx Justfile
# Run `just` or `just --list` to see all available commands

# Default command - show available recipes
default:
    @just --list

# Build the project
build:
    cargo build

# Build with ONNX features enabled (excludes broken gguf/safetensors)
build-all:
    cargo build -p hologram-ai --features="onnx"

# Build in release mode
build-release:
    cargo build --release -p hologram-ai --features="onnx"

# Build the CLI binary
build-cli:
    cargo build -p hologram-ai --features="onnx"

# Run all tests (uses nextest for faster parallel execution)
test:
    cargo nextest run --workspace --no-fail-fast

# Run tests with ONNX features
test-all:
    cargo test -p hologram-ai-onnx
    cargo test -p hologram-ai-common
    cargo test -p hologram-ai --features="onnx"

# Run tests with output
test-verbose:
    cargo test -- --nocapture

# Run specific test by name
test-one TEST:
    cargo test {{TEST}} -- --nocapture

# Run integration tests only
test-integration:
    cargo test --test '*'

# Run unit tests only
test-unit:
    cargo test --lib

# Run benchmarks
bench:
    cargo bench

# Check code without building
check:
    cargo check

# Run clippy linter (excludes broken gguf/safetensors)
clippy:
    cargo clippy --all-targets -p hologram-ai-onnx -p hologram-ai-common -p hologram-ai

# Fix clippy warnings automatically
clippy-fix:
    cargo clippy --fix --all-targets -p hologram-ai-onnx -p hologram-ai-common -p hologram-ai --allow-dirty

# Format code
fmt:
    cargo fmt

# Check formatting without modifying files
fmt-check:
    cargo fmt --check

# Run all quality checks (fmt, clippy, test)
ci: fmt-check clippy test

# Clean build artifacts
clean:
    cargo clean

# Generate documentation
doc:
    cargo doc --no-deps -p hologram-ai-onnx -p hologram-ai-common -p hologram-ai

# Generate and open documentation in browser
doc-open:
    cargo doc --no-deps --open -p hologram-ai-onnx

# Run cargo fix to apply automatic fixes
fix:
    cargo fix --allow-dirty

# Update dependencies
update:
    cargo update

# Check for outdated dependencies
outdated:
    cargo outdated

# Install the CLI binary
install:
    cargo install --path crates/hologram-ai

# Compile an ONNX model to .holo format
compile MODEL OUTPUT:
    cargo run --release -p hologram-ai --features="onnx" -- compile {{MODEL}} -o {{OUTPUT}}

# Run an ONNX model
run MODEL:
    cargo run --release -p hologram-ai --features="onnx" -- run {{MODEL}}

# Show info about an ONNX model
info MODEL:
    cargo run -p hologram-ai --features="onnx" -- info {{MODEL}}

# Download a model from HuggingFace
download MODEL:
    cargo run -p hologram-ai --features="onnx" -- download {{MODEL}}

# Validate an ONNX model
validate MODEL:
    cargo run -p hologram-ai --features="onnx" -- validate {{MODEL}}

# Run with backtrace enabled
run-trace ARGS:
    RUST_BACKTRACE=1 cargo run -p hologram-ai --features="onnx" -- {{ARGS}}

# Profile with cargo flamegraph (requires cargo-flamegraph)
profile ARGS:
    cargo flamegraph -- {{ARGS}}

# Measure code coverage (requires cargo-tarpaulin)
coverage:
    cargo tarpaulin --out Html --output-dir coverage -p hologram-ai-onnx

# Run miri for undefined behavior detection (requires cargo-miri)
miri:
    cargo +nightly miri test -p hologram-ai-onnx

# Audit dependencies for security vulnerabilities (requires cargo-audit)
audit:
    cargo audit

# Expand macros for a file
expand FILE:
    cargo expand --lib {{FILE}}

# Show the size of the compiled binary
bloat:
    cargo bloat --release -p hologram-ai

# Check for unused dependencies (requires cargo-udeps)
udeps:
    cargo +nightly udeps -p hologram-ai-onnx

# Watch for changes and run tests
watch:
    cargo watch -x "test -p hologram-ai-onnx"

# Watch for changes and run specific command
watch-cmd CMD:
    cargo watch -x "{{CMD}}"

# Release workflow - build, test, and create optimized binary
release: clean fmt clippy test build-release
    @echo "✓ Release build complete!"
    @ls -lh target/release/hologram-ai

# Development workflow - quick iteration
dev: fmt build test
    @echo "✓ Development build complete!"

# Full CI workflow - comprehensive checks
full-ci: clean fmt-check clippy test-all doc build-release
    @echo "✓ Full CI checks passed!"

# Count lines of code
loc:
    @echo "Source code:"
    @find crates -name '*.rs' | xargs wc -l | tail -1
    @echo "\nTests:"
    @find crates -path '*/tests/*.rs' | xargs wc -l | tail -1

# Show project statistics
stats:
    @echo "=== Project Statistics ==="
    @echo "Source files: $(find crates -name '*.rs' | wc -l)"
    @echo "Test files: $(find crates -path '*/tests/*.rs' | wc -l)"
    @echo "Total lines: $(find crates -name '*.rs' | xargs wc -l | tail -1 | awk '{print $1}')"
    @echo "\n=== Crate Info ==="
    @cargo tree --depth 1 -p hologram-ai-onnx

# Quick test of core functionality
quick-test:
    cargo test -p hologram-ai-onnx --lib

# Prepare for commit - run all checks
pre-commit: fmt clippy test
    @echo "✓ Ready to commit!"

# Create a new git commit with conventional format
commit MSG:
    git add -A
    git commit -m "{{MSG}}"

# Show git status in short format
status:
    git status --short

# Show recent git commits
log:
    git log --oneline -10

# Create release build and show binary info
binary-info: build-release
    @echo "=== Binary Information ==="
    @ls -lh target/release/hologram-ai
    @file target/release/hologram-ai
    @echo "\n=== Dependencies ==="
    @ldd target/release/hologram-ai 2>/dev/null || otool -L target/release/hologram-ai 2>/dev/null || echo "Not a dynamic binary"

# T5 specific commands
t5-compile:
    cargo run --release -p hologram-ai --features="onnx" -- compile /workspace/models/t5-small/encoder_model.onnx --output /workspace/models/t5-small/compiled/encoder.holo
    cargo run --release -p hologram-ai --features="onnx" -- compile /workspace/models/t5-small/decoder_model.onnx --output /workspace/models/t5-small/compiled/decoder.holo

t5-run PROMPT:
    cargo run --release -p hologram-ai --features="onnx" -- run --config examples/T5/t5.toml --prompt "{{PROMPT}}"

# ResNet classification
resnet-compile:
    cargo run --release -p hologram-ai --features="onnx" -- compile /workspace/models/resnet18/resnet18.onnx --output /workspace/models/resnet18/resnet18.holo

resnet-run IMAGE:
    cargo run --release -p hologram-ai --features="onnx" -- run --config examples/ResNet/resnet.toml --image {{IMAGE}}
