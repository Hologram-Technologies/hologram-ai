# hologram-onnx justfile
# Run `just` or `just --list` to see all available commands

# Default command - show available recipes
default:
    @just --list

# Build the project
build:
    cargo build

# Build with all features enabled
build-all:
    cargo build --all-features

# Build in release mode
build-release:
    cargo build --release

# Build the CLI binary
build-cli:
    cargo build --bin hologram-onnx

# Run all tests
test:
    cargo test --workspace

# Run tests with all features
test-all:
    cargo test --all-features

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

# Run clippy linter
clippy:
    cargo clippy --all-targets --all-features

# Fix clippy warnings automatically
clippy-fix:
    cargo clippy --fix --all-targets --all-features --allow-dirty

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
    cargo doc --no-deps --all-features

# Generate and open documentation in browser
doc-open:
    cargo doc --no-deps --all-features --open

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
    cargo install --path .

# Compile an ONNX model to .holo format
compile MODEL OUTPUT:
    cargo run --bin hologram-onnx -- compile {{MODEL}} -o {{OUTPUT}}

# Run an ONNX model
run MODEL:
    cargo run --bin hologram-onnx -- run {{MODEL}}

# Show info about an ONNX model
info MODEL:
    cargo run --bin hologram-onnx -- info {{MODEL}}

# Download a model from HuggingFace
download MODEL:
    cargo run --bin hologram-onnx -- download {{MODEL}}

# Validate an ONNX model
validate MODEL:
    cargo run --bin hologram-onnx -- validate {{MODEL}}

# Run with backtrace enabled
run-trace ARGS:
    RUST_BACKTRACE=1 cargo run -- {{ARGS}}

# Profile with cargo flamegraph (requires cargo-flamegraph)
profile ARGS:
    cargo flamegraph -- {{ARGS}}

# Measure code coverage (requires cargo-tarpaulin)
coverage:
    cargo tarpaulin --out Html --output-dir coverage

# Run miri for undefined behavior detection (requires cargo-miri)
miri:
    cargo +nightly miri test

# Audit dependencies for security vulnerabilities (requires cargo-audit)
audit:
    cargo audit

# Expand macros for a file
expand FILE:
    cargo expand --lib {{FILE}}

# Show the size of the compiled binary
bloat:
    cargo bloat --release

# Check for unused dependencies (requires cargo-udeps)
udeps:
    cargo +nightly udeps

# Watch for changes and run tests
watch:
    cargo watch -x test

# Watch for changes and run specific command
watch-cmd CMD:
    cargo watch -x "{{CMD}}"

# Release workflow - build, test, and create optimized binary
release: clean fmt clippy test build-release
    @echo "✓ Release build complete!"
    @ls -lh target/release/hologram-onnx

# Development workflow - quick iteration
dev: fmt build test
    @echo "✓ Development build complete!"

# Full CI workflow - comprehensive checks
full-ci: clean fmt-check clippy test-all doc build-release
    @echo "✓ Full CI checks passed!"

# Count lines of code
loc:
    @echo "Source code:"
    @find src -name '*.rs' | xargs wc -l | tail -1
    @echo "\nTests:"
    @find tests -name '*.rs' | xargs wc -l | tail -1

# Show project statistics
stats:
    @echo "=== Project Statistics ==="
    @echo "Source files: $(find src -name '*.rs' | wc -l)"
    @echo "Test files: $(find tests -name '*.rs' | wc -l)"
    @echo "Total lines: $(find src tests -name '*.rs' | xargs wc -l | tail -1 | awk '{print $1}')"
    @echo "\n=== Crate Info ==="
    @cargo tree --depth 1

# Quick test of core functionality
quick-test:
    cargo test --lib test_shapes test_parser test_translator

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
    @ls -lh target/release/hologram-onnx
    @file target/release/hologram-onnx
    @echo "\n=== Dependencies ==="
    @ldd target/release/hologram-onnx 2>/dev/null || otool -L target/release/hologram-onnx 2>/dev/null || echo "Not a dynamic binary"
