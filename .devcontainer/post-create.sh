#!/bin/bash
set -e

echo "Running post-create setup for hologram-onnx..."

cd /workspace

# Verify Rust toolchain
echo "Verifying Rust toolchain..."
rustc --version
cargo --version

# Verify protobuf compiler (used by prost-build)
echo "Verifying protobuf compiler..."
protoc --version || echo "Warning: protoc not found (prost-build will use bundled version)"

# Cache Rust dependencies
echo "Caching Rust dependencies..."
cargo fetch --locked 2>/dev/null || cargo fetch

# Verify workspace compiles
echo "Verifying workspace compiles..."
cargo check

echo ""
echo "Post-create setup complete!"
echo ""
echo "Project: hologram-onnx"
echo "  ONNX runtime using Hologram as the execution backend"
echo ""
echo "Build commands:"
echo "  cargo build                  # Build all crates"
echo "  cargo test                   # Run tests"
echo "  cargo clippy --all-targets   # Run clippy"
echo "  cargo doc --no-deps --open   # Generate docs"
echo ""
echo "Ready to develop!"
