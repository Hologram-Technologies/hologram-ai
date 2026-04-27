# hologram-ai — build & maintenance commands

set dotenv-load := true

# Default recipe: list all available recipes
default:
    @just --list

# Full CI: format check, clippy, tests
ci: fmt-check clippy test

# Run all tests
test:
    cargo nextest run --workspace

# Run clippy with deny warnings
clippy:
    cargo clippy --workspace -- -D warnings

# Format all code
fmt:
    cargo fmt --all

# Check formatting (no changes)
fmt-check:
    cargo fmt --all -- --check

# Build all crates
build:
    cargo build --workspace

# Build in release mode
release:
    cargo build --workspace --release

# Clean build artifacts
clean:
    cargo clean

# Pull latest architecture docs
sync:
    holoarch pull

# Check architecture conformance
check:
    holoarch check

# Generate test fixtures (ONNX models + quant golden vectors)
gen-fixtures:
    python3 scripts/gen-fixtures.py
    python3 scripts/gen-quant-vectors.py

# Run conformance tests (Tier 1: no external deps)
conformance:
    cargo test -p hologram-ai-conformance

# Run ORT conformance tests (Tier 2: requires ORT_DYLIB_PATH)
conformance-ort:
    cargo test -p hologram-ai-conformance --features=conformance

# Run validate integration tests
conformance-validate:
    cargo test -p hologram-ai --test validate_test

# Run all conformance tiers (Tier 1 + 2 + validate)
conformance-all: conformance conformance-ort conformance-validate

# Run tests for hologram base crate (sibling dependency)
test-base:
    cd ../hologram && cargo test --workspace

# Run clippy on hologram base crate
clippy-base:
    cd ../hologram && cargo clippy --workspace -- -D warnings

# Full CI across both repos
ci-all: ci test-base

# Cut a desktop release: tags `desktop-vVERSION` and pushes it, which
# triggers .github/workflows/release-desktop.yml on GitHub. The workflow
# builds the universal macOS .dmg and creates a draft GitHub Release.
#
# Preconditions checked:
#   - working tree clean
#   - on `main`, fully in sync with `origin/main`
#   - tag does not already exist (locally or remote)
#   - version matches apps/desktop/src-tauri/tauri.conf.json
#
# Usage: just release-desktop 0.1.0
release-desktop VERSION:
    #!/usr/bin/env bash
    set -euo pipefail
    TAG="desktop-v{{VERSION}}"

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "error: working tree not clean. Commit or stash changes first." >&2
        exit 1
    fi
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$BRANCH" != "main" ]; then
        echo "error: not on main (currently on $BRANCH)." >&2
        exit 1
    fi
    git fetch origin main --quiet
    if ! git merge-base --is-ancestor origin/main HEAD; then
        echo "error: local main is behind origin/main. Pull first." >&2
        exit 1
    fi
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)
    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "error: local main has unpushed commits. Push first." >&2
        exit 1
    fi

    if git rev-parse -q --verify "refs/tags/$TAG" > /dev/null; then
        echo "error: tag $TAG already exists locally." >&2
        exit 1
    fi
    if git ls-remote --tags origin "$TAG" | grep -q .; then
        echo "error: tag $TAG already exists on origin." >&2
        exit 1
    fi

    CONF_VERSION=$(grep -E '"version":' apps/desktop/src-tauri/tauri.conf.json | head -1 | sed -E 's/.*"version": *"([^"]+)".*/\1/')
    if [ "$CONF_VERSION" != "{{VERSION}}" ]; then
        echo "error: tauri.conf.json version is $CONF_VERSION, refusing to tag as {{VERSION}}." >&2
        echo "       Bump the version field in apps/desktop/src-tauri/tauri.conf.json first." >&2
        exit 1
    fi

    echo "Tagging $TAG ..."
    git tag -a "$TAG" -m "hologram-ai-desktop $TAG"
    git push origin "$TAG"

    echo ""
    echo "Pushed $TAG. Track the build at:"
    echo "  https://github.com/Hologram-Technologies/hologram-ai/actions/workflows/release-desktop.yml"
    echo ""
    echo "When the workflow completes, the draft release will appear at:"
    echo "  https://github.com/Hologram-Technologies/hologram-ai/releases"

# Trigger a release-desktop workflow run without tagging — useful for
# testing the build pipeline. Requires the `gh` CLI authenticated.
# Artifacts are uploaded to the workflow run page (no GitHub Release created).
release-desktop-dispatch:
    gh workflow run release-desktop.yml --ref main
