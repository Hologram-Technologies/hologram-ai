# hologram-ai task runner — the rewrite gate set.
# `just` with no args lists everything. `just vv` is the full local gate;
# CI mirrors it (.github/workflows/ci.yml).
#
# Toolchain note: on machines where Homebrew's cargo shadows rustup, export
# the pinned toolchain first (see AGENTS.md):
#   export PATH="$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin:$PATH"

set dotenv-load := true

default:
    @just --list

# Install the pre-push hook: the gate runs before every push.
install-hooks:
    @printf '#!/bin/bash\nset -e\necho "pre-push: running the gate (just vv)"\njust vv\n' > .git/hooks/pre-push
    @chmod +x .git/hooks/pre-push
    @echo "pre-push hook installed"

# ── Quality gates (each is also a CI gate) ──────────────────────────────────

fmt:
    cargo fmt --all --check
fmt-fix:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace --all-features

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# ── Portability ─────────────────────────────────────────────────────────────

# The portable core (schemas + bundle codec) builds no_std for wasm32.
portability:
    cargo check -p hologram-ai-core --no-default-features --target wasm32-unknown-unknown
    cargo check -p hologram-ai-bundle --no-default-features --target wasm32-unknown-unknown

# ── Architecture guards ─────────────────────────────────────────────────────

# No tensor/GPU/transformer dependencies, no public binary, no tless_store
# in bundle code paths.
arch-guards:
    ./scripts/arch-guards.sh

# Byte-identical double-build of the bundle + .holo fixture.
determinism:
    cargo test -p hologram-ai --test determinism

# Zero-alloc census around the steady-state inference path.
alloc-census:
    cargo test -p hologram-ai-r4 --test allocation_census

# ── Opt-in lanes (never in default CI) ──────────────────────────────────────

# Hermetic end-to-end compile of the committed tiny fixture through uor-r4
# (slow: teacher observation). Runs the full source → .holo → infer chain.
e2e-fixture:
    cargo test -p hologram-ai --test e2e -- --ignored --nocapture

# Live Hugging Face download + compile (network; pinned revision).
live-hf:
    HOLOGRAM_AI_LIVE_HF=1 cargo test -p hologram-ai-huggingface -- --ignored --nocapture

# ── The full local gate ─────────────────────────────────────────────────────

vv: fmt lint doc test portability arch-guards
    @echo "All gates green."
