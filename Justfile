# hologram-ai task runner — the docs-as-code / BDD / V&V gate set.
# `just` with no args lists everything. `just vv` is the full local gate;
# CI mirrors it job-for-job (.github/workflows/ci.yml).

set dotenv-load := true

default:
    @just --list

# Install the pre-push hook: the full V&V gate runs before every push.
install-hooks:
    @printf '#!/bin/bash\nset -e\necho "pre-push: running the V&V gate (just vv)"\njust vv\n' > .git/hooks/pre-push
    @chmod +x .git/hooks/pre-push
    @echo "pre-push hook installed"

# ── Quality gates (each is also a CI gate) ──────────────────────────────────

# Format check (CI uses --check; locally `just fmt-fix` rewrites).
fmt:
    cargo fmt --all --check
fmt-fix:
    cargo fmt --all

# Clippy across all targets. Warnings are errors.
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Unit + integration tests (includes the honesty meta-gate and the
# κ-materialization e2e witness).
test:
    cargo test --workspace

# Rustdoc must build clean (docs-as-code).
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# ── BDD / conceptual model / oracles ────────────────────────────────────────

# The Rust Gherkin suites, default lane (pure/local/network witnesses).
# The runner is model-driven and fails on any skipped or undefined step.
bdd:
    HOLOGRAM_AI_BDD_LANE=default cargo test -p hologram-ai-conformance --test bdd

# The ORT lane (needs ORT_DYLIB_PATH → ONNX Runtime v1.18.1).
bdd-ort:
    HOLOGRAM_AI_BDD_LANE=ort cargo test -p hologram-ai-conformance --features conformance --test bdd

# The model lane (needs the pinned real model on disk; see architectures.yml).
bdd-model:
    HOLOGRAM_AI_BDD_LANE=model cargo test -p hologram-ai-conformance --features conformance --test bdd

# The measured probes for `open` rows (non-gating; reported).
bdd-target:
    HOLOGRAM_AI_BDD_LANE=target cargo test -p hologram-ai-conformance --test bdd

# The honesty meta-gate: model ⇄ features ⇄ witnesses coverage + status discipline.
honesty:
    cargo test -p hologram-ai-conformance --test honesty -- --nocapture

# Every committed oracle artifact matches its recorded sha256 (offline).
oracles:
    cargo run -q -p xtask -- oracle-verify

# Every pinned upstream (git revs, HF revisions, release tags) is live (online).
pin-check:
    cargo run -q -p xtask -- pin-check

# Emit the conformance ledger.
report:
    cargo run -q -p xtask -- report

# ── Structural / conformance / portability axes ─────────────────────────────

# The substrate-contract witnesses: ZA, ZM, CE, CF, LW, IM (isolated binaries).
structural:
    cargo test --release -p hologram-ai-conformance --features=structural \
        --test structural_ce --test structural_za --test structural_zm \
        --test structural_cf --test structural_lw --test structural_im

# External-authority execution parity (needs ORT_DYLIB_PATH).
conformance-ort:
    cargo test --release -p hologram-ai-conformance --features=conformance

# The runtime core builds no_std on wasm + embedded; the browser binding
# builds with wasm-pack.
portability:
    cargo build --target wasm32-unknown-unknown -p hologram-ai-quant
    cargo build --target wasm32-unknown-unknown -p hologram-ai-tokenizer --no-default-features
    cargo build --target thumbv7em-none-eabi -p hologram-ai-quant
    cargo build --target thumbv7em-none-eabi -p hologram-ai-tokenizer --no-default-features
    cargo check --target wasm32-unknown-unknown -p hologram-ai-wasm

# Run the REAL substrate decode kernels compiled to the wasm target the browser
# ships (not just `cargo check`): the bare κ119/κ120 fused step, the resident-KV
# carry/steal at production head_dim 128, and the legacy-decode guard. The blind
# spot that let the v0.9.0 `unreachable` trap reach the deployed instance.
wasm-test:
    wasm-pack test --node crates/hologram-ai-wasm

# No canonical-instance constant leaks into generic code.
anti-hardcode:
    ./scripts/anti-hardcode.sh

# ── The browser journey (S4) ────────────────────────────────────────────────

# Build the wasm binding into the web app.
wasm:
    cd apps/web && pnpm wasm

# Install web deps.
web-install:
    cd apps/web && pnpm install --frozen-lockfile

# Web unit tests (vitest).
web-unit:
    cd apps/web && pnpm test

# The hermetic browser journey: download → compile → materialize → run →
# three-message handshake, in real Chromium against the fixture server.
journey:
    cd apps/web && pnpm bdd

# The live journey against the pinned SmolLM2 (network + weights; scheduled lane).
journey-live:
    cd apps/web && pnpm bdd:live

# ── The full local gate ─────────────────────────────────────────────────────

vv: fmt lint doc test bdd honesty oracles report anti-hardcode structural portability wasm-test web-install web-unit wasm journey pin-check
    @echo "V&V: all gates green."
