# AGENTS.md — hologram-ai

Guidance for agents (human or otherwise) working in this repository. The
normative architecture is `docs/rewrite-plan.md` + `docs/adrs/` (0001–0009);
`docs/archive/` is historical and non-normative.

## What this repo is

The production integration layer between model sources, `uor-r4`, and
Hologram: it downloads pinned Hugging Face (or local) model sources,
compiles them through `uor-r4` into R4G1 artifacts, packages them as
deterministic R4 inference bundles inside `.holo` `InferenceModel` layers,
and serves deterministic CPU-only inference through a Rust facade consumed
by the `hologram` CLI, FFI, and the Python/TypeScript SDKs.

It is **not** a tensor compiler and **not** an inference runtime owner:
all compilation, scoring, tokenization, status policy, and generation
algorithms live in `uor-r4`; all container/layer/archive machinery lives in
`hologram`. Never copy those algorithms into this repository.

## Workspace layout

- `crates/hologram-ai-core` — `no_std + alloc` portable schemas, canonical
  encoding, requests/completions, stable error categories (codes are ABI).
- `crates/hologram-ai-bundle` — `no_std + alloc` deterministic R4 inference
  bundle (schema v1) codec + validation.
- `crates/hologram-ai-r4` — the ONLY crate depending on `uor-r4`; private
  adapter. All current-upstream compatibility shims live here.
- `crates/hologram-ai-huggingface` — source acquisition (the ONLY
  networking/process-execution crate).
- `crates/hologram-ai` — public facade (compiler builder, application
  registry, sessions). No binary target exists or may be added.

Upstream patches are developed in sibling worktrees (`../uor-r4-facade`,
`../hologram-im`) and consumed by path until the upstream PRs merge; the
exact base revisions are pinned in `docs/rewrite-plan.md` and
`docs/upstream/`.

## Commands (daily drivers)

IMPORTANT: Homebrew's `/opt/homebrew/bin/cargo` shadows rustup on this
machine and ignores `rust-toolchain.toml`. Export the pinned toolchain first:

```bash
export PATH="$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin:$PATH"
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
cargo check -p hologram-ai-core --no-default-features --target wasm32-unknown-unknown
cargo check -p hologram-ai-bundle --no-default-features --target wasm32-unknown-unknown
```

All must be clean before every commit. (On machines where rustup shims are
on `PATH`, plain `cargo` works — `rust-toolchain.toml` pins 1.97.0.)

## Normative invariants (do not weaken)

- **R4G1 only**: no tensor/ONNX/transformer execution path, no `MatMul` in
  the deployed path, no GPU dependency, no runtime networking, no external
  inference provider, no `tless_store.bin` in bundles (ADR-0002).
- **Allocation**: zero heap allocations in `predict_next_into` and
  steady-state `generate_into` steps after initialization; enforced by the
  allocation-census test (ADR-0009).
- **Determinism**: identical pinned inputs ⇒ byte-identical bundle and
  `.holo`. No timestamps, machine paths, random IDs, or map-iteration order
  in canonical bytes.
- **Errors**: public failures are `AiError` with a stable `ErrorCategory`
  (codes 1–20; append-only). No `unwrap`/`expect`/panic on recoverable
  paths. `#![forbid(unsafe_code)]` everywhere.
- **Abstention** is a typed completion, never an exception, never a
  fabricated token. No sampling/temperature/top-p machinery.
- **Untrusted bytes**: bundle/archive parsers use checked arithmetic,
  explicit size caps, bounded allocation, and digest verification before
  engine initialization.

## Process conventions

- Conventional commits with scopes (`feat(bundle): …`), breaking `!` when
  applicable.
- Commit files **by name**, never `git add -A` (in-flight agent work must
  not be swept into unrelated commits).
- `models/` (local HF cache) is gitignored — never commit or delete it.
- Claim language: do not claim image/audio compilation or browser/WASM
  inference works — schemas exist, upstream compilation does not (see
  docs/rewrite-plan.md §9). Keep that section current.
