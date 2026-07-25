# Testing

## Gates (must be green before every commit)

```bash
# pinned toolchain — see AGENTS.md for the Homebrew-shadowing note
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo check -p hologram-ai-core   --no-default-features --target wasm32-unknown-unknown
cargo check -p hologram-ai-bundle --no-default-features --target wasm32-unknown-unknown
```

## Test layers

- **Schemas** (`hologram-ai-core`): canonical round-trips, determinism,
  truncation/trailing/hostile-length rejection, stable error codes,
  typed-abstention shape.
- **Bundle** (`hologram-ai-bundle`): round-trip, double-build byte
  identity, digest tampering, truncation fuzz, duplicate roles, unknown
  mandatory roles, capability⇔processor matrix (text/tokenizer,
  image/processor, audio/processor, multimodal, token-only exemption).
- **Acquisition** (`hologram-ai-huggingface`): hermetic via a fake `hf`
  executable — argv purity, cache-key determinism, staging/atomicity,
  locking, offline hit/miss, token redaction, unsupported-model,
  cancellation. Live HF tests are opt-in (`--ignored`).
- **R4 adapter** (`hologram-ai-r4`): artifact validation (garbage → typed
  errors), engine init from bundle bytes, status/abstention mapping,
  allocation census (zero-alloc steady state), determinism of repeated
  execution.
- **Facade/packaging** (`hologram-ai`): `.holo` round-trip through the
  official `HoloLoader`, v4 `InferenceModel` kind, entry uniqueness,
  0/1/N model layers, model-only archive (`primary = None`), explicit
  multi-model selection, fat-archive operation without an external store,
  no `tless_store.bin` in any produced bundle.
- **End-to-end** (hermetic fixture at `oracles/fixture/`): local source →
  uor-r4 compile adapter → bundle → `.holo` → load → select →
  predict/generate, proving the engine is R4G1 (engine id + artifact
  inspection). Long-running compile tests are `--ignored` in default CI.
- **Architecture guards**: dependency audit (no ort/onnxruntime/candle/
  tch/GPU crates in `cargo tree`), no public binary target, no `.r4g1`
  public output, allocation census, machine-code-free wrappers.

## Multimodal honesty

Schemas, packaging, discovery, value routing (inline vs κ-ref), and
processor-mismatch rejection are tested with deterministic fixtures.
No test claims real image/audio compilation — upstream uor-r4 does not
support it yet (docs/rewrite-plan.md §9).
