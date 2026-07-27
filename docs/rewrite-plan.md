# hologram-ai rewrite plan — uor-r4 / R4G1 / `.holo` InferenceModel

Status: normative for the `rewrite/uor-r4-inference-model` branch.
Supersedes every pre-rewrite architecture document (moved to `docs/archive/`).

## Pinned revisions (exact)

| Repository | Role | Commit |
|---|---|---|
| `UOR-Foundation/uor-r4` | compiler + R4G1 runtime | `f1b4859e65363eda9aa7dbeb0db467d93c8f4b02` (main) |
| `Hologram-Technologies/hologram` | `.holo` container, layers, CLI, FFI/SDK | `94ecb886811115491a77c8229e494965bea03fc2` (main, v0.12.1+46) |
| `Hologram-Technologies/hologram-ai` | rewrite base | `8e5a13c10490d4dc874f99d3c64d78f68940df7f` (main) |

Upstream patches are developed in sibling worktrees (PR-ready branches, not yet
merged upstream):

- `../uor-r4-facade` — branch `feature/typed-integration-facade`, tip
  `384a0e9` (off `f1b4859`)
- `../hologram-im` — branch `feature/inference-model-layer`, tip `fdd1190`
  (off `94ecb88`)

Until those PRs merge, this workspace depends on them by **path** (see root
`Cargo.toml`); the `Cargo.toml` comments name the exact git revs to substitute.
Patch-ready diffs and contracts are mirrored under `docs/upstream/`.

## Baseline (recorded before deletion)

- `cargo check --workspace --all-targets` on `main` @ 8e5a13c: **clean**.
- `cargo test --workspace` on `main` @ 8e5a13c: **green** (all suites,
  including doc-tests; run 2026-07-24 before any deletion).

## 1. Current architecture inventory (pre-rewrite)

~70.3K LOC Rust + ~9.7K LOC TS/JS, all built on the hologram **tensor**
substrate (`hologram-types/ops/graph/compiler/exec/archive/backend/host` pinned
at hologram v0.10.0 / rev `47a4955`):

- `crates/hologram-ai` (21.3K) — facade + CLI binary; HF downloader; tensor
  compiler driver; `runner.rs`/`engine.rs`/`decode.rs`/`speculative.rs`/
  `staged.rs` tensor execution, generation, speculative decoding.
- `crates/hologram-ai-common` (20.8K) — canonical `AiGraph` IR, opt passes,
  memory planner, lowering to `hologram-graph`.
- `crates/hologram-ai-conformance` (13.9K) — ORT cross-validation, 63 Gherkin
  BDD features, reference kernels.
- `crates/hologram-ai-onnx` (3.4K) — ONNX importer.
- `crates/hologram-ai-safetensors` (3.6K) — safetensors → AiGraph parametric
  decoder-graph compiler.
- `crates/hologram-ai-tokenizer` (3.3K) — native BPE/Unigram/WordPiece.
- `crates/hologram-ai-wasm` (2.6K) — wasm-bindgen browser shell.
- `crates/hologram-ai-quant` (0.8K) — Q4_0/Q8_0 block quant.
- `crates/hologram-ai-core` (0.8K), `crates/hologram-ai-model` (0.5K) —
  app-domain glue, TOML conceptual-model registries.
- `apps/web` (React browser app), `apps/desktop` (abandoned Tauri skeleton),
  `apps/extension` (3 JS files).
- `docs/adrs/` (14 ADRs incl. ADR-0016 "compiler, not a runtime"),
  `docs/conceptual-model/`, `docs/notes/` (substrate bug notes).
- CI: `ci.yml` (gate/structural/ort/portability/pin-check/journey),
  `architectures.yml` (HF model-download lanes), `pages.yml` (Pages deploy).

## 2. Deletion / retention

### Deleted (superseded architecture)

- All eight old crates (`hologram-ai{,-common,-onnx,-safetensors,-quant,
  -conformance,-wasm,-core}`), `xtask` oracle/report commands.
- `apps/` (web, desktop, extension) — tied to the tensor/ONNX pipeline.
- `features/` (63 tensor-architecture BDD features), `model/`, `specs/`,
  `oracles/onnx`, `oracles/quant`, `output/`.
- All hologram **substrate** git deps (hologram v0.10.0 pin); replaced by
  `hologram-archive` + `hologram-space` only, at the new pinned rev.
- Old CI workflows; replaced by a rewrite-scoped `ci.yml`.
- Old ADRs/docs → `docs/archive/` (non-normative).

### Retained (retargeted)

- `rust-toolchain.toml`, `Justfile` gate layout, `.githooks/pre-push`,
  `.cargo/config.toml` (wasm simd flags), CI caching/concurrency patterns.
- `oracles/blake3/test_vectors.json` (architecture-agnostic).
- `oracles/fixture/` (tiny safetensors model + `tokenizer.json`) — reused as
  the hermetic local-source compile fixture.
- `scripts/download-models.sh` shape (HF fetch with pinned commits).
- License declared (MIT OR Apache-2.0); **add missing LICENSE files**,
  `AGENTS.md`, `CONTRIBUTING.md`.

## 3. Target crate / dependency graph

```text
crates/
  hologram-ai-core/        no_std + alloc. Schemas: bundle manifest, value/
                           operation/capability descriptors, requests,
                           completions, status/finish reasons, stable errors,
                           canonical encoding. No I/O, no deps on hologram/uor-r4.
  hologram-ai-bundle/      no_std + alloc. R4 inference bundle schema v1:
                           deterministic encoder, bounded zero-copy parser,
                           digest verification, role/capability validation.
                           Deps: hologram-ai-core, blake3.
  hologram-ai-r4/          std. PRIVATE uor-r4 adapter: typed compile
                           orchestration, engine load-from-bundle-bytes,
                           predict/generate _into entrypoints, status/abstain/
                           witness mapping. Only crate depending on uor-r4.
  hologram-ai-huggingface/ std. ModelSourceProvider: HF pinned acquisition,
                           content-addressed cache, locking, resume, offline,
                           progress/cancellation; LocalDirectoryProvider.
  hologram-ai/             std. Public facade: Compiler builder, Application,
                           Model registry, Session, invoke/generate/predict,
                           hologram packaging (InferenceModel layers in .holo).
                           Deps: core, bundle, r4, huggingface,
                           hologram-archive, hologram-space.
```

Dependency direction (acyclic):

```text
hologram-ai-core ← hologram-ai-bundle ← hologram-ai-r4 ← hologram-ai
                                  ←— hologram-ai-huggingface ←—┘
hologram-ai ─→ hologram-archive, hologram-space   (hologram-im branch)
hologram-ai-r4 ─→ uor-r4-api (facade branch) ─→ uor-r4-{core,graph-*}
```

Hologram's CLI/FFI may depend on `hologram-ai` (upstream patch); `hologram-ai`
never depends on `hologram-cli`/`hologram-ffi`. No public `hologram-ai` binary.

## 4. Minimal upstream APIs required

### uor-r4 (`feature/typed-integration-facade`, new crate `uor-r4-api`)

Existing typed pieces are reused; the patch adds only orchestration shells:

- `compile(R4CompileRequest, &mut dyn ProgressSink)
  -> Result<CompiledR4Model, CompileError>` — typed wrapper over the existing
  3-stage pipeline (teacher bundle → cover → scored graph). Takes a local
  source dir + private work dir; returns artifact bytes with semantic roles
  (`graph`, `signature-artifact`, `tokenizer`, `score-report`, provenance);
  never dictates public filenames. Structured `CompileError`; resumability is
  a typed `CompileOutcome`, not stdout text.
- `R4Engine::load(R4ModelParts<'_>) -> Result<R4Engine, LoadError>` — engine
  from byte slices (scored R4G1 graph + signature artifact + optional
  tokenizer), no filesystem, no sidecar JSON paths. Moves the root package's
  `r4g1.rs` glue (status policy, tokenizer, `GraphScorer`) into the library.
- `predict_next_into` / `generate_into` with caller-owned buffers; zero
  steady-state heap allocation; typed `ResolutionStatus`/abstention/witness.
- `Tokenizer::from_bytes` (compile-side export already exists).
- `abi_version()` / `format_version()` handshake.
- No `.holo`, no networking, no algorithm changes, no legacy-path removal.

### hologram (`feature/inference-model-layer`)

- `.holo` **format v4**: `FORMAT_VERSION = 4`, `MIN_READ_VERSION` unchanged
  (2); `LayerKind::InferenceModel = 4` appended (existing discriminants
  untouched). Non-exit-bearing. `aux` carries the **engine identifier**
  (`"uor-r4"`) — the existing kind-specific typed tag field (arch for rootfs,
  surface for view); no schema field is added and no unrelated field is
  overloaded. Entry names: non-empty, unique per app (manifest validation).
- Manifest validation: `primary` may be `None` for model-only archives;
  `primary` still must be exit-bearing when present.
- `hologram ai {download,compile,inspect,infer}` CLI group delegating to
  `hologram-ai` (optional dependency).
- FFI: `hologram_ai_*` exports + stable error-code range + `FEATURES` strings;
  SDK metadata regenerated; Python ctypes and TS `NativeBinding` wrappers.
- No `uor-r4` dependency in any hologram crate; no tensor-plan fallback.

## 5. Product flow (target)

```text
HF repo@<full-sha> | local source dir
  → hologram-ai-huggingface (verified immutable cache entry)
  → uor-r4-api::compile (private resumable work dir)
  → hologram-ai-r4 validates artifacts (GraphView two-stage + CIDs)
  → hologram-ai-bundle: deterministic R4 inference bundle v1 (role-addressed)
  → hologram-ai: InferenceModel layer { kind=InferenceModel, content=κ(bundle),
    entry="ai.default", aux="uor-r4" } → fat .holo v4 via HoloWriter
  → verify with HoloLoader → atomic rename to <output>.holo
```

The only public compile output is a `.holo` file. `.r4g1` bytes never leave
the private work dir as a user-facing artifact.

## 6. R4 inference bundle schema v1 (summary; normative: docs/bundle-format.md)

Length-prefixed canonical sections, each with a semantic **role** (u16),
flags, length, and BLAKE3 digest; roles are closed and versioned:

- mandatory: `manifest` (canonical model manifest), `graph` (scored R4G1),
  `signature-artifact` (the artifact historically named `tless_artifacts.bin`
  — the historical name is never public), `score-report`, `status-policy`,
  `abi-versions`, `provenance`.
- conditional: `tokenizer` (required iff any text input/output operation),
  `image-processor`, `audio-processor`, `generation-config`,
  `vocabulary-metadata`, `label-map`, `witnesses`.
- excluded: `tless_store.bin`, corpus/observation/cover data, checkpoints,
  source weights, absolute paths, wall-clock timestamps.

Parser: checked arithmetic, max component/total sizes, bounded allocation,
duplicate-role and unknown-mandatory-role rejection, digest verification
before engine init, capability⇔processor consistency.

## 7. Phased implementation steps

1. **Audit/reset** (this document; delete §2; baseline recorded).
2. **Portable core** — `hologram-ai-core` schemas + errors + canonical encoding.
3. **Bundle codec** — `hologram-ai-bundle` + fuzz/malicious-input tests.
4. **R4 adapter** — uor-r4 facade branch (`uor-r4-api`) + `hologram-ai-r4`.
5. **Hologram packaging** — hologram v4/InferenceModel branch + facade
   packaging/discovery/selection.
6. **HF source provider** — `hologram-ai-huggingface` + compile orchestration.
7. **Runtime facade + SDK surface** — registry/session/invoke/streaming;
   hologram CLI + FFI + Python/TS patches.
8. **Docs/validation** — README, docs/*, ADRs, gates.

## 8. Acceptance tests (per phase)

- Repo health: fmt, clippy `-D warnings`, `cargo test --workspace
  --all-features`, `wasm32-unknown-unknown` check of core+bundle, no stale
  workspace members.
- Architecture guards: no public binary; no `.r4g1` public output; no GPU/
  transformer/MatMul/ORT deps in tree (`cargo tree` + dependency audit); no
  `tless_store.bin` in any bundle (artifact inspection test).
- Determinism: same inputs → byte-identical bundle and `.holo` (double-build
  test); no timestamps/paths/random in canonical bytes (scan test).
- Bundle/archive: round-trips, v4 InferenceModel kind, entry uniqueness,
  0/1/N model layers, model-only archive `primary=None`, capability⇔processor
  matrix, tamper/truncation/hostile-length/duplicate-role/unknown-role,
  fat-archive operation store-free.
- R4 inference: hermetic fixture (`oracles/fixture` local source) → compile →
  bundle → `.holo` → load → select → predict/generate; typed abstention;
  output limit; cancellation; reset; status/witness propagation; repeated-run
  determinism. Proves R4G1 path (engine id + artifact inspection), not a
  fallback.
- Allocation: counting-allocator census — zero heap allocs in
  `predict_next_into` and steady-state `generate_into` steps after init.
- HF: repo/revision validation, arg construction (no shell), cache-key
  determinism, locking, staging/atomicity, offline hit/miss, token redaction,
  unsupported-model error, progress/cancellation. No network in default CI.
- Multimodal (schema-level, deterministic fixtures — no false claims):
  text+image model, audio transcription model, multi-model archive, κ content
  refs vs inline media routing, processor-mismatch rejection.
- SDK: Rust smoke; Python/TS via hologram FFI patch (documented, run where
  toolchains exist).

## 9. Honest limits (kept current)

- Real image/audio R4 compilation does **not** exist upstream; multimodal
  support is schema/packaging/routing only until uor-r4 grows those paths.
- Browser/WASM inference is deferred (see ADR-0008); only the portable core
  (schemas + bundle codec) is wasm-gated now.
- uor-r4's deployed scorer (`GraphScorer`) is std-only today; zero-alloc
  guarantees are asserted at the `R4Engine` step boundary, inherited from
  uor-r4's own contract checks.
- Generation delegates to upstream `R4Engine::generate_into`: stream events
  are emitted as a post-run batch, and session cancellation is honored at
  run entry, not between generation steps (upstream exports no mid-run
  hook; duplicating its policy loop would fork the algorithm — ADR-0001).
  Compile cancellation IS honored between stages.
- Hugging Face branch/tag resolution (`Revision::Resolve`) recovers the
  resolved commit from `hf` download metadata; when unavailable it fails
  with guidance to pin. Only full-SHA pinning carries the complete
  immutability guarantee.
- The `hologram ai` CLI group and FFI surface are written and verified
  against a documented stub on the hologram side; enabling them requires
  uncommenting one dependency line per crate (see
  docs/upstream/hologram-contract.md) until hologram-ai is pinned by rev.
