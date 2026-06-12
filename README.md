# `hologram-ai`

`hologram-ai` is the AI model compiler and runtime-adjacent library for the
Hologram ecosystem. It imports models, validates and lowers them, and emits
`.holo` archives through `hologram-compiler`.

The repository now has two practical, working layers:

- model compilation and execution packaging for `.holo`
- app-domain foundations for deterministic AI applications above that layer

## What Works Now

### 1. Compile ONNX models to `.holo`

The primary path is ONNX-first. `hologram-ai` compiles ONNX graphs into `.holo`
archives that can be executed by Hologram runtime surfaces.

Example:

```bash
cargo run -p hologram-ai -- compile \
  --model models/bert-base-uncased/model.onnx \
  --output /tmp/out \
  --seq-len 8
```

### 2. Export deterministic fixtures for `holospaces`

`hologram-ai` can compile a model and export a deterministic fixture bundle for
`holospaces`. The bundle contains:

- the compiled `.holo` archive
- typed input blobs
- expected output blobs
- content-addressed κ labels
- a `manifest.json` describing archive, tensors, port order, and file layout

Example:

```bash
cargo run -p hologram-ai -- export-fixture \
  --model models/bert-base-uncased/model.onnx \
  --output /tmp/bert-holospaces \
  --preset bert-base-uncased \
  --seq-len 8
```

For `bert-base-uncased`, the current preset synthesizes the canonical sequence:

- `input_ids = [101, 2023, 2003, 1037, 3231, 102, 0, 0]`
- `attention_mask = [1, 1, 1, 1, 1, 1, 0, 0]`
- `token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0]`

### 3. Run exported `.holo` artifacts in `holospaces`

The intended bridge is:

1. compile or export from `hologram-ai`
2. hand the emitted `.holo` plus deterministic inputs to `holospaces`
3. execute with `holospaces::engine::HoloEngine::run`
4. verify the resulting output κ labels match the exported fixture manifest

The practical witness currently lives in the sibling `holospaces` repository.
The end-to-end BERT proof flow is:

```bash
cd /Users/auser/work/uor/hologram/hologram-ai
cargo run -p hologram-ai -- export-fixture \
  --model models/bert-base-uncased/model.onnx \
  --output /tmp/bert-holospaces \
  --preset bert-base-uncased \
  --seq-len 8
```

Then in the `holospaces` checkout:

```bash
HOLOSPACES_HOLO_FIXTURE_DIR=/tmp/bert-holospaces \
cargo test -p holospaces --test cc2_holo_engine \
  exported_hologram_ai_fixture_runs_to_expected_kappas -- --nocapture
```

That witness proves a real `hologram-ai`-compiled BERT `.holo` can execute
through `holospaces` and reproduce the expected output κ labels.

## New App-Domain Foundation: `hologram-ai-core`

This repository now also includes `crates/hologram-ai-core`, a deterministic
application-layer foundation above the compiler/runtime boundary.

It provides:

- content-addressed model, runner, prompt, request, output, and provenance types
- `AiEvent` variants for registration, submission, start, completion, and failure
- a pure `reduce(events: &[AiEvent]) -> AiView` projection
- a `ModelRunner` trait so inference execution stays outside reducers

This layer exists so higher-level AI applications can remain:

- deterministic
- replayable
- event-sourced
- independent from any single server or mutable coordinator

The intended execution model is:

1. a prompt is submitted as an event
2. the reducer projects a pending job
3. a worker observes that job
4. the worker runs inference through a real runner path
5. the worker emits completion or failure events

What is intentionally not wired yet:

- direct `holospaces` dependency from `hologram-ai-core`
- worker discovery and scheduling
- `.holo` input materialization and output decoding policy
- streaming output
- Wasm userland app SDK

## CLI Surface

`hologram-ai` currently exposes four primary commands:

- `compile`: compile a model into a `.holo` archive
- `export-fixture`: compile and export a deterministic `holospaces` bundle
- `run`: execute a compiled `.holo` archive
- `download`: fetch a model repository into `models/`

Example download:

```bash
cargo run -p hologram-ai -- download bert-base-uncased
```

Use `download` instead of external model download tools so the repository keeps
the expected layout (`model.onnx`, `tokenizer.json`, `config.json`, and related
assets under `models/`).

## Repository Layout

- `crates/hologram-ai`: top-level CLI and library surface
- `crates/hologram-ai-core`: deterministic AI app-domain foundation
- `crates/hologram-ai-common`: shared IR, shapes, and compilation data model
- `crates/hologram-ai-onnx`: ONNX import and lowering pipeline
- `crates/hologram-ai-conformance`: end-to-end conformance harness
- `models/`: checked-in development models and downloaded model repos
- `specs/docs/`: architecture and pipeline documentation

## Validation

The current `hologram-ai-core` foundation was validated with:

```bash
cargo fmt --check
cargo test -p hologram-ai-core
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

For more detail, see:

- [CLI docs](specs/docs/cli.md)
- [Architecture docs](specs/docs/architecture.md)
- [Repository layout docs](specs/docs/repository-layout.md)
- [Holo AI apps notes](specs/docs/holo-ai-apps.md)
