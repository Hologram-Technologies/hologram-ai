# Holo AI Apps

## Purpose

`hologram-ai` currently contains the compiler/import/runtime-adjacent surfaces
 for AI models. `holospaces` is the lower-level boot/runtime layer that can
 provision `.holo` artifacts, Wasm userland apps, and devcontainer-backed
 holospaces.

The new `hologram-ai-core` crate is the first practical app-domain foundation
 above those layers. It defines deterministic AI app state and inference event
 types without assuming:

- a server
- a database
- mutable global state
- a central coordinator

## Relation To holospaces

`holospaces` already provides the useful low-level primitives:

- `.holo` model execution via `HoloEngine::run(archive, inputs)`
- Wasm userland lifecycle via the `hg_*` container ABI
- holospace provisioning via `boot::provision` and manager surfaces
- substrate κ-label types and content-addressed sources

`hologram-ai-core` sits above that boundary:

- model/app manifests are content-addressed domain objects
- prompts are events
- inference results are events
- reducers fold events into view state
- workers/runners execute inference outside the reducer

## Why Inference Is Outside Reducers

Reducers must stay deterministic, replayable, and side-effect free. Running a
 model inside a reducer would couple state projection to compute, scheduling,
 runtime capabilities, and worker availability.

Instead, the intended flow is:

1. `PromptSubmitted` enters the event stream.
2. `reduce(events)` projects a pending job.
3. A worker observes the pending job.
4. The worker runs inference through a real execution path.
5. The worker emits `InferenceCompleted` or `InferenceFailed`.
6. The reducer folds the new event into `AiView`.

This preserves replayability and cleanly separates deterministic state from
side effects.

## Mapping To `.holo` Execution

The expected `.holo` path is:

1. `hologram-ai` compiles/imports a model and emits a `.holo` archive.
2. A model manifest points at the compiled archive by κ-label.
3. A runner implementation resolves that archive and materialized inputs.
4. The runner invokes the equivalent of `holospaces::engine::HoloEngine::run`.
5. The output bytes/κ become `InferenceCompleted` provenance.

The current implementation does not directly depend on `holospaces`. Instead it
 uses `KappaRef` as a narrow adapter boundary so integration can reuse the
 substrate κ type later without redesigning the app model.

## Mapping To Wasm Userland Apps

When the app itself runs as Wasm userland in `holospaces`, it should map to the
 existing container ABI:

- `hg_init`
- `hg_event`
- `hg_suspend`
- `hg_resume`
- `hg_callback`

That userland app should remain a deterministic event processor and projector.
 It may request or observe inference work, but the actual model execution should
 still happen through a capability-scoped runner path rather than inside the
 reducer.

## Implemented Now

- `hologram-ai-core` crate
- stable domain types for app/model/runner manifests, requests, outputs, and views
- `AiEvent` variants for registration, submission, start, completion, and failure
- deterministic `reduce(events: &[AiEvent]) -> AiView`
- runner abstraction via `ModelRunner`
- provenance structure linking prompt/model/runner/worker/output κ references
- unit tests covering empty reductions, model registration, pending/completed/failed transitions, and ordering semantics

## Intentionally Left As Integration Work

- direct dependency on `holospaces::realizations::Kappa`
- worker discovery / scheduling
- `.holo` input materialization and output decoding policy
- Wasm userland app SDK
- UI projections
- streaming output
- network/provider integrations
- retrieval / RAG
- confidentiality / membership policy
