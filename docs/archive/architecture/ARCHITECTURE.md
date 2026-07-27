# Architecture — hologram-ai

This document defines **how** the source specification
([`../conceptual-model/00-source.md`](../conceptual-model/00-source.md)) is realized:
the development flow, the workspace organization, the parametric framework, and the
substrate contract.

## 1. The docs-as-code flow

Development is strictly directional. Nothing is implemented before it is *defined*, and
nothing is *defined* without a place in the conceptual model.

```
        docs/conceptual-model/00-source.md          (1) the conceptual authority (prose)
                     │  transcribed, row by row, into
                     ▼
        model/{dictionary,status,oracles,usecases}.toml   (2) the model as typed data
                     │  parsed + invariant-checked by
                     ▼
        crates/hologram-ai-model                    (3) typed registries (single source of truth)
                     │  each dictionary row names a
                     ▼
        features/suites/<stage>/<row>.feature       (4) the BDD definition (Gherkin)
                     │  whose steps are bound in
                     ▼
        hologram-ai-conformance (cucumber, Rust)    (5) the test
        apps/web/bdd (cucumber-js + Playwright)         (browser rows run in real Chromium)
                     │  which exercises the
                     ▼
        the parametric implementation               (6) crates/* + apps/web
                     │  validated against
                     ▼
        oracles/ + live authorities                 (7) the external authoritative oracle
```

A feature is **done** only when (4)→(7) are present and green in CI. The honesty gate
mechanically forbids skipping any link — a dictionary row with no feature, an orphan
feature, a gating scenario with a pending step, or a witness asserting a claim its
status does not permit.

## 2. System context

hologram-ai is the **AI front-end** for the hologram runtime. hologram is a declarative,
UOR-native compute substrate: it has zero knowledge of AI model formats and operates
only over canonical forms — the closed `hologram_ops::OpKind` catalog,
`ConstrainedTypeShape` dtypes, content-addressed (κ-label) buffers. That canonicality is
the basis of its performance guarantees (content-addressed compute elision,
zero-movement buffers, compile-time weight layout, structural fusion).

hologram-ai owns everything *above* hologram's graph layer and nothing below it:

| hologram-ai owns | hologram owns |
|---|---|
| model acquisition (HuggingFace), file parsing (safetensors, ONNX) | tensor arithmetic + kernels |
| the AI IR (`AiGraph` / `AiOp`) and optimization passes | graph representation, scheduling, compilation |
| lowering `AiGraph` → canonical hologram `Graph` | execution (`InferenceSession`) + buffer pool |
| tokenization, sampling, the generation loop | archive format + content addressing |
| architecture-family knowledge (parametric decoder registry) | fusion, elision, warm-start, weight layout |
| quantization decisions, κ-map emission + materialization | dtype/shape/op canonical semantics |

**Principle:** hologram-ai *declares*, it does not *dispatch*. Every `AiOp` has a
complete canonical realization (ops the closed catalog lacks are desugared into
canonical pipelines); op parameters are supplied by operand shapes, sparse attribute
tables, or extra operands — never carried on ops. There is no KV-cache, no runtime shape
machinery, and no failure/fallback path: autoregressive reuse is content-addressed
elision (see [`01-k-representation.md`](../conceptual-model/01-k-representation.md)).

## 3. Workspace organization

Crate boundaries follow axes of change:

| Crate | Responsibility | Changes when… |
|---|---|---|
| `hologram-ai-model` | Parses `model/*.toml` into typed, invariant-checked registries (dictionary, status ledger, oracle registry, use-cases). No pipeline code. | the conceptual model changes |
| `hologram-ai-common` | The AI IR (`AiGraph`), optimization passes, lowering to the canonical `Graph`, κ-map emission. Runtime-core posture. | the compilation pipeline changes |
| `hologram-ai-safetensors` | safetensors import: header streaming, and the **parametric decoder builder** (graph from `config.json` + tensor manifest, via the architecture-family registry). | a model family or the format handling changes |
| `hologram-ai-onnx` | ONNX protobuf → `AiGraph` (host shell; the byte-parsing perimeter). | the ONNX import changes |
| `hologram-ai-quant` | Q4_0/Q8_0 dequantization (GGML-conformant), `no_std`. | quantization changes |
| `hologram-ai-tokenizer` | Tokenizer encode/decode (`no_std` core) + train/load (host shell). | tokenization changes |
| `hologram-ai` | The facade + CLI: compile → materialize → run → generate; the κ-store seams. | the user-facing pipeline changes |
| `hologram-ai-wasm` | The browser binding: streamed compile, κ-hashing, materialization-at-load, generation with token streaming. | the browser API changes |
| `hologram-ai-core` | App-domain foundation: content-addressed manifests, event-sourced projection (`reduce`), runner abstraction. | the application domain changes |
| `hologram-ai-conformance` | The cucumber BDD runner (Rust rows), the honesty meta-gate, structural witnesses (ZA/ZM/CE/CF/LW/IM), oracle comparators. | the V&V wiring changes |
| `xtask` | Automation: oracle verification, pin checks, the conformance ledger. | tooling changes |
| `apps/web` | The application: pages, workers (download/generate), OPFS κ-store, and the browser BDD suite (cucumber-js driving Chromium). | the application changes |

The substrate (hologram member crates + holospaces) is imported at a pinned git
revision; `Cargo.lock` is the pin, checked by `xtask pin-check` against the live
upstream.

## 4. The k-form pipeline

### 4.1 Streamed acquisition (S1)

The download worker streams each safetensors shard: 8-byte header length → JSON header →
tensor byte ranges. Each tensor is incrementally hashed (κ = `blake3:<hex>`) and
persisted to OPFS as `tensors/{κ}.bin`. Peak memory is one tensor, not one shard.
Companions (`config.json`, `tokenizer.json`, `generation_config.json`) are persisted per
model. Storage is deduplicated by construction: κ-equality is content-equality.

### 4.2 Parametric compilation (S2)

`build_parametric_graph` consumes only `config.json` + the tensor manifest. Every
quantity is a function of the configuration — hidden size, layers, heads, KV heads, head
dim, vocabulary, `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, context length,
tensor dtypes. The family registry maps `config.architectures` to a builder; unsupported
families fail loud. Weights enter the graph as `AiParam::External { κ }`; lowering emits
0-byte constants plus the `holospaces.kappa_map` extension binding each constant slot to
its κ. The compiled `.holo` is a pure k-form: structure, schedule, ports, extensions —
no parameters.

### 4.3 κ-materialization (S3)

Materialization turns a k-form archive plus a κ-store into an executable archive:

1. read `holospaces.kappa_map` (constant slot → κ);
2. resolve each κ against the store (OPFS in the browser, a κ-directory natively);
3. **verify** each buffer re-hashes to its κ (content addressing is the integrity
   check) and matches the constant's declared dtype × shape byte length;
4. re-encode the constants section via `hologram-archive`'s public codec, dropping the
   compile-time warm-fold section (its folded results were derived over empty
   constants; the session re-derives the cone lattice from real content at load);
5. load the result into an `InferenceSession`.

A missing or corrupt κ aborts with the label. The κ-store trait has two realizations:
the browser resolver (the generate worker reads OPFS synchronously) and the native
directory store (conformance tests and the CLI).

### 4.4 Generation and chat (S4)

Generation is re-execution: each decode step runs the compiled graph with the grown
token sequence; the unchanged prefix cone re-derives resident κ-labels and is elided
(measured by the `decode-elision` witness via the session's dispatch/skip counters).
Sampling (temperature, top-k, seed) and templates come from the model's own
`generation_config.json` / chat template. The chat application drives the three-message
handshake defined in [`02-user-journey.md`](../conceptual-model/02-user-journey.md).

## 5. Verification & validation

The V&V axes, each a `just` target and a CI gate (see the [Justfile](../../Justfile)):

- **bdd** — the Gherkin suites: Rust rows via `cucumber` (fails on any skipped/undefined
  step), browser rows via cucumber-js in headless Chromium against the hermetic fixture.
- **honesty** — the meta-gate: model ⇄ features ⇄ witnesses bidirectional coverage,
  status discipline, tier discipline.
- **oracles** — every committed oracle artifact matches its recorded sha256;
  `pin-check` confirms pinned upstreams are live.
- **structural** — the substrate-contract witnesses: zero-alloc (ZA), zero-movement
  (ZM), content-addressed elision (CE), canonical-forms (CF), lowering-vs-reference
  (LW), import perimeter (IM).
- **conformance** — external-authority parity: ONNX Runtime execution diff, the ONNX
  node-test corpus, HF tokenizers, quant goldens, HF Hub resolution.
- **portability** — the runtime core builds `no_std` on `wasm32-unknown-unknown` and
  `thumbv7em-none-eabi`; the wasm binding builds with `wasm-pack`.
- **journey** — the browser journey (download → compile → run → chat handshake) in real
  Chromium; hermetic on every push, live models on the scheduled matrix. The Pages
  deployment **requires** this gate.

## 6. Parametricity

The canonical use-case instance is a real published model at a pinned revision; a
second, arbitrary tiny configuration (`model/usecases.toml`) exercises the same code
paths end-to-end (build → compile → materialize → generate) to prove no canonical
constant leaks into generic code. An anti-hardcode gate greps the builder for literal
canonical dimensions. Model identity appears only in data (catalogue, use-cases,
oracles) — never in code.
