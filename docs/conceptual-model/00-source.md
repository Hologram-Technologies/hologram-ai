# hologram-ai — Implementation Definition

> **This file is the conceptual authority.** It defines the system this repository
> realizes. Every other artifact — the typed model in [`model/`](../../model/), the BDD
> features in [`features/`](../../features/), the V&V witnesses — derives from and must
> stay consistent with this document.

Status: living document; surgical edits.

## What this defines

**hologram-ai is an in-browser AI application**: it downloads models from HuggingFace,
compiles them into content-addressed `.holo` archives, and runs them — entirely
client-side, on the [hologram] compute substrate with the k-representation discipline
inherited from [holospaces]. The same pipeline is available natively (CLI + library),
which is where its external-authority conformance is anchored.

The system is **parametric**: every model-specific quantity (hidden size, layer count,
head counts, head dimension, vocabulary, RoPE base, normalization epsilon, weight tying,
context length, dtype) is a function of the model's own published configuration
(`config.json`) and tensor manifest. No architecture constant, no model identity, and no
capacity limit is hard-coded. Supported behavior is declared per *architecture family*
in a registry, never per model.

The conceptual model splits into what external sources establish (the safetensors
format, the HuggingFace Hub API, BLAKE3 addressing, ONNX operator semantics, reference
tokenization, GGML quantization) and what is a defined build on top (the parametric
decoder graph, streamed weightless compilation, κ-materialization, the chat
application). Both are tracked explicitly in the dictionary; neither is asserted beyond
what its oracle shows.

## The k-representation principle

hologram's performance guarantees are content-addressed: a buffer's identity is its
κ-label (`blake3` of its bytes), a node's output label derives from its op signature and
operand labels, and any computation whose label is already resident is **elided**
([`01-k-representation.md`](01-k-representation.md)). hologram-ai keeps model content in
k-representation at every stage:

- **Weights** are never a monolithic file. Each tensor is streamed, hashed, and persisted
  under its own κ (`tensors/{κ}.bin` in the browser's OPFS; a κ-store directory
  natively). Identical tensors — across shards, revisions, or *models* — deduplicate
  structurally.
- **The compiled archive is a pure k-form.** Streamed compilation consumes only the
  tensor *manifest* (names, shapes, dtypes, κ-labels) — never weight bytes. The `.holo`
  it emits carries the graph, schedule, and a `holospaces.kappa_map` binding each weight
  constant to its κ. The archive is small, model-structure-addressed, and independent of
  weight storage.
- **Materialization is resolution, not loading.** At session-load time the κ-map is
  resolved against the κ-store; every resolved buffer is re-hashed and must reproduce its
  κ — content addressing *is* the integrity check. A missing or corrupt κ fails loud with
  the label.
- **Autoregressive reuse is elision, not caching machinery.** There is no KV-cache and no
  runtime shape machinery. A decode step re-executes the compiled graph; the unchanged
  prefix cone re-derives the same κ-labels and is skipped by the substrate's residency
  check. The scaling story for browser workloads is this structural degeneracy — the same
  cache-collapse the UOR-Atlas-UTQC proof measures — not classical buffer management.

## The user journey (normative)

The application's contract is one journey, verified end-to-end in a real browser
([`02-user-journey.md`](02-user-journey.md)):

1. **Download** — the user names any HuggingFace repository; the app resolves its file
   manifest, streams safetensors shards through a persistent worker, persists each tensor
   under its κ, and stores the companion assets (`config.json`, `tokenizer.json`,
   `generation_config.json`).
2. **Compile** — the app builds the parametric graph from `config.json` + the tensor
   manifest and compiles a weightless κ-form `.holo`.
3. **Run** — the app materializes the archive against the OPFS κ-store and executes it in
   an inference session.
4. **Chat** — a three-message handshake completes against the running model: user →
   assistant → user → assistant → user → assistant, with streamed tokens, the model's own
   chat template, and its declared stop conditions.

A journey step that cannot proceed (unsupported architecture, resource budget exceeded,
missing κ) fails loud, with the reason, before any partial state is published.

## Sources

Every established claim is bound to an authority this repository did not author
([`model/oracles.toml`](../../model/oracles.toml)):

- **BLAKE3** — the official test vectors (KATs) for κ-addressing.
- **safetensors** — the reference `safetensors` crate as format authority.
- **HuggingFace Hub** — the live Hub API at pinned revisions for resolution/companions.
- **ONNX + ONNX Runtime** — operator semantics (official node-test corpus) and execution
  parity (ORT) for compilation and end-to-end logits.
- **HuggingFace `tokenizers`** — reference tokenization on the model's own
  `tokenizer.json`.
- **GGML/ONNX quantization references** — committed golden vectors for Q4_0/Q8_0.
- **hologram / holospaces** — the substrate's structural witnesses (zero-alloc,
  zero-movement, content-addressed elision, canonical forms), exercised at the pinned
  substrate revision.

Claims without a source are design decisions, marked as such in the dictionary and
validated as builds (against structural invariants), never asserted as sourced facts.

## Status vocabulary

`verified` = a sourced fact reproduced (an oracle artifact, a live authority, or a
substrate witness). `build` = a precisely-scoped construction on verified pieces,
validated against structural invariants and deterministic references. `open` = a genuine
unknown, measured and reported, never asserted true. See
[`03-status-discipline.md`](03-status-discipline.md).

[hologram]: https://github.com/Hologram-Technologies/hologram
[holospaces]: https://github.com/Hologram-Technologies/holospaces
