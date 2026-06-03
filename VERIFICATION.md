# hologram-ai — Verification & Validation

> **Scope.** What hologram-ai verifies and how. Every part of hologram-ai is
> validated against an **external authority** (a published spec or an
> independent reference implementation) — never against hologram-ai itself —
> and verified to **preserve hologram's structural guarantees** end-to-end.
> This mirrors hologram's own V&V discipline (see
> [`../hologram/VERIFICATION.md`](../hologram/VERIFICATION.md)).
>
> The normative invariant catalog is [`CONFORMANCE.md`](CONFORMANCE.md).
> The full suite is reproducible via `just vv`.

## Principle: external ground truth, not self-reference

A test that checks hologram-ai's lowering against hologram-ai's own reference,
or a model address against a constant hologram-ai produced, proves only
internal consistency. V&V here means conformance to an authority we did
**not** author:

| part | external authority |
|---|---|
| ONNX import | the **ONNX format spec** + **ONNX Runtime** graph (independent loader) |
| GGUF import | the **GGUF format spec** + cross-tool (independent loaders, via uor-addr realizations) |
| numeric lowering (matmul, norm, softmax, gelu, attention, …) | the **ONNX operator spec** + an independent **f64 reference**; **IEEE-754** |
| quantization (Q4_0 / Q8_0 dequant) | the **GGML / ONNX** quantization reference |
| full-graph inference | an independent runtime's outputs (**ONNX Runtime / PyTorch**) |
| tokenization | a reference tokenizer (**HuggingFace `tokenizers`** / sentencepiece) |
| model addressing (κ-labels) | **uor-addr** (itself externally validated) + TC-05 replay |
| structural guarantees | hologram's own contract (no_std, zero-alloc, zero-copy, canonical-forms-only) |

## Principle: faithful UOR-native client

hologram is a *declarative* runtime: its operations act **only over canonical
forms** (the closed `OpKind` catalog, `ConstrainedTypeShape` types,
κ-label-addressed buffers), and that is the basis of its performance
guarantees — content-addressed compute elision, zero-movement buffers,
compile-time weight layout. hologram-ai must not undermine this. It therefore:

- **declares, does not dispatch** — it maps each AI op to a canonical
  `hologram_ops::OpKind` whose meaning is a `Term` tree over the 10 upstream
  primitives, rather than emitting an imperative op enum (the deleted
  `FloatOp`);
- **hands hologram only canonical forms** (CF) — never a parallel tensor/op
  representation that would defeat addressing/elision;
- **relies on content-addressed elision, not a KV-cache** (CE) — autoregressive
  reuse is structural (κ-label recognition of the unchanged prefix), so the
  legacy KV-cache optimization is abandoned;
- **preserves no_std / zero-alloc / zero-copy** in the runtime core (NS/ZA/ZM),
  so a compiled model runs unchanged on wasm, embedded, and platform targets.

## V&V axes (reproducible: `just vv`)

1. **Architecture** — `cargo fmt --check`, `cargo clippy --workspace -D warnings`, `cargo build` + `cargo nextest run` against hologram 0.5.0 (class AR).
2. **Import & correctness conformance** — ONNX/GGUF import vs spec + ORT; lowering vs independent f64 reference; quant vs GGML; tokenizer vs reference (classes IM, LW, CF, QZ, TK).
3. **End-to-end** — full-model logits vs ONNX Runtime / PyTorch (class EE).
4. **Addressing & replay** — model → uor-addr κ-label + TC-05 witness round-trip (class MA).
5. **Structural** — content-addressed elision replaces KV-cache (CE); zero-movement / zero-copy (ZM); zero runtime heap allocation via a counting allocator (ZA).
6. **Portability** — `just vv-wasm` + `just vv-embedded`: the runtime-core stack builds on `wasm32-unknown-unknown` and `thumbv7em-none-eabi`, `no_std` (class NS).
7. **Performance / no-bottleneck** — release benches with per-stage budgets; a stage regressing past budget fails V&V (class PV).

## Runtime-core vs. host-shell

The structural axes (CF, ZA, ZM, NS, CE) bind the **runtime core** —
`hologram-ai-common`, `hologram-ai-quant`, and the tokenizer encode/decode path
— the crates that must run on-device, in-browser, and on bare metal. Model
ingest (ONNX protobuf, GGUF), download, tokenizer training, and the CLI are
**host shells** (std, allocating), exactly as hologram's CLI is a std host over
its `no_std` lib stack. See `CONFORMANCE.md` § "Runtime-core vs. host-shell".

## Status (living document — gaps are stated, not hidden)

Status marks describe the current build, not a plan. AR-3 (the workspace builds
against hologram 0.5.0) is the root invariant; until it holds, the other
classes' witnesses cannot run, so they read 🚧/⛔. The marks move to ✅ as the
implementation conforms to [`architecture.md`](specs/docs/architecture.md) — the
framework measures conformance to the structural and performance contract; it
does not prescribe an order in which to achieve it.
