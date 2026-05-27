# hologram-ai — Conformance

> **Purpose.** The normative invariant catalog hologram-ai must uphold to
> claim it is a correct, bottleneck-free, **UOR-native** AI front-end for
> hologram. Each invariant has a class + number, a normative statement, an
> enforcement mechanism, and a traced artifact (test/bench/check). Mirrors
> the discipline of hologram's own [`CONFORMANCE.md`](../hologram/CONFORMANCE.md).
> Reproduced by `just vv`.
>
> **Status legend:** ✅ enforced & passing · 🟡 partial · ⛔ gap (tracked) · 🚧 blocked (does not yet build against hologram 0.5.0).

## Principle

hologram-ai owns everything *above* hologram's graph layer — model ingest,
the AI IR, lowering, quantization, tokenization, sampling, generation — and
nothing below it. It is validated against **external authorities we did not
author** (the ONNX/GGUF format specs, ONNX Runtime / PyTorch, the GGML/ONNX
quantization references, reference tokenizers), never against hologram-ai's
own output. See [`VERIFICATION.md`](VERIFICATION.md).

It must also be a **faithful UOR-native client of hologram**: it operates only
over hologram's **canonical forms** (the basis of hologram's performance
guarantees), declares rather than dispatches, and preserves hologram's
structural properties end-to-end — `no_std`, zero runtime heap allocation,
zero-copy / zero-movement / zero-cost, multi-target (wasm + embedded +
platform).

## Runtime-core vs. host-shell boundary

The structural invariants (CF, ZA, ZM, NS, CE) apply to the **runtime core** —
the crates that run on-device, in-browser, and on bare metal. Model ingest,
download, tokenizer training, and the CLI are **host shells** (std, allocating)
and are exempt from NS/ZA/ZM, exactly as hologram's CLI is a std host over its
`no_std` lib stack.

| Tier | Crates | Bound by |
|---|---|---|
| **Runtime core** (`no_std`, zero-alloc, zero-copy) | `hologram-ai-common` (IR + lowering + memory plan), `hologram-ai-quant` (dequant kernels), `hologram-ai-tokenizer` (encode/decode path) | CF, LW, QZ, ZA, ZM, NS, CE |
| **Host shells** (`std`) | `hologram-ai-onnx` (protobuf import), GGUF import, downloader, tokenizer train/JSON, `hologram-ai` CLI, desktop app, `hologram-ai-conformance` | IM, TK, EE, MA, AR, PV |

## Classes

| Class | Scope | Enforcement |
|---|---|---|
| **AR** | Architecture — fmt, clippy `-D warnings`, builds against hologram 0.5.0 | `just vv-arch` |
| **IM** | Import conformance vs ONNX/GGUF format spec + cross-tool | conformance tests vs independent loaders / ORT graph |
| **LW** | Lowering — every `AiOp` lowers to a **canonical** `hologram_ops::OpKind`, semantics-preserving, no silent fallback | lowering tests vs independent f64 reference |
| **CF** | Canonical-forms-only — hologram-ai never hands hologram a non-canonical representation | type/lowering unit tests + boundary assertions |
| **QZ** | Quantization vs GGML/ONNX dequant reference | conformance tests vs reference vectors |
| **TK** | Tokenization vs reference tokenizer (HF tokenizers / sentencepiece) | conformance tests vs reference encodings |
| **EE** | End-to-end logits vs ONNX Runtime / PyTorch within tolerance | `--features conformance` ORT tests |
| **MA** | Model addressing — model → uor-addr κ-label, order-independent composition, TC-05 replay | tests (inherited from uor-addr) |
| **CE** | Content-addressed elision replaces KV-cache — decode-step prefix reuse is recognized by κ-label and elided | exec reuse tests (dispatch counting) |
| **ZM** | Zero-movement / zero-copy / zero-cost — lowering + packaging introduce no per-node copies; weights borrowed/mmap'd | exec instrumentation + review gate |
| **ZA** | Zero runtime heap allocation — the inference hot path allocates nothing | counting-allocator harness |
| **NS** | `no_std` portability — runtime-core crates build on `wasm32-unknown-unknown` + `thumbv7em-none-eabi` | cross-target builds |
| **PV** | Performance — every stage within budget, no bottlenecks | benches with baselines/budgets |

---

## AR — Architecture

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **AR-1** | `cargo fmt --all --check` is clean. | `just fmt-check` | CI | ✅ |
| **AR-2** | `cargo clippy -- -D warnings` is clean across the six crates. | `just clippy` | CI | ✅ |
| **AR-3** | The six hologram-ai crates build against hologram `0.5.0` (the UOR-native runtime). | `just build` | CI | ✅ |
| **AR-4** | `cargo test` passes (smoke + conformance + quant golden). | `just test` | CI | ✅ |

> **AR-3 holds.** `hologram-ai-common`, `-quant`, `-onnx`, `-tokenizer`, the
> `hologram-ai` lib+bin, and `-conformance` all compile against hologram 0.5.0
> on the canonical `OpKind` model. (The Tauri desktop app needs GTK system libs
> — an OS dependency, out of scope for the core.) AR-1/2/4 (fmt/clippy/tests) and
> the correctness classes are next; gaps are stated here, not hidden.

## IM — Import conformance (external: ONNX/GGUF spec + cross-tool)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **IM-1** | A well-formed ONNX model parses to an `AiGraph` that compiles + runs; validated against the **official ONNX backend node-test corpus** (the operator spec's artifacts). | live spec corpus | `onnx_spec_conformance.rs` | ✅ |
| **IM-2** | A well-formed GGUF (v2/v3) model parses to an `AiGraph`; tensor metadata matches the GGUF spec. | test vs spec vectors | `hologram-ai-conformance` | 🚧 |
| **IM-3** | Byte-level model parsing is confined to `Grounding` impls at the input boundary (no mid-graph byte parsing). | review gate + grep check | `just vv-arch` | ⛔ |

## LW — Lowering (external: ONNX op spec + IEEE-754, independent f64 ref)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **LW-1** | Every `AiOp` lowers to a **canonical** `hologram_ops::OpKind` (or a desugared pipeline); the compiled+run output equals the **ONNX operator spec's** authoritative expected output. | live ONNX node-test corpus | `onnx_spec_conformance.rs` | 🟡 |
| **LW-2** | Every `AiOp` has a complete canonical realization — mapped to an `OpKind`, attrs/operands attached, or desugared into a canonical `OpKind` pipeline. There are **no** unsupported ops and no runtime failure path. Each desugaring equals an independent f64 reference of the op it replaces. | exhaustive `AiOp` lowering test vs f64 ref | `hologram-ai-common` | 🚧 |
| **LW-3** | Fused AI ops (attention, SwiGLU, RoPE) lower to hologram's canonical fused `OpKind`s (`Attention`, `FusedSwiGlu`, `RotaryEmbedding`), not hand-rolled custom handlers. | lowering test | `hologram-ai-common` | 🚧 |

## CF — Canonical-forms-only

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **CF-1** | All tensors/dtypes/shapes handed to hologram are declared as `ConstrainedTypeShape` over `hologram-types`; hologram-ai keeps no parallel tensor/op type system at the boundary. | type-boundary unit test | `hologram-ai-common` | ⛔ |
| **CF-2** | Every op handed to hologram is a member of the closed `OpKind` catalog; hologram-ai emits no non-canonical op encoding. | lowering exhaustiveness test | `hologram-ai-common` | ⛔ |
| **CF-3** | Operating only over canonical forms is what makes content-addressing/elision (CE) and zero-movement (ZM) hold — CF is the precondition, verified jointly with CE-1/ZM-1. | composite test | `hologram-ai-conformance` | ⛔ |

## QZ — Quantization (external: GGML / ONNX dequant reference)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **QZ-1** | Q4_0 dequant equals the GGML reference over the golden vectors. | test vs golden | `quant_golden.rs` | ✅ |
| **QZ-2** | Q8_0 dequant equals the GGML reference over the golden vectors. | test vs golden | `quant_golden.rs` | ✅ |
| **QZ-3** | A quantized weight lowers to canonical `Dequantize → MatMul` carrying its scale/zero-point as `QuantAttrs`; hologram fuses it to `MatMulDequant`, which reads the **packed** weight in-register (dense f32 never materialized). Per-tensor (scalar scale folded into the node) and per-channel (exact ONNX axis, scale f32 / zero-point widened to i32 vectors). Output matches the f64 reference. | lowering+exec test | `hologram-ai` (`tests/quantized_weight_memory.rs`) | ✅ |
| **QZ-4** | Quantized weights occupy their **packed** size at runtime — i8 ≈ ¼, i4 ≈ ⅛ of dense f32 (measured via `resident_bytes()`). i4 is genuinely sub-byte (two nibbles/byte). | exec test | `hologram-ai` (`tests/quantized_weight_memory.rs`) | ✅ |
| **QZ-5** | `DequantizeLinear` matches the **official ONNX backend node-test vectors** — `test_dequantizelinear` (per-tensor uint8) and `test_dequantizelinear_axis` (per-channel) — imported and verified against the spec's own `output_0.pb`. | `--features onnx-spec`, `HOLOGRAM_AI_LIVE=1` | `tests/onnx_spec_conformance.rs` | ✅ |
| **QZ-6** | **Arbitrary weight-quant configs lower without panic or fallback.** Asymmetric / unsigned / negative zero-points, per-channel along any axis, i4, and a **runtime (non-constant) scale** each compile + run correct (vs f64): a constant scale takes the packed `MatMulDequant` path, anything else the canonical primitive `(toᶠ³²(x)−toᶠ³²(zp))·scale` — never a `bail`/`panic` on a valid model. | test | `hologram-ai` (`tests/quant_arbitrary_models.rs`) | ✅ |

## TK — Tokenization (external: reference tokenizer)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **TK-1** | BPE encode of a corpus matches the HuggingFace `tokenizers` reference token-for-token. | test vs reference | `hologram-ai-tokenizer` | 🚧 |
| **TK-2** | Decode(encode(x)) == x for the round-trippable corpus. | round-trip test | `hologram-ai-tokenizer` | 🚧 |

## EE — End-to-end (external: ONNX Runtime / PyTorch)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **EE-1** | A full multi-layer model (`mini_transformer`: 18 nodes — MatMul, Softmax attention, Sigmoid-gated FFN, residual Adds, Transposes) compiled + run through hologram-ai matches **ONNX Runtime** on the same input within tolerance (observed max relative error 2.2e-5). | `--features conformance` | `tests/ort_full_model_e2e.rs` | ✅ |
| **EE-1b** | Operator-spec outputs match the official ONNX backend node-test corpus (relu/add/matmul/softmax/mul/sub). | `--features onnx-spec`, `HOLOGRAM_AI_LIVE=1` | `tests/onnx_spec_conformance.rs` | ✅ |
| **EE-2** | Large published models (TinyLlama, ResNet-50, MobileNetV2, MiniLM) match ORT within tolerance end-to-end. | `--features conformance` + model downloads | (infra-bound: needs model fetch) | 🚧 |

## MA — Model addressing (external: uor-addr, TC-05 replay)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **MA-1** | A model addresses to a verifiable uor-addr κ-label via `uor_addr::{onnx,gguf,json}::address`; the TC-05 witness re-certifies to the same label. The label is carried in `HoloArchive.metadata` as the model's dedup / warm-start identity. | test | `hologram-ai` (`address.rs`, `tests/ma_external_models.rs`) | ✅ |
| **MA-1b** | The minted κ-labels are **byte-identical** to uor-addr's authoritative pins (`tests/external_models.rs`) for published GGUF/ONNX models (Qwen2-0.5B, MobileNetV2-7, all-MiniLM-L6-v2). | live test (`HOLOGRAM_AI_LIVE=1` + network) | `hologram-ai` (`tests/ma_external_models.rs`) | ✅ |
| **MA-2** | Multi-component models compose via `compose_model` (order-independent E₈ product) — components addressed on the BLAKE3 axis, folded in canonical order so the identity is a pure function of the component *set*. | test | `hologram-ai` (`address.rs`: `component_kappa`/`compose_model`/`compose_models`; `tests/ma_external_models.rs`) | ✅ |

## CE — Content-addressed elision (replaces KV-cache)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **CE-1** | Autoregressive decode reuses the prior steps' prefix by κ-label (the SG/prefix case) — the unchanged prefix sub-graph is **elided**, not recomputed, with no mutable KV buffer. | exec reuse test (dispatch counting) | `hologram-ai-conformance` | ⛔ |
| **CE-2** | The elided generation is observationally identical to a non-elided recompute (held to the f64 reference). | exec equality test | `hologram-ai-conformance` | ⛔ |

> **Why no KV-cache.** hologram 0.5.0 has no KV-cache; in the UOR-native model
> a node's output κ-label is derived from its op signature + operand labels, so
> identical compute across decode steps is recognized and elided structurally.
> The legacy KV-cache (a mutable pre-allocated buffer) is abandoned, per the
> "abandon legacy optimization" mandate. See [`VERIFICATION.md`].

## ZM — Zero-movement / zero-copy / zero-cost

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **ZM-1** | Lowering introduces no per-node tensor copy; weights are borrowed/mmap'd from the archive, not cloned into a side store. | exec instrumentation (copy counter) | `hologram-ai-conformance` | ⛔ |
| **ZM-2** | Constant weights are content-addressed by their bytes (compose with hologram's warm-start/packing) — hologram-ai adds no copy-back on reuse. | exec test | `hologram-ai-conformance` | ⛔ |

## ZA — Zero runtime heap allocation

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **ZA-1** | A full prefill+decode inference call performs **zero** heap allocations after warm-up (the per-call scratch is reused across calls). | counting-allocator harness | `hologram-ai-conformance::alloc` | ⛔ |
| **ZA-2** | Lowering a graph allocates a bounded, input-independent amount (no per-node growth in steady state). | counting-allocator harness | `hologram-ai-conformance::alloc` | ⛔ |

## NS — no_std portability

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **NS-1** | The runtime core (`hologram-ai-quant` dequant + `hologram-ai-tokenizer` encode/decode, `--no-default-features`) builds on `wasm32-unknown-unknown`, `no_std`. | `just vv-wasm` | CI | ✅ |
| **NS-2** | The runtime core builds on `thumbv7em-none-eabi`, `no_std`. | `just vv-embedded` | CI | ✅ |
| **NS-3** | The runtime-core crates declare `#![no_std]` and pull no transitive `std`-only dependency: `hologram-ai-quant` (`half` w/o `std`); `hologram-ai-tokenizer` core (`hashbrown` + `regex-automata`, both `no_std`+`alloc`), with JSON loading / archive sections behind the `std` feature. | cross-target build | `just vv-portability` | ✅ |
| **NS-4** | The full **inference path** builds on `wasm32-unknown-unknown` and **executes a compiled `.holo` in the browser**: `hologram-ai-common` lowering + hologram's `exec`/`backend` (`parallel` off — rayon can't spawn threads on wasm). Verified by running the engine under node, output checked. This is the substrate of the browser GUI (ADR-0017). | `wasm-pack test --node` | `hologram-ai-wasm` (`describe`/`run` tests) | ✅ |

> **NS-4** extends NS from "the runtime core *builds* on wasm" to "the engine
> *runs a model* on wasm". `parallel` is now an opt-in feature (default-on for
> the native lib/CLI, off for `hologram-ai-wasm`), not a workspace pin. The
> browser GUI (`apps/web`) is the React app served as a static bundle, calling
> the real pipeline through `hologram-ai-wasm` — the same code paths as the CLI.

> The on-device runtime core is `hologram-ai-quant` (dequant) and the tokenizer
> encode/decode path; import/lowering/quantization are **compile-time host**
> concerns (architecture §2). `hologram-ai-quant` is `#![no_std]`+`alloc` and
> builds on wasm + bare-metal; converting the tokenizer encode path is the
> remaining NS work.

## PV — Performance (budgets, no bottleneck)

| ID | Statement | Enforcement | Witness | Status |
|---|---|---|---|---|
| **PV-1** | **No arbitrary limit at scale.** LLM-scale architectures (1B / 3B / 5B / 20B params) compile with no hardcoded cap, dimension clamp, or integer-overflow ceiling (ADR-060). Observed: 1B in ~0.6 ms, 20B in ~1.8 ms. | `tests/perf_contract.rs` + `bench scaling` | `just vv-perf` | ✅ |
| **PV-2** | **Content-addressed reuse is the win.** Re-executing an unchanged graph on the same inputs is a κ-label memo hit (O(1), no compute/copy) — far faster than recompute. Observed (256³ matmul): cold 1.93 ms vs reuse 176 ns (~11000×). | `tests/perf_contract.rs` (`content_addressed_reuse_beats_recompute`) | `just vv-perf` | ✅ |
| **PV-3** | **Bounded, weight-size-independent compile.** Compile cost tracks graph structure, not parameter count (weights never materialize at compile). | `tests/perf_contract.rs` (`compile_cost_is_independent_of_parameter_count`) | `just vv-perf` | ✅ |
| **PV-4** | Matmul throughput holds its efficiency across the 64/128/256/512 sweep (mirrors hologram's matmul scaling); every size compiles + runs end to end. | `bench scaling` + `tests/perf_contract.rs` (`matmul_sweep_runs_at_every_size`) | `just vv-perf` | ✅ |
| **PV-5** | **Full-weight billion-parameter execution.** A real forward pass over ~1B f32 weights (3.76 GB, weights resident in the content-addressed pool) runs end to end, and the κ-label reuse contract holds with weights resident. Observed (939M params): cold forward 4.41 s vs reuse 479 ns (**~9.2 M×**). Scales with host RAM via `HOLOGRAM_AI_PARAMS`. | `HOLOGRAM_AI_LARGE=1` release test | `tests/perf_contract_large.rs` (`just vv-perf-large`) | ✅ |
| **PV-6** | **The memory limit is the (packed) weight set, not an arbitrary cap.** A peak-allocator characterization shows: logical resident is bounded (the pool recycles), and forward peak grows ∝ the weight bytes — with **no dense-f32 dequant intermediate** (quant(i8) peak is ~4× under the f32 baseline at the same shape; every `Dequantize→MatMul` fuses). So quantization reduces the runtime ceiling proportionally (i8 4×, i4 8×); larger models need linearly more RAM, not the removal of a hidden limit. | characterization test | `tests/quant_memory_characterization.rs` | ✅ |

---

## Status discipline

This document is the authoritative invariant contract for the architecture in
[`specs/docs/architecture.md`](specs/docs/architecture.md). An invariant is not
"done" until its witness exists and passes under `just vv`. Status marks reflect
the current build only — a 🚧/⛔ mark is a true statement that the witness does
not yet pass, never a promise about ordering. Correctness is not negotiable
against convenience: an invariant that cannot be met means the implementation is
wrong, not that the invariant should be relaxed.
