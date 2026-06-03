# int8 Weight Quantization (Baseline) — Design

**Date:** 2026-06-03
**Status:** Approved (design); pending implementation plan
**Primary target:** wasm (browser); native is secondary

## Context

Single-token LLM decode at batch=1 is memory-bandwidth-bound: each token streams
the full weight set through the cores. On wasm specifically, weights are also
*download size* and sit under a browser memory ceiling (TinyLlama-1.1B in f32 ≈
4.4 GB likely will not load in a tab). The SIMD matmul/GEMV kernels just landed
(PR #32) made the *compute* fast, which makes weight bandwidth and footprint the
next bottleneck.

Quantizing linear weights f32 → int8 cuts weight bytes ~4×: smaller download,
fits browser memory, and ~2–4× faster bandwidth-bound decode. This is the first
of a sequence (int8 now; int4 and embedding/lm_head quantization as follow-ons).

## What already exists (reused, not rebuilt)

- Dequantize kernels for `DTYPE_I8`/`U8`/`I4`, per-tensor **and per-channel**
  scale/zero-point — `hologram/crates/hologram-backend/src/cpu/kernels.rs:176`.
- `matmul_dequant_float` (dequant-then-f32-matmul) calling `matmul_f32_blocked`
  — `hologram/crates/hologram-backend/src/cpu/float_kernels.rs:311`. Inherits
  the new SIMD/NEON/wasm kernels automatically. Output f32.
- `Dequantize→MatMul` fusion producing `matmul_dequant`, with a passing test
  from hand-built quantized graphs —
  `hologram/crates/hologram-exec/tests/quantization.rs:252`.
- Compiler lowers ONNX `DequantizeLinear` → `OpKind::Dequantize` with QuantAttrs
  (per-channel `channels`/`inner`/`axis`) — `hologram-compiler/src/lower.rs:697`.
- All dequant paths are `no_std`/wasm-compatible.
- `--quantize {none,q4_0,q8_0,q2_0}` flag + `QuantStrategy` enum exist but are
  **inert** — `hologram-ai/crates/hologram-ai/src/cli.rs:44`,
  `hologram-ai-common/src/lower/builder.rs:46`.

## What is missing (the work)

The **encoder**: nothing converts f32 weights → int. This design adds a
compile-time, per-channel symmetric int8 weight encoder and wires the (currently
inert) `QuantStrategy` to it. No new backend kernel is required for int8.

## Decisions (locked)

1. **Scope:** int8 first (this spec); int4 and embedding/lm_head are separate
   follow-on specs.
2. **Representation:** per-channel `DTYPE_I8` reusing the wired dequant→matmul
   path. NOT llama.cpp `Q8_0` block format.
3. **Weights quantized:** linear/matmul weights only — attention Q/K/V/O, MLP
   gate/up/down. Token embeddings, `lm_head`, LayerNorm/RMSNorm weights, and
   biases stay f32.
4. **Flag rename:** `--quantize` values become `none`/`int8`/`int4` (drop the
   misleading `q8_0`/`q4_0` block-format names). int4 is accepted by the parser
   but rejected as "not yet implemented" until the int4 spec lands.
5. **Accuracy bar:** output-logit cosine similarity ≥ **0.999** (f32 vs int8) on
   the CI fixture; also report max abs logit delta.

## Architecture

```
hologram-ai (NEW encoder pass)                 hologram (REUSED, unchanged)
────────────────────────────────              ────────────────────────────
ModelCompiler { quant_strategy }
  │  graph of f32 weights + MatMul
  ▼
quantize_weights pass (NEW)
  • find MatMul weight (B) constants
  • per-column symmetric int8 encode
  • replace f32 const → i8 const
  • insert Dequantize(i8, scales) → B
  │  graph: ... Dequantize → MatMul ...
  ▼
lower to hologram graph ──────────────▶ Dequantize→MatMul fusion
                                          ▼
                                        matmul_dequant (dequant→matmul_f32_blocked)
                                          ▼ (SIMD: NEON / wasm SIMD128)
                                        f32 output
```

### Unit: weight encoder (`hologram-ai-quant` — the existing no_std quant crate)
The encoder math (the inverse of the dequant unpackers already in this crate)
lives here so it stays `no_std`/wasm-clean and co-located with dequant. The
graph pass that *calls* it lives in `hologram-ai-common`'s lowering (where
`QuantStrategy`/`builder.rs` already are).
- **Does:** given an f32 weight tensor `[k, n]`, produce `(i8 bytes [k,n], scales
  [n] f32)` using per-column symmetric quantization:
  `scale_j = max(|B[:,j]|)/127` (guard `scale_j == 0` → 1.0);
  `q[i,j] = clamp(round(B[i,j]/scale_j), -127, 127)`.
- **Interface:** pure function `encode_int8_per_channel(&[f32], k, n) ->
  (Vec<i8>, Vec<f32>)`. No I/O, no graph deps → unit-testable in isolation.
- **Depends on:** nothing (math only); `no_std`-compatible.

### Unit: quantize-weights graph pass (`hologram-ai`)
- **Does:** when `quant_strategy == Int8`, walk the graph; for each MatMul whose
  B input is an f32 constant in the linear-weight set, call the encoder, swap the
  constant for the i8 constant, and insert a `Dequantize` node (per-channel,
  `axis = 1`, zero-point 0) feeding B.
- **Interface:** `apply(graph, strategy) -> graph`. Idempotent; a no-op for
  `None`.
- **Depends on:** the graph IR + the encoder. Selection of "linear weight"
  excludes embeddings/`lm_head`/norms/biases (by op role / tensor identity).

### Reused (no change): dequant kernels, fusion, `matmul_dequant`, lowering.

## Data flow / numerics

- B `[k,n]`, per **output channel** = per column (`n`). Dequant kernel indexes
  channel as `(elem / inner) % channels` with `channels = n, inner = 1` (axis 1).
- Symmetric: zero-point 0 (no zero-point vector emitted).
- Activations (A) stay f32; output stays f32. Only weights are quantized.

## Accuracy gate & testing

1. **Encoder unit tests:** round-trip max abs error ≤ `scale/2` per element;
   per-column scale correctness; zero-column guard.
2. **Graph accuracy test (CI):** small model fixture (reuse the existing
   `hologram-ai` mini fixture), run f32 vs int8, assert logit cosine ≥ 0.999 and
   report max abs delta.
3. **wasm:** run the quantized fixture under wasmtime (`wasm32-wasip1`,
   `+simd128`) — confirms the dequant-matmul path is correct on the primary
   target.
4. **Manual E2E (not CI):** compile TinyLlama with `--quantize int8`, compare
   logits / generated text vs f32, and record the `.holo` size reduction.
   Documented as an on-demand step (heavy AI-stack build).

## Risks / de-risk order

1. **Fusion firing (highest risk, do first):** a spike confirming that a
   `Dequantize` feeding a MatMul **B** input fuses to `matmul_dequant` for the
   encoder's exact node shape. The existing test fuses hand-built graphs; if the
   encoder's shape differs, fix the fusion matcher before building the pass.
2. **Per-channel axis correctness:** verify `axis=1` yields per-column scales in
   the dequant kernel (the channel-index math) via a tiny end-to-end numeric
   test before trusting the full pass.
3. **Weight-set selection:** mis-identifying a non-linear weight (e.g. an
   embedding used in a MatMul-shaped op) would quantize something it shouldn't;
   selection is by op role + tensor identity, covered by the accuracy test.

## Out of scope (explicit)

- int4 / group-wise quantization (separate spec).
- Embedding / `lm_head` quantization (separate spec).
- Activation quantization (weights only here).
- A fused-int matmul (int8×f32 direct). The dequant-then-matmul path already runs
  on the SIMD kernel; a fused-int kernel is a later optimization, not required
  for the size/bandwidth win.
- llama.cpp `Q4_0/Q8_0` block-format compatibility.
