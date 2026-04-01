# Plan 042: Archive Size Fix + MatMulActivationFusion + Q4 Performance

**Status:** Open
**Created:** 2026-04-01
**Branch:** `feat/cpu-inference-perf`
**Depends on:** Plan 040 (performance), Plan 041 (ONNX correctness)

## Problem

Three blockers preventing 39.1+ tok/s on TinyLlama ONNX:

1. **Archive bloat** — 9.4 GB for a 1.1B Q4 model (should be ~0.5 GB).
   Weights duplicated in sub-archive + shared blob.
2. **Missing MatMulActivationFusion** — the pass that fuses MatMul + SiLU/GeLU
   was removed. This was responsible for 20.5 → 39.1 tok/s (2x).
3. **Q4 LUT-GEMM slower than BLAS** — psumbook kernel doesn't use AMX on
   Apple Silicon. Q4 runs at 1.0 tok/s vs f32 BLAS at 2.5 tok/s.

## Fix 1: Archive Weight Deduplication

**Root cause:** `compile_components` embeds weights in the first sub-archive (4.4 GB)
AND adds them to the shared WeightStore blob (4.4 GB more). The decode sub-archive
(~450 KB) shares weights via zero-copy borrow at load time.

**Fix:** For LLM pipeline (prefill+decode with same weight_group), skip the
shared WeightStore entirely. Embed weights ONLY in the first sub-archive.
The decode component resolves weights from the prefill component at load time
(already working via `set_weights_borrowed`).

**Files:**
- `crates/hologram-ai/src/compiler.rs` — `compile_components()`: skip `weight_store.insert()`
  for components that share the same weight_group as an already-embedded component.

**Expected result:** Q4 archive = ~0.5 GB (quantized weights only + graph).
F32 archive = ~4.1 GB (weights + graph). No duplication.

## Fix 2: MatMulActivationFusion Pass

**What it does:** Pattern-matches chains of `MatMul → Activation` (SiLU, GeLU, ReLU)
in the AiGraph and fuses them into `AiOp::MatMulSilu`, `AiOp::MatMulGelu`,
`AiOp::MatMulRelu`. The lowering (`wrap_graph_op` in `strategy.rs`) already handles
these fused variants, emitting `GraphOp::FusedMatMulActivation` which maps to the
`InlineMatMulActivation` tape kernel that applies activation in-register.

**Implementation:**
1. Create `crates/hologram-ai-common/src/opt/matmul_activation_fusion.rs`
2. Pattern: find MatMul node → single consumer is Activation (SiLU/GeLU/ReLU) →
   replace with fused variant, bypass the activation node.
3. Wire into `OptPipeline::mvp()` in `pipeline.rs` (after AttentionFusion, before
   lowering).

**Where the fused variants already exist:**
- `AiOp::MatMulSilu`, `AiOp::MatMulGelu`, `AiOp::MatMulRelu` — defined in ir/op.rs
- `wrap_graph_op()` in `strategy.rs` — maps them to `GraphOp::FusedMatMulActivation`
- `InlineMatMulActivation` tape kernel — applies activation in-register

**What's missing:** The optimization PASS that creates the fused AiOp variants.
The lowering and kernel are wired; only the pass is missing.

**Files:**
- NEW: `crates/hologram-ai-common/src/opt/matmul_activation_fusion.rs`
- `crates/hologram-ai-common/src/opt/pipeline.rs` — add to MVP pipeline
- `crates/hologram-ai-common/src/opt/mod.rs` — register module

**Expected result:** ~2x decode speedup (MatMul + activation eliminated intermediate
buffer + apply activation in-register during matmul writeback).

## Fix 3: Q4 Kernel Performance (AMX Hybrid)

**Root cause:** The LUT-GEMM psumbook kernel accumulates Q4 dot products using scalar
code. On Apple Silicon, Accelerate BLAS dispatches to the AMX coprocessor for matrix
multiply, achieving ~2 TFLOPS. The psumbook kernel achieves ~0.1 TFLOPS.

**Options (pick one):**

### Option A: Dequant-per-tile → cblas_hgemm (recommended)
- During matmul, dequant each KC×NR tile from Q4 to f16
- Call `cblas_hgemm` (Accelerate half-precision GEMM → AMX)
- Gets both Q4 compression (reduced archive + bandwidth) AND AMX speed
- **File:** hologram-exec `float_dispatch/matmul.rs` (hologram base)

### Option B: Fused dequant + sgemm
- Dequant full Q4 weight matrix to f32 once, cache the result
- Use cblas_sgemm (f32 AMX) on the dequantized weights
- Simpler but loses memory savings (weights expand to f32 at runtime)
- **File:** hologram-exec `float_dispatch/matmul.rs` (hologram base)

### Option C: Keep BLAS for large MatMuls, Q4 for small
- Only quantize MatMuls below a size threshold where psumbook is competitive
- Large MatMuls (>256K elements) stay at f32 BLAS
- Compromise: some bandwidth savings, full AMX speed for large ops
- **File:** hologram-ai-common `lower/builder.rs` (size gate)

## Fix 4: Archive Compression

**Explore:** `hologram-compression` crate for on-disk compression. Two strategies:

### Strategy A: Compress-at-write, decompress-at-load
- `HoloWriter::compress_weights()` — compress weight section with zstd/lz4
- `HoloLoader::from_path()` — detect compression flag, decompress to cache file
- Already partially implemented in HoloRunner (`is_compressed`, `decompress_archive`)

### Strategy B: Streaming decompression
- Decompress weight pages on-demand during execution
- Requires changes to mmap loader — read compressed, decompress per-page
- More complex but avoids full decompression at load time

## Implementation Order

```
Session N+1:
  [1] Archive dedup — fix compile_components weight_store for LLM pipeline
  [2] MatMulActivationFusion pass — pattern match + wire into pipeline
  [3] Test: TinyLlama f32 with fusion → target 5-10 tok/s
  [4] Test: TinyLlama Q4 → verify archive <1 GB, correct output

Session N+2:
  [5] Q4 AMX hybrid (Option A or B) — hologram base matmul.rs
  [6] Test: TinyLlama Q4 with AMX → target 20-40 tok/s
  [7] Compression — wire hologram-compression for on-disk archives

Session N+3:
  [8] Variable-length execution fix (Plan 041) — resolve_size for all seq-dependent ops
  [9] Speculative decoding (Plan 040 Tier 2.1) → target 60-100 tok/s
```

## Verification

- TinyLlama ONNX: "The capital of France is Paris." (greedy, correct)
- Archive size: Q4 < 1 GB, f32 < 4.5 GB
- tok/s checkpoints: f32 ~5-10, Q4+AMX ~20-40, Q4+AMX+spec ~60-100
- All conformance tests pass
- No regression in non-LLM models (BERT, ResNet, SD)
