# Plan 053: Tier 3 — Q2 Quantization + Speculative Decoding → 60-160 tok/s

## Context

TinyLlama 1.1B decode at 43 tok/s (21ms/step). Profiling: 96% matmul time at AMX
hardware throughput. All Tier 1-2 optimizations are exhausted — AMX is processing
data as fast as physically possible. The ONLY path forward is **reducing data per step**.

Two approaches stack multiplicatively:
- **Phase 1: Q2 quantization** — 2-bit weights (4 centroids), 2x less data → ~60-80 tok/s
- **Phase 2: Speculative decoding** — 2-3x effective tokens per forward pass → ~120-160 tok/s

## Critical Insight

Q2 only delivers 2x if we **bypass the dequant→f32 cache** and use the LUT-GEMM path
directly. The dequant cache is always f32 (same size regardless of Q2 vs Q4). For M=1
decode, Q2 LUT-GEMM reads 1 MB per 2048×2048 matmul vs 16 MB f32 — a 16x bandwidth
reduction that dominates even though NEON is slower per-op than AMX.

---

## Phase 1: Q2 Quantization (Target: 60-80 tok/s)

### Phase 1A: Core Q2 types + kernel (hologram base)

| File | Change |
|------|--------|
| `hologram-exec/src/lut_gemm/psumbook.rs` | Add `Q2_LEVELS=4`, `Psumbook2` (16 bytes, 4 f32 slots) |
| `hologram-exec/src/lut_gemm/orbit.rs` | Add `OrbitMap2` (4 entries) |
| `hologram-exec/src/lut_gemm/quantize.rs` | Add `QuantizedWeights2`, `quantize_2bit()`, `pack_q2()`/`unpack_q2()` (4 indices/byte), `dequantize_error_q2()` |
| `hologram-exec/src/lut_gemm/matmul.rs` | Add `lut_gemm_2bit()` kernel (Psumbook2 accumulation, NEON + scalar) |
| `hologram-exec/src/lut_gemm/mod.rs` | Export Q2 types |

**Q2 packing:** 4 indices per byte (2 bits each). For 2048 columns: 512 bytes/row (vs 1024 Q4, 2048 Q8).

**Psumbook2 inner loop:**
```
For each weight index (0-3):
    book.sums[idx] += a_val
Final: output = Σ(book.sums[i] * centroids[i]) for i=0..3
```
Only 4 multiply-adds for the final dot product. Extremely cache-friendly.

### Phase 1B: Q2 tape dispatch (hologram base)

| File | Change |
|------|--------|
| `hologram-exec/src/tape.rs` | Add `MatMulLut2(ConstantId)`, `MatMulLut2Activation(ConstantId, FloatOp)` to TapeKernel |
| `hologram-exec/src/tape.rs` | Add `dispatch_lut_gemm_2()` — **always uses LUT-GEMM, never dequant+BLAS** |
| `hologram-exec/src/kv/weight_cache.rs` | Add `CachedWeight::Q2`, `get_q2()` |
| `hologram-exec/src/tape_builder.rs` | Wire `MatMulLut2` in `resolve_kernel()` |

**Key decision:** `dispatch_lut_gemm_2` uses LUT-GEMM on ALL platforms (including macOS).
Q2's 16x bandwidth reduction dominates over AMX's compute advantage for M=1 decode.
For M>1 (prefill), optionally dequant+BLAS if benchmarks show it's faster.

### Phase 1C: Q2 graph ops (hologram base)

| File | Change |
|------|--------|
| `hologram-graph/src/graph/mod.rs` | Add `MatMulLut2(ConstantId)` to `GraphOp` |
| `hologram-graph/src/builder/mod.rs` | Add `matmul_lut_2bit()` builder method |

### Phase 1D: Q2 compiler integration (hologram-ai)

| File | Change |
|------|--------|
| `hologram-ai-common/src/lower/builder.rs` | Add `QuantStrategy::Q2_0`, `try_convert_f32_to_lut2()` |
| `hologram-ai/src/cli.rs` | Add `"q2_0"` to `--quantize` match |

### Phase 1E: Quality validation

Q2 (4 centroids per tensor) is aggressive. Expected ~20-30% relative RMSE.

**Validation plan:**
1. `dequantize_error_q2()` on all TinyLlama weight tensors — measure per-layer RMSE
2. E2E: "The capital of France is" → must produce coherent English
3. **If quality unacceptable:** Mixed precision — Q2 for FFN weights (gate/up/down, 75% of
   params), Q4 for attention projections (Q/K/V/O, 25% of params). FFN weights have wider
   distributions → more robust to aggressive quantization.

### Phase 1F: Expected performance

| Path | Weight data per 2048×2048 matmul | tok/s (M=1 decode) |
|------|----------------------------------|---------------------|
| f32 BLAS (current f32 weights) | 16 MB | ~43 tok/s (current) |
| Q4 dequant → f32 BLAS (current Q4) | 16 MB (dequant cache) | ~43 tok/s |
| Q2 LUT-GEMM (Psumbook2) | **1 MB** (packed indices) | **~60-80 tok/s** (projected) |

**Milestone: 60+ tok/s on TinyLlama decode.**

---

## Phase 2: Speculative Decoding (Target: 120-160 tok/s effective)

### Prerequisite: Variable-length execution (Plan 045)

Decode tape compiled at seq=1. Verification needs seq=N (N=4-8).

**Preferred approach:** Compile a separate **verification tape** at seq=8 (fixed).
Store as third component in the pipeline archive alongside prefill and decode tapes.
Simpler than 0-sentinel dims and sufficient for speculative decoding.

### Phase 2A: SpeculativeDecoder (hologram-ai)

| File | Change |
|------|--------|
| `hologram-ai/src/speculative.rs` | **New**: `SpeculativeDecoder` struct, acceptance/rejection logic |
| `hologram-ai/src/commands/run_cmd.rs` | `--draft-model`, `--draft-steps` CLI args, wire into decode loop |
| `hologram-ai/src/compiler.rs` | Compile verification tape at seq=N |

**Algorithm (Leviathan et al. 2023):**
1. Draft model generates N tokens autoregressively (fast, each at M=1)
2. Target model verifies all N+1 tokens in one forward pass (M=N+1)
3. For each position i: accept if `rand() < p_target[token_i] / p_draft[token_i]`
4. First rejection at position j → discard tokens j+1..N, resample from adjusted target distribution
5. On average, 60-70% acceptance → 2-3 effective tokens per target forward pass

**Start with self-speculative:** Same model as both target and draft (simpler, no second
model needed). Lower acceptance rate (~50%) but still ~2x effective throughput. True
draft model (smaller) is a follow-up.

### Phase 2B: KV cache coordination

Both target and draft need independent KV cache states. On rejection at position j:
- Draft cache: reset `write_pos` to pre-speculation position
- Target cache: already correct (verified up to position j)

---

## Implementation Order

```
Phase 1A-1C: Q2 types + kernel + graph ops (hologram base)     ← START HERE
    ↓
Phase 1D: Q2 compiler integration (hologram-ai)
    ↓
Phase 1E-1F: Quality validation + benchmark → GATE: 60+ tok/s?
    ↓ (can start in parallel after Phase 1A)
Phase 2 prereq: Verification tape at seq=N
    ↓
Phase 2A-2B: Speculative decoding
    ↓
Combined: Q2 + speculative → GATE: 120+ tok/s?
```

## Tok/s Milestones

| Checkpoint | Expected | Gating criteria |
|---|---|---|
| Current baseline | 43 tok/s | — |
| Phase 1: Q2 LUT-GEMM | 60-80 tok/s | Ship if ≥ 60 |
| Phase 2: Q2 + Speculative (N=6, 65% accept) | 120-160 tok/s | Effective throughput |
| Future: Q2 + Speculative + true draft model | 150-200+ tok/s | — |

## Verification

1. `cargo test -p hologram-exec` — Q2 matmul correctness vs naive reference
2. `cargo test -p hologram-ai-conformance` — all op conformance unchanged
3. Recompile TinyLlama: `hologram-ai compile --quantize q2_0 --seq-len 24`
4. Run: verify coherent English output + check tok/s ≥ 60
5. Profile: `HOLOGRAM_PROFILE=1` — verify LUT-GEMM path used (not dequant+BLAS)
