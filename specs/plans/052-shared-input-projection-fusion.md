# Plan: Projection Fusion — 66 fewer BLAS calls per decode step

## Context

Profiling TinyLlama decode at 43 tok/s (21ms/step) shows 156 BLAS calls per step. Two groups of matmuls share inputs and can be fused:

**QKV projections (attention):** 3 calls → 1, saves 44 calls
- MatMul(1×2048×2048) Q + MatMul(1×2048×256) K + MatMul(1×2048×256) V
- → Fused MatMul(1×2048×2560) + 3 zero-cost Slices

**Gate+Up projections (FFN):** 2 calls → 1, saves 22 calls
- MatMul(1×2048×5632) gate + MatMul(1×2048×5632) up
- → Fused MatMul(1×2048×11264) + 2 zero-cost Slices

Total: **66 fewer BLAS calls → ~1.5ms saved → ~48-52 tok/s**

Key: fusion happens at AiGraph level (f32 weights) BEFORE Q4 quantization in lowering. The combined gate+up weight gets Q4-quantized as one tensor — no re-quantization needed.

## Implementation

### 1. New file: `hologram-ai-common/src/opt/shared_input_projection_fusion.rs`

**Pattern detection:** Before AttentionFusion runs, the graph has separate MatMul nodes for Q/K/V. Find groups of 3 MatMul nodes that:
- Share the same `inputs[0]` (hidden state)
- Have `inputs[1]` as a 2D parameter `[K, N_i]` with same K
- Have N dimensions matching Q/K/V pattern: one large N (Q) + two equal smaller Ns (K, V)

**Weight concatenation:** At compile time, concatenate W_q `[K, N_q]`, W_k `[K, N_k]`, W_v `[K, N_v]` row-by-row into W_qkv `[K, N_q+N_k+N_v]`. Register as new `AiParam::Inline`.

**Graph rewrite:** Replace 3 MatMuls with:
```
hidden → MatMul(hidden, W_qkv) → qkv_out [new tid]
                                → Slice(axis=-1, 0..N_q) → [reuse Q MatMul's output tid]
                                → Slice(axis=-1, N_q..N_q+N_k) → [reuse K MatMul's output tid]
                                → Slice(axis=-1, N_q+N_k..N_total) → [reuse V MatMul's output tid]
```

Reusing original output TensorIds means all downstream consumers (Reshape, Transpose, etc.) need no rewiring.

### 2. Register the pass

**`hologram-ai-common/src/opt/mod.rs`**: Add `pub mod shared_input_projection_fusion;` + re-export.

**`hologram-ai-common/src/opt/pipeline.rs`**: Insert between AddRmsNormFusion and PositionIdsInjection:
```rust
Box::new(AddRmsNormFusion),
Box::new(SharedInputProjectionFusion),    // NEW
Box::new(PositionIdsInjection),
Box::new(AttentionFusion),
```

### 3. Fix attention guard in `hologram-ai-common/src/lower/builder.rs`

The fused MatMul has N=2560 (for TinyLlama). The attention guard checks if N matches `attn_dims` (2048 or 256). N=2560 would NOT match → the lowering would Q4-quantize it (wrong: dequant cache makes Q4 slower for cached decode).

Fix: extend `attn_dims` to include the fused sum: `N_q + 2*N_kv`. After collecting head dims from GQA nodes:
```rust
// Also skip fused QKV matmuls (N = n_q + n_k + n_v)
for &nq in &q_dims {
    for &nkv in &kv_dims {
        attn_dims.push(nq + 2 * nkv);
    }
}
```

### 4. No hologram base changes needed

- `AiOp::Slice` already exists with axes/starts/ends/steps fields
- `FloatOp::Slice` lowering already handles single-axis contiguous slices
- `InlineSlice` tape kernel is zero-copy (metadata adjustment only)
- `InlineMatMul` with baked m/k/n handles the fused matmul

**Reference files (read-only):**
- `opt/swiglu_fusion.rs` — pattern for consumer/producer maps + graph rewrite
- `opt/attention_fusion.rs` — pattern for tracing Q/K/V paths
- `ir/op.rs:152` — AiOp::Slice definition
- `lower/strategy.rs` — Slice lowering (already handles this)

## Verification

1. `cargo test -p hologram-ai-common` — new unit tests for the fusion pass
2. `cargo test -p hologram-ai-conformance` — op conformance unchanged
3. Recompile TinyLlama: `hologram-ai compile --quantize q4_0 --seq-len 24`
4. Run decode: verify "The capital of France is Paris" + check tok/s improvement
5. Profile: `HOLOGRAM_PROFILE=1` should show fewer MatMul calls per step

### 5. Gate+Up FFN Projection Fusion (same pass)

Extend `SharedInputProjectionFusion` to also handle 2-MatMul groups (gate+up):

**Pattern:** Two MatMul nodes sharing `inputs[0]` with equal output dimensions N:
```
hidden → MatMul(hidden, W_gate) → gate_out  (→ SiLU → FusedSwiGLU)
       → MatMul(hidden, W_up)   → up_out    (→ FusedSwiGLU)
```

**Detection:** After grouping MatMuls by shared input, find pairs where both have the same N (e.g., 5632) and N > hidden_size (to distinguish from QKV projections).

**Rewrite:**
```
hidden → MatMul(hidden, W_gate_up) → gate_up_out [new tid]
                                   → Slice(axis=-1, 0..N) → [reuse gate MatMul's output tid]
                                   → Slice(axis=-1, N..2N) → [reuse up MatMul's output tid]
```

**Q4 quantization:** The fused weight `[K, 2N]` is f32 at this point. During lowering, `try_convert_f32_to_lut4` sees one large MatMul with `rows=K, cols=2N`. It Q4-quantizes the combined weight with 16 centroids across the full distribution. This is fine — the gate and up weights have similar distributions (both are FFN projections). The Slices after the MatMulLut4 output produce the same f32 values.

**Attention guard:** N=11264 (2×5632) does NOT match any attention dim, so the guard won't skip it. But we WANT it Q4-quantized (FFN weights). The guard correctly allows it through since 11264 ∉ {2048, 256, 2560}.

**Interaction with SwiGluFusion:** SwiGluFusion runs BEFORE this pass in the pipeline. It has already fused `SiLU(gate) × up` into `FusedSwiGLU(gate_tid, up_tid)`. Our pass then fuses the two upstream MatMuls. The FusedSwiGLU node still receives the correct gate/up tensors via the reused output TensorIds — no conflict.

### Pipeline order (updated)

```rust
Box::new(SwiGluFusion),               // fuse SiLU+Mul first
Box::new(MatMulActivationFusion),
Box::new(AddRmsNormFusion),
Box::new(SharedInputProjectionFusion), // NEW: fuse QKV + gate+up
Box::new(PositionIdsInjection),
Box::new(AttentionFusion),
```

## Updated key files

| File | Change |
|------|--------|
| `crates/hologram-ai-common/src/opt/shared_input_projection_fusion.rs` | **New**: ~250 lines, handles both QKV (3-way) and gate+up (2-way) |
| `crates/hologram-ai-common/src/opt/mod.rs` | Add module + re-export |
| `crates/hologram-ai-common/src/opt/pipeline.rs` | Insert in mvp() pipeline |
| `crates/hologram-ai-common/src/lower/builder.rs` | Extend attn_dims for fused QKV dim |

## Expected impact

| Optimization | BLAS calls saved | Time saved | Cumulative tok/s |
|-------------|-----------------|------------|-----------------|
| QKV fusion (22 layers × 2) | 44 | ~0.9ms | ~47 tok/s |
| Gate+Up fusion (22 layers) | 22 | ~0.6ms | ~50-52 tok/s |
| **Total** | **66** | **~1.5ms** | **~50-52 tok/s** |
