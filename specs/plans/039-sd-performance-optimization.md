# Plan 039: Stable Diffusion Performance Optimization

**Status:** Complete
**Created:** 2026-03-31
**Scope:** hologram-exec (hologram base), hologram-ai (CLIP quantization only)

## Context

SD v1.5 pipeline runs 337s on CPU (M-series Mac with Accelerate BLAS) for 512×512. UNet dominates (~127s for 10 CFG-doubled steps).

**Already exists in hologram base (verified):**
- `FusedConv2dActivation` GraphOp + `InlineConv2dActivation` / `InlineConv2dBiasActivation` TapeKernels + fusion pass in `float_fusion.rs` — Conv2d+activation fusion is done
- `FusedGroupNormActivation` GraphOp + `InlineGroupNormActivation` TapeKernel — graph-level fusion fires
- Activation checkpointing infrastructure — skip connection identification, eviction in `tape.rs`
- Sparse V attention — implemented in Plan 038
- Pre-allocated Conv2d tile buffers
- MatMul+bias+activation fusion, LUT-GEMM+activation fusion

**Still needed (this plan):**
1. GroupNorm `_into` kernel — the fused `InlineGroupNormActivation` dispatch still allocates intermediate Vec
2. Depthwise Conv2d fast path — no special handling
3. Winograd F(2,3) — not implemented
4. CLIP weight quantization wiring
5. Activation checkpointing wiring — infrastructure exists but recomputation not actively triggered

## Phase 1: GroupNorm `_into` + inline activation

**Problem:** `InlineGroupNormActivation` dispatch (tape.rs ~1974) calls `dispatch_group_norm()` → allocates `Vec<u8>` → extends into `out_buf` → applies activation in second pass. The fusion pass fires correctly at graph level, but the kernel wastes 2 passes + 1 allocation.

**Files (hologram base):**
- `hologram-exec/src/float_dispatch/norm.rs` — add `dispatch_group_norm_into` and `dispatch_group_norm_activation_into`
- `hologram-exec/src/tape.rs` — update `InlineGroupNormActivation` dispatch (~line 1974)

**Implementation:**
- [ ] `dispatch_group_norm_into(inputs, num_groups, epsilon, out_buf)` — copy `dispatch_group_norm` logic but resize `out_buf` and work in-place instead of allocating+returning
- [ ] `dispatch_group_norm_activation_into(inputs, num_groups, epsilon, activation, out_buf)` — fuse activation into the normalize-and-scale inner loop: `let normed = (*v - mean) * inv_std * s + b; *out = apply_activation(normed, activation);`
- [ ] Update tape.rs `InlineGroupNormActivation` to call `dispatch_group_norm_activation_into` directly instead of `dispatch_group_norm` + `extend_from_slice` + `apply_activation_to_out_buf`
- [ ] Update tape.rs `InlineGroupNorm` (if unfused GroupNorm exists in tape dispatch) to call `dispatch_group_norm_into`

**Tests:**
- [ ] `group_norm_into_matches_allocating` — bit-exact match
- [ ] `group_norm_silu_fused_matches_separate` — within 1e-5
- [ ] `group_norm_activation_into_sd_shapes` — [1,320,64,64], [1,640,32,32], [1,1280,16,16]

**Impact:** ~2% overall. 40 GroupNorm+SiLU per UNet step.

## Phase 2: Depthwise Conv2d fast path

**Problem:** Depthwise convolutions (groups == channels, 1 input channel per group) use generic im2col+GEMM, creating a column matrix for a single multiply-add per output pixel.

**Files (hologram base):**
- `hologram-exec/src/float_dispatch/conv.rs` — add `conv2d_depthwise` early return in `conv2d_core` (or `dispatch_conv2d_direct`)

**Implementation:**
- [ ] At top of conv dispatch, detect `ic_per_group == 1` (i.e., `group == in_channels`), route to `conv2d_depthwise`
- [ ] `conv2d_depthwise(data, weight, bias, ...)` — direct nested loop per (batch, channel, oh, ow): accumulate `data[c, oh*sh+kh, ow*sw+kw] * weight[c, kh, kw]` over kernel window
- [ ] Specialize 3×3 kernel with unrolled inner loop
- [ ] Apply bias inline
- [ ] Support padding, stride, dilation

**Tests:**
- [ ] `conv2d_depthwise_matches_generic` — depthwise (group=channels=64) on [1,64,32,32] with 3×3, bit-exact vs im2col path
- [ ] `conv2d_depthwise_stride2` — verify output dims
- [ ] `conv2d_depthwise_with_bias` — bias applied correctly
- [ ] `conv2d_depthwise_dilation` — dilation=2 case

**Impact:** ~2% overall. Low risk, simple implementation.

## Phase 3: Winograd F(2,3) for 3×3 convolutions

**Problem:** 3×3 stride=1 convolutions are ~60-70% of UNet Conv2d compute. Winograd F(2,3) reduces multiplications from 9 to 4 per 2×2 output tile (2.25× theoretical).

**Files (hologram base):**
- `hologram-exec/src/float_dispatch/conv.rs` — add `conv2d_winograd_f23`

**Implementation:**
- [ ] Gate: `kh == 3 && kw == 3 && sh == 1 && sw == 1 && dh == 1 && dw == 1 && ic_per_group >= 16`
- [ ] Weight transform (once per unique weight): `U = G × g × G^T` where G is the 4×3 Winograd transform matrix, g is 3×3 filter. Result is 4×4. Cache via `thread_local!` keyed by weight pointer.
- [ ] Input tile transform: for each overlapping 4×4 tile (stride-2 tiling), compute `V = B^T × d × B`. Pack all tiles into batched format `[16, IC, n_tiles]`.
- [ ] Batched element-wise GEMM: for each of 16 Winograd elements, compute `M[e] = U[e] × V[e]` via BLAS sgemm `[OC, IC] × [IC, n_tiles] → [OC, n_tiles]`
- [ ] Output transform: `Y = A^T × M × A` (4×4 → 2×2), scatter to output with bias
- [ ] Handle partial tiles at spatial boundaries (pad or fallback)

**Tests:**
- [ ] `conv2d_winograd_matches_im2col` — 3×3 stride=1 pad=1 on [1,64,32,32], within 1e-4
- [ ] `conv2d_winograd_odd_spatial` — [1,64,33,33], partial tiles handled
- [ ] `conv2d_winograd_realistic_sd_shapes` — [1,320,64,64], [1,640,32,32], [1,1280,16,16]
- [ ] `conv2d_winograd_fallback_stride2` — stride=2 routes to im2col, not Winograd
- [ ] `conv2d_winograd_numerical_accuracy` — 99.9th percentile element error < 1e-4

**Impact:** ~15-20% overall. Highest single-phase impact.

## Phase 4: CLIP weight quantization

**Problem:** CLIP text encoder uses f32 weights (~500MB). Quantizing to Q8/Q4 via existing fused dequant-matmul halves bandwidth.

**Files (hologram-ai):**
- `hologram-ai/src/compiler.rs` or `hologram-ai/crates/hologram-ai-common/src/lower/builder.rs` — verify `--quantize q8_0` fires `try_convert_f32_to_lut4` for non-LLM models

**Implementation:**
- [ ] Verify `--quantize q8_0` on a CLIP .holo archive — check if quantization pass fires or skips
- [ ] If it gates on LLM detection, remove or relax the gate to include any model with MatMul weights above size threshold
- [ ] Verify quality: CLIP embeddings with Q8 should have cosine similarity > 0.999 vs f32

**Tests:**
- [ ] `clip_q8_compile_fires_conversions` — compile CLIP with `--quantize q8_0`, verify MatMulLut nodes in graph
- [ ] `clip_q8_embedding_quality` — cosine similarity > 0.999

**Impact:** CLIP 9.9s → ~5-6s.

## Phase 5: Wire activation checkpointing

**Problem:** Checkpointing infrastructure exists in tape.rs (skip connection identification, checkpoint_map) but recomputation is not actively triggered in the eviction loop.

**Files (hologram base):**
- `hologram-exec/src/tape.rs` — activate recompute in eviction loop

**Implementation:**
- [ ] Add `pub checkpoint_enabled: bool` field to `EnumTape` (default false)
- [ ] In eviction loop, after decrementing consumer count to 0, check `checkpoint_map`: if node is checkpointable and has remaining future consumers, evict buffer and store recompute instruction index
- [ ] When a future consumer needs an evicted-checkpointed buffer, recompute it by re-executing its producer instruction
- [ ] Handle cascading: if producer's inputs were also evicted, recompute them first (max depth 3)
- [ ] Wire through CLI: `--checkpoint` flag

**Tests:**
- [ ] `checkpoint_evict_and_recompute` — synthetic graph with skip connection, verify correct output
- [ ] `checkpoint_peak_memory_reduced` — compare peak arena slots with/without
- [ ] `checkpoint_cascade_depth_limit` — deep skip chain respects max_depth=3

**Impact:** Memory only — 51GB → ~2-3GB for VAE. ~30% slower due to recomputation.

## Expected cumulative impact

| Phase | What | Impact |
|-------|------|--------|
| 1 | GroupNorm `_into` kernel | ~2% speedup |
| 2 | Depthwise fast path | ~2% speedup |
| 3 | Winograd 3×3 | ~15-20% speedup |
| 4 | CLIP Q8 | ~5s saved |
| 5 | Checkpointing | 51GB → 2-3GB memory |

**Total: 337s → ~260s (1.3×), VAE memory 51GB → 2-3GB**

## Verification

1. `cargo test` in hologram base after each phase
2. `cargo clippy -- -D warnings` clean in both repos
3. SD pipeline E2E test: output PSNR > 40dB vs baseline
4. Per-component timing comparison
