# Plan 037: Fused Dequant-MatMul & Kernel Optimizations

**Status:** Active
**Created:** 2026-03-30
**Scope:** hologram-exec (hologram base), hologram-archive

## Motivation

Q4_0 Psumbook kernel is **30x slower than f32 BLAS** (SPRINT.md P7). The root cause: the Q4_0
path in `dispatch_gemm` fully dequantizes the weight matrix to f32 via `decode_weights` before
running `matmul_k_outer`, doubling memory bandwidth.

Key insight: fuse dequantization with matrix multiplication so weights are dequantized per-block
(32 values) in registers, never materializing the full f32 matrix.

## What Already Exists (verified)

- **CPU matmul** (`float_dispatch/matmul.rs`): Goto/BLIS-style with KC=256 L2 cache blocking,
  B-panel packing, const-generic `micro_kernel<MR=4, NR=8>`, rayon parallel M-tile distribution,
  Accelerate BLAS on macOS. Well-optimized for f32.
- **SIMD precedent**: `hologram-core/src/view/simd.rs` has AVX2/SSE4.2/NEON intrinsics using
  compile-time `#[cfg(target_arch, target_feature)]`.
- **LUT-GEMM**: Separate algorithm (psumbook accumulation). Fiber kernel (16-pass radix),
  tiled kernel (4-column), scalar fallback.
- **HoloArchive**: Page-aligns sections (4KB) but not individual tensors.

## Phase 1: Fused Q4_0 Dequant-MatMul (highest impact)

**Goal:** Eliminate the full-matrix dequantization that causes 30x slowness.

**File:** `hologram-exec/src/float_dispatch/matmul.rs` (or new `dequant_matmul.rs`)

**Approach:** New function `matmul_dequant_q4_0(a, b_q4, out, m, k, n)` that:
- Uses same KC-blocked, MR×NR tiled structure as `matmul_k_outer`
- Inner loop dequantizes Q4_0 blocks on-the-fly (18 bytes → 32 f32s in registers)
- B-panel packing becomes dequant-into-pack: dequant block directly into packed_b buffer

**Integration:** `dispatch_gemm()` when `quant_b == 1` calls fused kernel instead of
`decode_weights` → `matmul_k_outer`.

**Expected impact:** 5-15x speedup (30x gap → ~2-6x).

## Phase 2: Adaptive Micro-Kernel Selection for Remainders

**Goal:** Eliminate scalar remainder paths in `matmul_k_outer`.

**File:** `hologram-exec/src/float_dispatch/matmul.rs`

**Current weakness:**
- Remainder columns (n%8): scalar per-column, no NR-width vectorization (lines 716-737)
- Remainder rows (m%4): iterates full N, no MR blocking (lines 780-797)

**Approach:** Instantiate existing `micro_kernel<MR, NR>` at smaller tile sizes:
- `micro_kernel::<MR, 4>` for remainder columns when n_rem >= 4
- `micro_kernel::<1, NR>` for remainder rows (NR-wide vectorization)
- `micro_kernel::<1, 4>` for corner case

## Phase 3: SIMD Psumbook Dot Product

**Goal:** Accelerate LUT-GEMM dot product phase with explicit SIMD.

**File:** `hologram-exec/src/lut_gemm/psumbook.rs`

**Approach:** Add NEON/AVX2 implementations of `Psumbook8::dot()` (256-element f32 dot product)
following hologram-core's `view/simd.rs` pattern.

## Phase 4: Page-Aligned Tensors in HoloArchive

**Goal:** Enable zero-copy GPU weight loading.

**Files:** `hologram-archive/src/format/header.rs`, `writer/holo_writer.rs`, `weight/index.rs`

**Approach:** Add `FLAG_TENSOR_PAGE_ALIGNED` header flag. When enabled, insert `align_to_page()`
padding between tensors in the weight blob. Reader unchanged (uses WeightIndex offsets).

**Follow-up:** Metal backend switches from `new_buffer_with_data()` (copies) to
`newBuffer(bytesNoCopy:)` for page-aligned tensors.

## Execution Order

All phases are independent. Phase 1 is highest priority (addresses 30x gap).
Phase 2 is lowest risk. Phases 3 and 4 can run in parallel with anything.

## Key Files

| File | Phase | Change |
|------|-------|--------|
| `hologram-exec/src/float_dispatch/matmul.rs` | 1, 2 | Fused dequant kernel + remainder micro-kernels |
| `hologram-exec/src/float_dispatch/cast.rs` | 1 | Reference for Q4_0 block format |
| `hologram-exec/src/lut_gemm/psumbook.rs` | 3 | SIMD dot product methods |
| `hologram-archive/src/format/header.rs` | 4 | New flag constant |
| `hologram-archive/src/writer/holo_writer.rs` | 4 | Page-align tensor writing |
| `hologram-archive/src/weight/index.rs` | 4 | Aligned offset builder |

## Verification

1. Q4_0 fused kernel: compare against `decode_weights` → `matmul_k_outer` for identical output
2. Benchmark: TinyLlama Q4_0 inference speed (target: 30x → <5x vs f32 BLAS)
3. Benchmark: f32 decode tok/s regression check (currently 39.1 tok/s)
4. `cargo test` in hologram-exec — matmul + psumbook tests
5. Archive roundtrip: write page-aligned, reload, verify offsets are 4096-aligned
