# Plan 051: Full Byte-Domain Q4 MatMul — 60+ tok/s

## Problem

The current NEON NibbleView Q4 kernel (`lut_gemm_4bit_neon_nibbleview`) achieves 12.5 tok/s
pure-LUT because of a per-K-row widening pipeline bottleneck:

```
Per K-row, per 32 output columns (current):
  2 ops: vqtbl1q_s8 × 2 (table lookup — the fast part)
 32 ops: vmovl_s8 → vmovl_s16 → vcvtq_f32_s32 → vmulq_f32(a_val*dequant)
         → vzip1/vzip2 → vld1q/vaddq/vst1q × 4 groups
```

The lookup is 4 ops. The int8→f32 widening + scale + interleave + accumulate is 32 ops.
That's 8:1 overhead-to-work ratio. The f32 conversion happens **per K-row** because each
row has a different activation value `a_val`, so we can't accumulate int8 across rows.

## Solution: Quantize Activations → Pure Integer Accumulation

If we quantize the activation vector to int8 too, the inner loop becomes integer-only:

```
Per K-row, per 32 output columns (proposed):
  2 ops: vqtbl1q_s8 × 2 (centroid lookup — same as before)
  4 ops: vmlal_s8 × 4 (widening multiply-accumulate: int8 × int8 → int16)
```

That's 6 ops per 32 columns vs 36 ops — a **6x reduction** in inner-loop work.

The key insight: `a_val_i8 × centroid_i8` produces int16, and int16 values accumulate
across K rows in int32. The f32 conversion happens ONCE at the end per output element.

## Detailed Design

### Phase 1: Activation Quantization (per-row, ~10 NEON ops)

Before the K-loop, quantize the entire activation row to int8:

```rust
// Per activation row (done ONCE, cost amortized over N output columns):
let a_max = a_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
let a_scale = a_max / 127.0;          // dequant factor
let a_inv_scale = 127.0 / a_max;      // quant factor

let mut a_i8 = [0i8; K];
for l in 0..k {
    a_i8[l] = (a_row[l] * a_inv_scale).round() as i8;
}
```

For TinyLlama K=2048, this is 2048 multiply+round+cast ops — negligible vs the K×N
inner loop (2048 × 2048 = 4M iterations for a single FFN matmul).

NEON vectorization of quantization: `vcvtq_f32_s32` + `vmulq_f32` + `vcvtnq_s32_f32` +
narrow `vqmovn_s32` → `vqmovn_s16` processes 16 elements per iteration. K=2048 ÷ 16 = 128
iterations. Trivial.

### Phase 2: Integer Inner Loop (NEON vmlal_s8)

The core loop processes 32 output columns per iteration with 6 NEON ops:

```rust
// Accumulators: int32, one per output column (initialized to 0)
// For 32 columns: 8 × int32x4 registers
let mut acc = [[vdupq_n_s32(0); 4]; 2]; // [hi/lo][4 groups of 4]

for l in 0..k {
    let a_val = vdup_n_s8(a_i8[l]);  // broadcast activation (1 op)

    // Load 16 packed bytes = 32 Q4 indices
    let packed = vld1q_u8(idx_ptr);   // 1 op
    let lo_idx = vandq_u8(packed, mask_lo);
    let hi_idx = vshrq_n_u8(packed, 4);

    // Table lookup: 4-bit index → int8 centroid
    let lo_i8 = vqtbl1q_s8(tbl, lo_idx);  // 1 op, 16 lookups
    let hi_i8 = vqtbl1q_s8(tbl, hi_idx);  // 1 op, 16 lookups

    // Widening multiply-accumulate: int8 × int8 → int16, accumulate in int16
    // vmlal_s8: Vr[i] += Va[i] * Vb[i], with int8→int16 widening
    // Then widen int16 → int32 for safe accumulation
    //
    // lo path (even columns):
    acc_lo_0 = vmlal_s8(acc_lo_0, vget_low_s8(lo_i8), a_val);   // 8 columns
    acc_lo_1 = vmlal_s8(acc_lo_1, vget_high_s8(lo_i8), a_val);  // 8 columns
    // hi path (odd columns):
    acc_hi_0 = vmlal_s8(acc_hi_0, vget_low_s8(hi_i8), a_val);
    acc_hi_1 = vmlal_s8(acc_hi_1, vget_high_s8(hi_i8), a_val);
}
```

**Overflow analysis:** `vmlal_s8` produces int16 (range ±16384). Each product is
`a_i8 × centroid_i8` ∈ [-127×127, +127×127] = [-16129, +16129]. Accumulating across
K rows in int16 would overflow after just 2 rows. So we need int32 accumulation.

**Revised approach — `vmlal_s8` accumulates int8→int16, flush to int32 periodically:**

Actually, NEON `vmlal_s8` accumulates into int16. We'd overflow after K=2 rows.
Instead, use two-stage accumulation:

```
Option A: smlal (int16 += int8 × int8) — flush int16→int32 every 1 row
Option B: Direct int32 accumulation via vmull_s8 + vaddw_s16
Option C: vdot_s32 (int32 += int8 × int8, 4 products at once) — ARMv8.2-A+
```

### Phase 2 (Revised): Two viable NEON strategies

#### Strategy A: vmull + vaddw (works on ALL AArch64)

```rust
for l in 0..k {
    let a_val = vdup_n_s8(a_i8[l]);
    let packed = vld1q_u8(idx_ptr.add(l * n_bytes + chunk * 16));
    let lo_idx = vandq_u8(packed, mask_lo);
    let hi_idx = vshrq_n_u8(packed, 4);
    let lo_i8 = vqtbl1q_s8(tbl, lo_idx);
    let hi_i8 = vqtbl1q_s8(tbl, hi_idx);

    // vmull_s8: int8 × int8 → int16 (8 products)
    // vaddw_s16: int32 += int16 (widen and accumulate)
    let lo_prod_low = vmull_s8(vget_low_s8(lo_i8), a_val);   // 8 × int16
    let lo_prod_high = vmull_s8(vget_high_s8(lo_i8), a_val);  // 8 × int16
    let hi_prod_low = vmull_s8(vget_low_s8(hi_i8), a_val);
    let hi_prod_high = vmull_s8(vget_high_s8(hi_i8), a_val);

    // Widen int16 → int32 and accumulate
    acc[0] = vaddw_s16(acc[0], vget_low_s16(lo_prod_low));    // 4 × int32
    acc[1] = vaddw_s16(acc[1], vget_high_s16(lo_prod_low));   // 4 × int32
    acc[2] = vaddw_s16(acc[2], vget_low_s16(lo_prod_high));
    acc[3] = vaddw_s16(acc[3], vget_high_s16(lo_prod_high));
    acc[4] = vaddw_s16(acc[4], vget_low_s16(hi_prod_low));
    acc[5] = vaddw_s16(acc[5], vget_high_s16(hi_prod_low));
    acc[6] = vaddw_s16(acc[6], vget_low_s16(hi_prod_high));
    acc[7] = vaddw_s16(acc[7], vget_high_s16(hi_prod_high));
}
```

**Op count per K-row per 32 columns:**
- 1 vdup_n_s8 (broadcast)
- 1 vld1q_u8 (load packed indices)
- 2 vandq/vshrq (extract nibbles)
- 2 vqtbl1q_s8 (table lookup)
- 4 vmull_s8 (widening multiply)
- 8 vaddw_s16 (widen-accumulate into int32)
- **Total: 18 ops per 32 columns**

vs current 36 ops — **2x reduction**.

But we can do better. The hi/lo interleaving doubles the accumulator count.
If we deinterleave at the end instead of per-row, we save the vaddw calls.

#### Strategy B: Accumulate int16, flush every 128 rows (sweet spot)

Each int16 accumulator can hold 128 × 127 = 16256 < 32767 before overflow.
Flush to int32 every 128 K-rows:

```rust
let flush_interval = 128;
let mut acc_i32 = [vdupq_n_s32(0); 8]; // 32 columns, int32

for l_base in (0..k).step_by(flush_interval) {
    let l_end = (l_base + flush_interval).min(k);
    let mut acc_i16 = [vdupq_n_s16(0); 4]; // 32 columns as 4×int16x8

    for l in l_base..l_end {
        let a_val = vdup_n_s8(a_i8[l]);
        // ... load + lookup as before ...

        // vmlal_s8: int16 += int8 × int8 (8 products, accumulated)
        acc_i16[0] = vmlal_s8(acc_i16[0], vget_low_s8(lo_i8), a_val);
        acc_i16[1] = vmlal_s8(acc_i16[1], vget_high_s8(lo_i8), a_val);
        acc_i16[2] = vmlal_s8(acc_i16[2], vget_low_s8(hi_i8), a_val);
        acc_i16[3] = vmlal_s8(acc_i16[3], vget_high_s8(hi_i8), a_val);
    }

    // Flush int16 → int32 (once per 128 rows)
    acc_i32[0] = vaddw_s16(acc_i32[0], vget_low_s16(acc_i16[0]));
    acc_i32[1] = vaddw_s16(acc_i32[1], vget_high_s16(acc_i16[0]));
    // ... (8 total vaddw per flush)
}
```

**Op count per K-row per 32 columns (Strategy B):**
- 1 vdup_n_s8
- 1 vld1q_u8
- 2 vandq/vshrq
- 2 vqtbl1q_s8
- 4 vmlal_s8 (accumulate directly in int16!)
- **Total: 10 ops per 32 columns** + 8 vaddw every 128 rows (amortized: 0.06 ops/row)

**vs current 36 ops → 3.6x reduction in inner loop work.**

### Phase 3: Final Conversion (ONCE per output element)

After the K-loop, convert int32 accumulators to f32 and apply combined scale:

```rust
let combined_scale = a_scale * centroid_dequant; // f32
let v_scale = vdupq_n_f32(combined_scale);

// Convert int32 → f32 and scale (32 columns = 8 × f32x4)
// Also interleave hi/lo to correct column order
for g in 0..4 {
    let hi_f32 = vmulq_f32(vcvtq_f32_s32(acc_i32[g + 4]), v_scale);
    let lo_f32 = vmulq_f32(vcvtq_f32_s32(acc_i32[g]), v_scale);
    let zip1 = vzip1q_f32(hi_f32, lo_f32);
    let zip2 = vzip2q_f32(hi_f32, lo_f32);
    vst1q_f32(out_ptr.add(col_base + g * 8), zip1);
    vst1q_f32(out_ptr.add(col_base + g * 8 + 4), zip2);
}
```

This is 32 ops per 32 columns — but it runs ONCE per output element, not per K-row.
For K=2048, this is amortized to 32/2048 = 0.016 ops per K-row per column. Negligible.

## Theoretical Performance

### Op count comparison (per K-row, per 32 output columns):

| Kernel | Ops/row/32col | Relative |
|--------|---------------|----------|
| Current NEON (f32 per row) | 36 | 1.0x |
| Strategy A (vmull+vaddw) | 18 | 2.0x |
| **Strategy B (vmlal_s8 + flush-128)** | **10** | **3.6x** |
| Theoretical min (load+lookup) | 6 | 6.0x |

### Throughput projection:

Current pure-LUT: 12.5 tok/s → inner loop is the bottleneck.
Current hybrid (BLAS for N≥1024): 37 tok/s → BLAS handles the heavy matmuls.

With Strategy B:
- Inner loop 3.6x faster → pure LUT goes from 12.5 to ~45 tok/s
- BUT: BLAS path (AMX) for N≥1024 matmuls is ~37 tok/s
- Strategy B pure-LUT at 45 tok/s BEATS BLAS for some matmul sizes
- **Combined (best of both):** ~50-55 tok/s

Additional wins stacking on top:
- Compile-time centroid table is already done (no change needed)
- NEON activation quantization vectorized: ~0.5% overhead
- Reduced memory bandwidth: int8 activations read once, not f32
- **Projected total: 55-65 tok/s**

### Why this beats BLAS for decode (M=1):

BLAS (sgemm via Accelerate/AMX) is optimized for large GEMM (M≥32). For M=1 (decode),
it's a vecmat — BLAS overhead (function call, parameter validation, tile setup) is
significant relative to the actual computation. Our LUT kernel:

1. Zero overhead: inline, no function call
2. Reads 0.5 bytes per weight (Q4) vs 4 bytes (f32 dequant for BLAS)
3. 8x less memory bandwidth
4. Pure integer inner loop — no FP pipeline stalls

For N=2048, K=2048, M=1: ~2M weight bytes (Q4) vs ~16M (f32). At 200 GB/s memory
bandwidth (Apple M-series), that's 0.01ms vs 0.08ms. The compute is ~2M × 10 ops =
20M ops, at ~8 GOPS (NEON int8) = 2.5ms. Memory-bound, not compute-bound — the 8x
bandwidth advantage is the real win.

## Implementation Plan

### Step 1: Activation quantization helper (hologram-exec)

New function in `lut_gemm/matmul.rs`:

```rust
/// Quantize f32 activation row to int8, return (int8 buffer, scale factor).
/// The scale factor converts int8 back to f32: f32_val ≈ int8_val * scale.
#[inline]
fn quantize_activation_row(a_row: &[f32], a_i8: &mut [i8]) -> f32 {
    let a_max = a_row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if a_max < 1e-12 { a_i8.fill(0); return 0.0; }
    let inv_scale = 127.0 / a_max;
    for (i, &v) in a_row.iter().enumerate() {
        a_i8[i] = (v * inv_scale).round() as i8;
    }
    a_max / 127.0
}
```

NEON-vectorized version for aarch64 (processes 16 elements per iteration).

### Step 2: New kernel function (hologram-exec)

```rust
fn lut_gemm_4bit_neon_int8(
    activations: &[f32],
    weights: &QuantizedWeights4,
    output: &mut [f32],
    m: usize, k: usize, n: usize,
    centroids: &[f32; 16],
)
```

This replaces `lut_gemm_4bit_neon_nibbleview`. Single kernel, no dispatch.

### Step 3: Remove hybrid BLAS dispatch (hologram-exec tape.rs)

Once the int8 kernel beats BLAS for M=1 vecmat, remove the `n >= 1024` BLAS branch
in `dispatch_lut_gemm_4`. Single path: always use the LUT kernel for Q4 matmuls.

### Step 4: Scalar fallback update

Update `lut_gemm_4bit_scalar_premul` to also use int8 activation quantization for
consistency. The scalar path benefits similarly (fewer f32 multiplies per row).

### Step 5: WASM path

The same algorithm works on WASM with `i8x16_swizzle` (already in hologram-core simd.rs)
replacing `vqtbl1q_s8`. No WASM-specific changes needed — the scalar fallback handles it.

## Quantization Error Analysis

Double quantization (weights Q4 + activations int8) introduces additional error:

- Weight quantization: 16 centroids, ~12.6% relative error (measured)
- Activation quantization: int8 per-row symmetric, ~0.4% relative error
- Combined: ~13% relative error (activation quant error is negligible)

The activation quantization adds minimal error because:
1. Per-row scale maximizes dynamic range utilization
2. 127 levels for activations vs 16 for weights — activations are much better quantized
3. The dominant error source remains the 16-centroid weight quantization

## Overflow Safety

**int16 accumulation (Strategy B):**
- Each product: `a_i8 × centroid_i8` ∈ [-16129, +16129]
- After N rows: sum ∈ [-16129×N, +16129×N]
- int16 range: [-32768, +32767]
- Safe for N ≤ 2 rows (32767/16129 = 2.03)
- **Flush interval: 2 rows** — too frequent, negates the benefit

**Revised: Use int32 accumulation throughout (Strategy A)**

Actually, the flush-128 analysis was wrong. Let me recalculate:
- If `a_i8` ∈ [-127, +127] and `centroid_i8` ∈ [-127, +127]
- Product ∈ [-16129, +16129]
- After 2 rows, sum ∈ [-32258, +32258] — exceeds int16 range!

**Conclusion: Strategy B (int16 accumulation) requires flush every 2 rows — not viable.**

**Strategy A (vmull → vaddw into int32) is the correct approach.**

### Revised op count (Strategy A):

Per K-row per 32 columns:
- 1 vdup_n_s8 (broadcast activation)
- 1 vld1q_u8 (load packed indices)
- 2 vandq/vshrq (nibble extract)
- 2 vqtbl1q_s8 (table lookup)
- 4 vmull_s8 (int8 × int8 → int16, 8 products each)
- 8 vaddw_s16 (int16 → int32 accumulate)
- **Total: 18 ops per 32 columns**

vs current 36 → **2x reduction**. This gets us from 12.5 tok/s pure-LUT to ~25 tok/s.

### Can we do better? Unrolled K-loop with register reuse

Unroll the K-loop by 4, reusing the accumulator registers:

```rust
for l in (0..k).step_by(4) {
    // Load 4 activation values at once
    let a0 = vdup_n_s8(a_i8[l]);
    let a1 = vdup_n_s8(a_i8[l+1]);
    let a2 = vdup_n_s8(a_i8[l+2]);
    let a3 = vdup_n_s8(a_i8[l+3]);

    // Process 4 rows × 32 columns with interleaved loads
    // (pipelining hides load latency)
    for each of 4 rows: load + nibble + lookup + vmull + vaddw
}
```

K-unroll doesn't reduce op count but improves instruction-level parallelism (ILP)
by hiding load latency. Expected: 1.2-1.5x additional speedup.

**With K-unroll: 25 × 1.3 ≈ 32 tok/s pure-LUT.**

### N-chunking: process 64 columns per iteration

Double the chunk size to 64 columns (32 bytes of packed indices):

```rust
for chunk in 0..n_bytes/32 {
    let packed0 = vld1q_u8(ptr);        // columns 0-31
    let packed1 = vld1q_u8(ptr + 16);   // columns 32-63
    // ... 4 lookups, 8 vmull, 16 vaddw
    // But: 32 NEON registers available, 16 used for accumulators = fits
}
```

This improves load amortization (activation broadcast reused across 64 columns).
Expected: 1.1x additional.

### Combined projection (pure LUT):

| Optimization | tok/s | Cumulative |
|-------------|-------|------------|
| Current (f32 per row) | 12.5 | 1.0x |
| Strategy A (int8 act, vmull+vaddw) | 25 | 2.0x |
| + K-unroll by 4 | 32 | 2.6x |
| + N-chunk 64 | 35 | 2.8x |

### Hybrid strategy: beat BLAS threshold

If pure-LUT reaches 35 tok/s, and BLAS gives 37 tok/s — they're neck-and-neck.
The LUT kernel reads 8x less memory. For decode (M=1), it should win on
bandwidth-limited matmuls. We can:

1. Lower the BLAS threshold from N≥1024 to N≥4096 (or remove entirely)
2. Profile both paths on representative matmul sizes
3. Pick the winner per-size at compile time (baked into the tape)

**Realistic target: 45-55 tok/s** by using pure-LUT for all Q4 matmuls where it
beats BLAS, and BLAS only for the largest matmuls where AMX truly dominates.

## Compile-Time Optimizations (maximize what's precomputed)

Everything possible should be done at archive-compile time, not inference time:

1. **Centroid int8 table** — already done (compile-time in `lut_gemm_4bit`)
2. **Weight indices** — already packed Q4 in archive (zero-copy mmap)
3. **Centroid dequant scale** — precompute and store in `QuantizedWeights4`
4. **Layout** — weights already row-major packed, optimal for streaming

At inference time, the ONLY dynamic computation is:
- Activation quantization (per-row, ~128 NEON ops for K=2048)
- The int8 inner loop
- Final int32→f32 conversion + scale

## Files Changed

### hologram base (hologram-exec)
| File | Change |
|------|--------|
| `lut_gemm/matmul.rs` | New `lut_gemm_4bit_neon_int8` kernel, `quantize_activation_row` helper |
| `lut_gemm/matmul.rs` | Remove `lut_gemm_4bit_neon_nibbleview` (replaced) |
| `lut_gemm/matmul.rs` | Update scalar fallback with int8 activation path |
| `lut_gemm/quantize.rs` | Add `centroid_dequant: f32` to `QuantizedWeights4` (compile-time) |
| `tape.rs` | Profile and potentially remove BLAS branch in `dispatch_lut_gemm_4` |

### hologram-ai
| File | Change |
|------|--------|
| (none initially) | Recompile archives to include `centroid_dequant` field |

## Testing

1. **Correctness:** `lut_gemm_q4_vs_naive_*` tests must pass (same tolerance)
2. **Overflow:** Test with K=4096 (max expected) to verify int32 doesn't overflow
3. **Edge cases:** K not divisible by 4 (unroll remainder), N not divisible by 32 (chunk remainder)
4. **End-to-end:** TinyLlama "capital of France" produces coherent English
5. **Benchmark:** `cargo bench` matmul M=1 K=2048 N=2048 — measure tok/s improvement

## Risk Assessment

- **Low risk:** Algorithm is straightforward (well-known int8 quantization technique)
- **Low risk:** NEON intrinsics are stable, well-documented
- **Medium risk:** Achieving >2x speedup depends on memory bandwidth being the bottleneck,
  not instruction throughput. If NEON int8 pipeline is the bottleneck, speedup may be <2x.
- **Mitigation:** Profile with `instruments` to verify bandwidth-bound vs compute-bound

## Decision: Strategy A (vmull + vaddw into int32)

Strategy B (int16 accumulation) is not viable — overflow after 2 rows.
Strategy A gives 2x inner-loop reduction with K-unroll for ILP.

The plan is:
1. Implement Strategy A kernel
2. Add K-unroll by 4
3. Profile vs BLAS
4. Lower/remove BLAS threshold based on measurements
5. Target: 45-55 tok/s combined
