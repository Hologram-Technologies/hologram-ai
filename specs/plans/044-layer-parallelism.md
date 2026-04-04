# Plan 044: Layer-Level Parallelism — 60-80 tok/s

**Status:** Open
**Created:** 2026-04-02
**Branch:** `feat/cpu-inference-perf`
**Baseline:** 40.9 tok/s (TinyLlama, M4 Max, single-threaded)

## Motivation

At 40 tok/s, each decode step takes 22ms. Within each transformer layer, several
operations are independent and can run in parallel:
- Q, K, V projections (3 independent MatMuls)
- Gate and Up projections (2 independent MatMuls in SwiGLU FFN)
- Attention heads (32 independent dot-product attention computations)

Parallelizing these on M4 Max (12 performance cores) can reduce per-layer
time by ~2x, giving ~60-80 tok/s.

## Scope

Layer-level parallelism, NOT instruction-level. The old `execute_parallel` was
removed because it added per-instruction overhead. Instead, we parallelize
WITHIN specific kernel calls using rayon.

## Implementation

### 1. Parallel Attention Heads

**Where:** `hologram-exec/src/float_dispatch/attention.rs` — `dispatch_attention`

Currently: sequential loop over `num_q_heads` (32 for TinyLlama).
Each head computes `Q_h × K_h^T → softmax → scores × V_h` independently.

```rust
// Before: sequential
for &(q_off, k_off, o_off) in &head_offsets {
    // compute attention for one head
}

// After: parallel when num_q_heads >= 4
head_offsets.par_iter().for_each(|&(q_off, k_off, o_off)| {
    // compute attention for one head (writes to disjoint output region)
});
```

**Safety:** Each head writes to a disjoint region of the output buffer.
Use `par_chunks_mut` on the output or pre-split into per-head slices.

**Expected speedup:** 32 heads on 12 cores → ~3-4x for the attention kernel.
Attention is ~30% of total step time → ~1.3-1.5x overall.

### 2. Parallel QKV Projections

**Where:** Tape executor level — identify Q, K, V MatMul instructions in the
same level and dispatch them in parallel.

Currently: Q, K, V projections execute sequentially (3 × 0.09ms = 0.27ms per layer).
With parallelism: 0.09ms (latency of one projection).

**Implementation:** Add a `ParallelGroup` concept to the tape:
- At tape build time, identify independent MatMuls in the same level
- At execution time, dispatch them with rayon::join or par_iter

**Expected speedup:** Small per-layer (0.18ms saved per layer × 22 layers = 4ms).

### 3. Parallel FFN Gate + Up

**Where:** Same as QKV — gate and up projections are independent MatMuls.

Currently: gate + up execute sequentially (2 × 0.49ms = 0.98ms per layer).
With parallelism: 0.49ms (latency of one projection).

**Expected speedup:** 0.49ms saved per layer × 22 layers = 10.8ms → ~1.5x.

## Combined Projection

| Optimization | Per-layer savings | Total savings (22 layers) | New tok/s |
|-------------|------------------|--------------------------|-----------|
| Parallel attention heads | ~2ms | 44ms → ~15ms | ~55 |
| Parallel QKV | 0.18ms | ~4ms | ~45 |
| Parallel gate+up | 0.49ms | ~11ms | ~50 |
| All combined | ~2.7ms | ~22ms → ~8ms | **~75** |

## Files

- `hologram-exec/src/float_dispatch/attention.rs` — parallel head loop
- `hologram-exec/src/tape.rs` — `execute_direct` parallel group dispatch
- Feature-gated behind `parallel` (rayon)

## Testing

- Correctness: parallel output matches sequential (bit-for-bit for deterministic ops)
- Performance: measure per-step time reduction
- Thread safety: no data races (each parallel unit writes to disjoint output region)
