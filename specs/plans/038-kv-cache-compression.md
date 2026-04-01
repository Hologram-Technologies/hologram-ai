# Plan 038: KV Cache Compression & Attention-Gated Decode

**Status:** Complete
**Created:** 2026-03-31
**Scope:** hologram-exec (hologram base), hologram-ai, hologram-archive
**Inspiration:** KV cache compression research (ICLR 2026) — attention-gated decode, asymmetric K/V, boundary-layer protection

## Motivation

At long context (8K+), decode throughput is dominated by the `weights @ V` accumulation
in the attention kernel — every cached V position is dequantized and accumulated regardless
of its attention weight. Research has demonstrated that 90%+ of softmax positions at 32K
context contribute below numerical significance (< 1e-6) and can be skipped entirely,
yielding +22.8% decode speedup with zero measurable perplexity change.

Additionally, the hologram base crate already implements asymmetric KV quantization with
boundary-layer protection and Walsh-Hadamard rotation, but hologram-ai never wires any of
it — every callsite uses `KvCacheState::new()` which defaults to all-F32.

## What Already Exists in hologram base (verified)

**KV cache quantization** (`hologram-exec/src/kv_cache.rs`):
- `KvCacheConfig` — `k_bits: KvBits`, `v_bits: KvBits`, `boundary_layers: usize`, `wht_rotation: bool`
- `KvBits` enum — `F32`, `Q8`, `Q4`
- `KvCacheConfig::asymmetric_q4()` — K at F32, V at Q4 with WHT and 2 boundary layers
- `is_boundary_layer(layer, n_layers)` — first/last `boundary_layers` layers stay F32
- `effective_k_bits(layer, n_layers)` / `effective_v_bits(layer, n_layers)` — per-layer dtype
- `LayerBuffer` enum — `F32(Vec<f32>)`, `Q8 { indices, params, heads }`, `Q4 { packed, params, heads, head_dim }`, `Empty`
- `quantize_channel_q8/q4` — per-head-per-position affine quantization (quantize-on-write)
- `dequantize_q8/q4` — dequantize-on-read
- Walsh-Hadamard rotation — `wht_rotate`/`wht_unrotate` applied to V only, skipped for boundary layers
- `KvCacheState::with_config(n_layers, n_kv_heads, head_dim, max_seq, config)` — full config constructor
- Lazy allocation — buffers allocated on first write per layer with correct format

**Attention kernel** (`hologram-exec/src/float_dispatch/attention.rs`):
- Online softmax (Flash Attention-style) — single-pass QK^T → softmax → V accumulation
- BLAS path (Accelerate on macOS) — separate scores matrix + softmax + sgemm
- GQA support via `group_size = num_q_heads / num_kv_heads`
- **No sparse V support** — all positions accumulated unconditionally

**What hologram-ai does today** (NOT wiring KvCacheConfig):
- `run_cmd.rs:322` — `KvCacheState::new(n_layers, n_kv_heads, head_dim, max_seq)`
- `compiler.rs:2015` — `KvCacheState::new(...)` in verification
- All conformance tests — `KvCacheState::new(...)` (default = all F32)
- No CLI flags for KV quantization
- `ModelMetaSection` has no KV quantization fields

## Phase 1: Sparse V Decode (NEW — attention-gated V accumulation)

**Goal:** Skip V accumulation for positions with negligible attention weight. Format-agnostic
optimization that works with f32, Q8, and Q4 KV cache.

**File:** `hologram-exec/src/float_dispatch/attention.rs` (hologram base)

**Algorithm — online softmax path (non-BLAS):**

The current inner loop (lines 197–225) computes `w = exp(score - row_max)` then
unconditionally accumulates `o_row[d] += w * v_row[d]`. After the normalization
step (`o_row *= 1/row_sum`), positions where `w/row_sum < τ` contribute less than
τ × max(|V|) to the output — below f32 numerical significance.

We cannot skip *before* normalization because `w` is unnormalized. Instead, we check
`w < τ * row_sum` which is equivalent to `w/row_sum < τ` but avoids division.
However, `row_sum` is a running total and changes as we process more positions.

**Practical approach:** Skip when `w` (unnormalized) is extremely small relative to
the running sum. The check `w < SPARSE_V_THRESHOLD` where `SPARSE_V_THRESHOLD = 1e-6`
works because if `w < 1e-6` and `row_sum >= 1.0` (which it always is after the first
contributing position), then `w/row_sum < 1e-6`.

```rust
let w = (score - row_max).exp();
if w < SPARSE_V_THRESHOLD {
    row_sum += w;  // still count in denominator for numerical stability
    continue;      // skip V accumulation
}
row_sum += w;
let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
for (o, &v) in o_row.iter_mut().zip(v_row.iter()) {
    *o += w * v;
}
```

**BLAS path:** After softmax normalization, zero out weights below threshold before
the `sgemm` call. This converts them to exact zeros, and BLAS will skip the multiply-add
for sparse inputs (though BLAS doesn't guarantee this — the real win is numerical cleanliness).
Alternatively, use a sparse V loop instead of BLAS for the `scores × V_head` step.

**Implementation:**
- [ ] Add `const SPARSE_V_THRESHOLD: f32 = 1e-6` to `attention.rs`
- [ ] Online softmax path: add `if w < SPARSE_V_THRESHOLD { row_sum += w; continue; }`
  before V accumulation (line ~221)
- [ ] BLAS path: after softmax normalization, zero out weights < threshold (line ~257)
- [ ] Add `sparse_v: bool` field to `FloatOp::Attention` (default `true`)
- [ ] Thread `sparse_v` through `dispatch_attention` parameters

**Tests** (in `hologram-exec/src/float_dispatch/attention.rs` or `tests/`):
- [ ] `attention_sparse_v_zero_quality_loss` — compare attention output with sparse_v
  on/off at seq_k=512, 2048, 8192 (random Q/K/V, causal mask). Assert max element-wise
  difference < 1e-5 (threshold is conservative enough for exact match).
- [ ] `attention_sparse_v_skip_rate_scales_with_context` — use synthetic Q/K where one
  query position has high dot product with one key and negligible with all others (e.g.,
  orthogonal keys). Verify that at seq=64 the skip rate is lower than at seq=4096.
  Measure by counting how many `w < SPARSE_V_THRESHOLD` in a custom test kernel.
- [ ] `attention_sparse_v_uniform_weights_no_skip` — Q = K (identity-like) so all
  attention weights are equal (~1/seq). At seq=64, each weight is ~0.016 >> 1e-6,
  so nothing should be skipped. Verify output matches non-sparse path exactly.
- [ ] `attention_sparse_v_single_dominant_position` — one K row identical to Q row,
  all others orthogonal. Output should be dominated by that V row. Verify sparse and
  non-sparse paths produce identical output (bit-exact after rounding).
- [ ] `attention_sparse_v_threshold_boundary` — construct Q/K so that `w` for one
  position is exactly `1.5e-6` (above threshold, NOT skipped) and another is `0.5e-6`
  (below threshold, skipped). Verify the first contributes to output and the second doesn't.
- [ ] `attention_sparse_v_gqa_compatibility` — test with num_q_heads=32, num_kv_heads=4
  (group_size=8). Verify sparse V works correctly when multiple Q heads share K/V heads.
- [ ] `attention_sparse_v_causal_mask_no_regression` — causal mask at seq_q=1, seq_k=1024
  (decode scenario). Verify output matches non-sparse path. Masked positions already have
  weight=0 from `-inf` masking, so sparse V should not change behavior for those.

## Phase 2: Wire KV Cache Config through hologram-ai

**Goal:** Expose hologram base's existing `KvCacheConfig` via CLI flags and archive metadata.
This is pure plumbing — no new quantization logic needed.

**Files:**
- `hologram-ai/src/commands/run_cmd.rs` — CLI flag + KV state construction
- `hologram-archive/src/section/model_meta.rs` — add KV config fields (hologram base)
- `hologram-ai/src/compiler.rs` — thread config into archive metadata

### 2a: CLI flags in `run_cmd.rs`

**Implementation:**
- [ ] Add `--kv-cache` CLI arg: `f32` (default), `q8`, `q4`, `q8:q4` (asymmetric K:V)
- [ ] Add `--kv-boundary-layers` CLI arg: `u32` (default 2)
- [ ] Add `--kv-wht` CLI flag: enable Walsh-Hadamard rotation for V (default off)
- [ ] Parse into `KvCacheConfig` and pass to `KvCacheState::with_config()`
- [ ] Replace `KvCacheState::new(...)` with `KvCacheState::with_config(...)` at line 322

**Tests:**
- [ ] `cli_kv_cache_flag_parsing` — verify `--kv-cache q8:q4` parses to
  `KvBits::Q8` for K, `KvBits::Q4` for V
- [ ] `cli_kv_cache_default_is_f32` — no `--kv-cache` flag defaults to F32/F32
- [ ] `cli_kv_boundary_layers_default` — no flag defaults to 2
- [ ] `cli_kv_wht_flag` — `--kv-wht` sets `wht_rotation: true`

### 2b: ModelMetaSection KV fields

**Implementation:**
- [ ] Add `kv_k_bits: u8` field to `ModelMetaSection` (0=F32, 1=Q8, 2=Q4)
- [ ] Add `kv_v_bits: u8` field to `ModelMetaSection`
- [ ] Add `kv_boundary_layers: u8` field to `ModelMetaSection` (default 2)
- [ ] Add `kv_wht: bool` field to `ModelMetaSection` (default false)
- [ ] Backward compat: deserialize missing fields as defaults (F32/F32, boundary=2, wht=false)

**Tests:**
- [ ] `model_meta_kv_roundtrip` — serialize ModelMetaSection with `kv_k_bits=1, kv_v_bits=2,
  kv_boundary_layers=2, kv_wht=true`, deserialize, verify all fields match
- [ ] `model_meta_kv_defaults` — serialize without KV fields, verify deserialization
  produces F32/F32 defaults

### 2c: Compiler threading

**Implementation:**
- [ ] Add `kv_config: Option<KvCacheConfig>` to `ModelCompiler`
- [ ] Thread through to `ModelMetaSection` builder
- [ ] `run_cmd.rs` reads KV fields from `ModelMetaSection` at load time and constructs
  `KvCacheConfig` accordingly (archive-embedded config overrides CLI defaults)
- [ ] Update `compiler.rs:2015` verification path to use config if set

**Tests:**
- [ ] `compiler_kv_config_in_metadata` — compile with `kv_config = Some(KvCacheConfig::asymmetric_q4())`,
  verify archive's `ModelMetaSection` contains `kv_k_bits=0, kv_v_bits=2, kv_wht=true`
- [ ] `compiler_kv_config_default_none` — compile without kv_config, verify metadata
  has F32/F32 defaults

## Phase 3: End-to-End Validation

**Goal:** Verify the full pipeline works with quantized KV cache on real models.

**Tests:**
- [ ] `e2e_tinyllama_kv_q8_symmetric` — compile TinyLlama, run with `--kv-cache q8`.
  Generate 20 tokens. Verify coherent English output. Compare top-5 token overlap
  with f32 baseline (expect >= 4/5 overlap for first 10 tokens).
- [ ] `e2e_tinyllama_kv_asymmetric_q8_q4` — run with `--kv-cache q8:q4`. Generate
  20 tokens. Verify coherent output. Expect quality between symmetric q8 and symmetric q4.
- [ ] `e2e_tinyllama_kv_q4_with_boundary` — run with `--kv-cache q4 --kv-boundary-layers 2`.
  Verify first/last 2 layers use F32 (inspect via `kv.config().is_boundary_layer()`
  assertion in test). Verify output quality better than uniform q4.
- [ ] `e2e_tinyllama_kv_wht` — run with `--kv-cache q4 --kv-wht`. Verify WHT rotation
  improves quality vs q4 without WHT (compare perplexity or top-k overlap).
- [ ] `e2e_tinyllama_sparse_v` — run with sparse_v on (default) and off. Verify
  identical top-1 token predictions for 50 tokens (greedy decode).
- [ ] `e2e_kv_memory_savings` — log KV cache memory at seq=512 for f32, q8, q4.
  Verify Q8 uses ~26% of F32 (34 bytes per 32 elements + params overhead).
  Verify Q4 uses ~14% of F32 (18 bytes per 32 elements + params overhead).
  (Approximate — overhead from `ChannelParams` per head per position.)
- [ ] `e2e_benchmark_sparse_v_decode` — Criterion benchmark: decode at seq=2048 with
  sparse_v on/off. Log speedup. Not a hard assertion, but expect measurable improvement
  at longer contexts.

## Verification

1. **Unit tests (Phase 1):** 7 attention kernel tests. Run with `cargo test` in hologram base.
2. **Integration tests (Phase 2):** 8 tests across CLI, metadata, and compiler. Run with
   `cargo test` in hologram-ai.
3. **E2E tests (Phase 3):** 7 tests on TinyLlama. Run with `cargo test --test` in hologram-ai.
4. **Performance:** Criterion benchmarks in `benches/inference.rs` for decode throughput
   and memory usage comparisons.

## Critical files to modify

| File | Repo | Change |
|------|------|--------|
| `hologram-exec/src/float_dispatch/attention.rs` | hologram base | Sparse V threshold + skip logic |
| `hologram-core/src/op/float_op.rs` | hologram base | Add `sparse_v: bool` to `FloatOp::Attention` |
| `hologram-archive/src/section/model_meta.rs` | hologram base | Add KV config fields |
| `hologram-ai/src/commands/run_cmd.rs` | hologram-ai | CLI flags + config wiring |
| `hologram-ai/src/compiler.rs` | hologram-ai | Thread KV config to metadata |

## Non-Goals (future work)

- **PolarQuant / centroid-based quantization** — sub-4-bit KV formats. Current Q4 uses
  affine quantization; PolarQuant uses 16 learned centroids. Deferred.
- **Paged KV cache** (Plan 016) — orthogonal to compression. Can combine later.
- **Temporal decay** — compress older KV entries more aggressively. Experimental.
- **GPU kernels** — sparse V for Metal/CUDA attention kernels. Deferred to GPU sprint.
- **Block size optimization** — current Q8/Q4 use per-head-per-position granularity
  (not block-based like llama.cpp). research finding about block_size=128 alignment
  with WHT scope applies if we switch to block-based quantization.
