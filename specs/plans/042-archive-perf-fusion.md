# Plan 042: Path to 100-200 tok/s — Archive + Fusion + Kernels + Variable-Length + Speculative + Compression

**Status:** Open
**Created:** 2026-04-01
**Branch:** `feat/cpu-inference-perf` in both repos
**Target:** 100-200+ effective tok/s for TinyLlama 1.1B ONNX on Apple Silicon

## Current State

- TinyLlama ONNX f32: **2.5 tok/s** (bandwidth-limited, decode graph at seq=1 working)
- TinyLlama ONNX Q4: **1.0 tok/s** (LUT-GEMM psumbook slower than BLAS, no AMX)
- Archive size: **9.4 GB** for Q4 (weights duplicated in sub-archive + shared blob)
- Correctness: ✅ "The capital of France is Paris." (matches ORT, KV cache decode works)
- Known broken: variable-length execution (must compile at exact prompt length)

## Performance Math

```
Current:               2.5 tok/s (f32, BLAS)
× MatMulActivation:    2x → 5 tok/s
× Q4 + AMX hybrid:    8x → 40 tok/s
× Speculative decode:  2.5x → 100 tok/s
= Target:              ~100 tok/s effective
```

---

## Phase 1: Archive Dedup + MatMulActivationFusion (→ ~5 tok/s)

### 1A. Fix Archive Weight Duplication

**Problem:** `compile_components()` embeds weights in the first sub-archive (4.4 GB)
AND inserts them into the shared `WeightStore` blob (4.4 GB more). Total: 2× weights.

**Fix:** Skip `weight_store.insert()` for LLM pipeline components that share the
same `weight_group` as an already-embedded component. The shared blob is only needed
for multi-component models with DIFFERENT weight groups (e.g., Stable Diffusion
text_encoder + unet with separate weights).

**Files:**
- `crates/hologram-ai/src/compiler.rs` — `compile_components()`:
  ```rust
  // Only insert into shared store if weight_group has multiple distinct
  // weight sources. For LLM pipeline (prefill+decode), same weights → skip.
  if n > 1 && is_first_in_group {
      weight_store.insert(&spec.name, &spec.weight_group, w);
  }
  ```
  Also: skip `build_with_shared_weights()` when shared blob is empty.

**Expected:** F32 archive: ~4.4 GB. Q4 archive: ~0.5 GB.

**Tests:**
- Archive size assertion in conformance test
- Verify both prefill and decode sub-archives load and execute correctly

### 1B. MatMulActivationFusion Pass

**Problem:** The optimization pass that fuses `MatMul → Activation` was never
created (the SPRINT says "wired end-to-end" but only the lowering + kernel exist,
not the pass that creates the fused AiOp variants).

**Implementation:**

File: `crates/hologram-ai-common/src/opt/matmul_activation_fusion.rs`

```rust
pub struct MatMulActivationFusion;

impl OptPass for MatMulActivationFusion {
    fn run(&self, mut graph: AiGraph) -> Result<AiGraph> {
        // For each node: if it's a MatMul whose SOLE consumer is
        // SiLU/GeLU/ReLU, replace MatMul with the fused variant
        // and mark the activation node as Passthrough.
        //
        // Pattern: MatMul(A, W) → SiLU(x) becomes MatMulSilu(A, W)
        //
        // Constraints:
        // - MatMul must have exactly ONE consumer
        // - Consumer must be a supported activation (SiLU, GeLU, ReLU)
        // - MatMul must not already be fused
        // - Skip if MatMul feeds into attention (already handled by GQA)
    }
}
```

Wire into pipeline:
- `crates/hologram-ai-common/src/opt/pipeline.rs` — add after SwiGluFusion
- `crates/hologram-ai-common/src/opt/mod.rs` — register module

**Existing infrastructure that already works:**
- `AiOp::MatMulSilu`, `AiOp::MatMulGelu`, `AiOp::MatMulRelu` in `ir/op.rs`
- `wrap_graph_op()` in `strategy.rs` → `GraphOp::FusedMatMulActivation`
- `InlineMatMulActivation` tape kernel in hologram base

**Expected:** ~2x decode speedup (eliminates intermediate buffer, applies activation
in-register during matmul writeback).

**Tests:**
- Unit test: synthetic MatMul → SiLU graph → verify fused to MatMulSilu
- Conformance: TinyLlama before/after fusion produces same top-5 tokens
- Performance: decode step time halved

---

## Phase 2: Q4 AMX/BLAS Hybrid Kernel (→ ~40 tok/s)

### 2A. Dequant-per-tile AMX Hybrid (hologram base)

**Problem:** LUT-GEMM psumbook kernel is scalar — ~0.1 TFLOPS on Apple Silicon vs
~2 TFLOPS with Accelerate AMX. Q4 is slower than f32 because it skips BLAS.

**Fix:** During Q4 matmul dispatch, dequant each KC-blocked tile from Q4 centroids
back to f16, then call `cblas_hgemm` (Accelerate half-precision GEMM → AMX hardware).
This gets BOTH Q4 compression (4× smaller archive, 4× less bandwidth) AND AMX speed.

**Implementation:**

File: `hologram-exec/src/float_dispatch/matmul.rs` (hologram base)

```rust
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn matmul_q4_amx_hybrid(
    q4_weights: &QuantizedWeights4,
    activations: &[f32],
    output: &mut [f32],
    m: usize, k: usize, n: usize,
) {
    // For each KC×NR tile:
    //   1. Dequant Q4 indices → f16 using centroid lookup
    //   2. Convert activations to f16
    //   3. Call cblas_hgemm(m_tile, n_tile, k_tile, ...)
    //   4. Accumulate f32 output
}
```

Also add dispatch routing: when platform has Accelerate AND op is MatMulLut4,
use AMX hybrid instead of psumbook.

**File:** `hologram-exec/src/tape.rs` — `InlineMatMulLut4` dispatch:
```rust
// Before: always psumbook
// After: if cfg!(accelerate) { amx_hybrid } else { psumbook }
```

**Fallback:** Non-macOS platforms continue using psumbook (pure Rust, works on WASM).

**Expected:** Q4 matmul at ~80% of f32 BLAS speed (dequant overhead ~20%).
TinyLlama Q4: ~32-40 tok/s (vs 2.5 f32 / 1.0 Q4 current).

**Tests:**
- Unit test: Q4 AMX hybrid matches psumbook output within tolerance
- Benchmark: Q4 AMX vs f32 BLAS vs psumbook at representative sizes
- E2E: TinyLlama Q4 correct output + tok/s measurement

---

## Phase 3: Variable-Length Execution Fix (→ any prompt length)

### 3A. Fix resolve_size() for Seq-Dependent Ops (hologram base)

**Problem:** When compiled seq_len (e.g., 32) differs from runtime input (e.g., 24),
ops like Reshape/Expand use compiled values while Softmax/RmsNorm infer from buffer
lengths. This inconsistency corrupts shapes and produces garbage.

**Root cause:** `resolve_size()` in `float_dispatch/mod.rs` falls back to `n_floats`
when `n_floats % compiled_size != 0`. But the compiled_size is an axis dimension (e.g.,
hidden_dim=2048), not a total element count. The fallback is usually correct for
hidden-dim-dependent ops but wrong for seq-dependent ops.

**Fix strategy:** Use 0-sentinels for ALL seq-dependent dimensions during lowering.
The executor already resolves 0 → infer from buffer. Currently only some ops use
0-sentinels; extend to Reshape, Expand, and all ops that embed seq_len in parameters.

**Files:**
- `hologram-exec/src/float_dispatch/mod.rs` — improve `resolve_size()` heuristics
- `hologram-ai-common/src/lower/strategy.rs` — emit 0 for seq-dependent dims in
  Reshape, Expand, Slice target shapes
- `hologram-ai-common/src/lower/builder.rs` — track which dims are seq-dependent

**Tests:**
- Compile TinyLlama at seq=64, run with 24-token prompt → correct output
- Compile at seq=128, run with 10 tokens → correct output
- KV cache decode still works (seq=1)

---

## Phase 4: Speculative Decoding (→ 100-200 tok/s)

### 4A. Draft Model + Batched Verification

**Impact:** 2-4x effective throughput — the ONLY way past the memory bandwidth wall.

**How:** Small draft model (Llama 3.2 1B or TinyLlama for testing) generates N
candidate tokens at ~4x speed. Large model verifies all N in one batched forward
pass. 60-70% acceptance rate → 2-3x net speedup.

**Key insight:** Verification of N tokens costs ~same as generating 1 token (weights
read once regardless of batch size).

**Implementation:**

File: `crates/hologram-ai/src/speculative.rs` (new)

```rust
pub struct SpeculativeDecoder {
    target: HoloRunner,      // large model
    draft: HoloRunner,       // small model
    draft_steps: usize,      // candidates per batch (4-8)
}

impl SpeculativeDecoder {
    pub fn generate(&mut self, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
        // 1. Draft: generate N candidate tokens with draft model
        // 2. Build batched input: [prompt + candidates]
        // 3. Verify: run target model on full sequence (1 forward pass)
        // 4. Accept/reject: compare draft vs target distributions
        // 5. Accept first K matching tokens, reject rest, repeat
    }
}
```

CLI: `--draft-model <path>` flag on `hologram-ai run`.

**Prerequisites:**
- Variable-length fix (Phase 3) — verification pass processes variable-length input
- Batched matmul (M > 1 during verification) — already supported
- Draft/target tokenizer compatibility check at load time

**Memory budget:** Target Q4 (~0.5 GB) + draft Q4 (~0.1 GB) = ~0.6 GB total.

**Expected:** 2.5x effective throughput multiplier. With Q4+AMX at 40 tok/s base,
speculative gives **~100 tok/s effective**.

**Tests:**
- Unit test: acceptance/rejection logic matches reference implementation
- Test: speculative output matches greedy decode for deterministic case
- Benchmark: effective tok/s vs non-speculative

---

## Phase 5: Archive Compression

### 5A. Wire hologram-compression

**Implementation:**
- At compile time: optionally compress the archive with `HoloWriter::compress_weights()`
- At load time: `HoloRunner::from_path()` already detects compression via
  `is_compressed()` and decompresses to a cache file for instant mmap on re-runs
- CLI: `--compress` flag on `hologram-ai compile`

**Files:**
- `crates/hologram-ai/src/compiler.rs` — add compression after `build_final_archive()`
- `crates/hologram-ai/src/cli.rs` — `--compress` flag
- (HoloRunner decompression already implemented)

**Expected:** Q4 archive: ~0.5 GB → ~0.2 GB compressed. F32: ~4.1 GB → ~2.5 GB.

**Tests:**
- Compile with --compress, run → correct output
- Second run uses cache (instant load)

---

## Implementation Order

```
Phase 1 (session N+1): Foundation — ~5 tok/s
  [1A] Archive weight dedup (compiler.rs, ~30 min)
  [1B] MatMulActivationFusion pass (new file + pipeline, ~1 hr)
  [--] Test: TinyLlama f32 → 5+ tok/s, archive < 4.5 GB
  [--] Test: TinyLlama Q4 → archive < 1 GB, correct output

Phase 2 (session N+2): Kernel acceleration — ~40 tok/s
  [2A] Q4 AMX/BLAS hybrid in hologram base (matmul.rs + tape.rs, ~2 hr)
  [--] Test: TinyLlama Q4 → 20-40 tok/s

Phase 3 (session N+3): Flexibility — any prompt length
  [3A] Variable-length fix (resolve_size + 0-sentinels, ~2 hr)
  [--] Test: compile at seq=64, run with 24 tokens → correct

Phase 4 (session N+4): Speculative decoding — ~100 tok/s
  [4A] SpeculativeDecoder (new module, ~3 hr)
  [--] Test: TinyLlama + TinyLlama-draft → 60-100 tok/s

Phase 5 (session N+5): Polish
  [5A] Archive compression (compiler + CLI, ~1 hr)
  [--] Test: compressed archives load and execute correctly
```

## Performance Checkpoints

| Phase | tok/s | Archive Size | Prompt Flexibility |
|-------|-------|-------------|-------------------|
| Current | 2.5 (f32) / 1.0 (Q4) | 9.4 GB (Q4) | exact seq_len only |
| Phase 1 | ~5 (f32+fusion) | ~4.4 GB (f32), ~0.5 GB (Q4) | exact seq_len only |
| Phase 2 | ~40 (Q4+AMX) | ~0.5 GB (Q4) | exact seq_len only |
| Phase 3 | ~40 | ~0.5 GB | any prompt length |
| Phase 4 | ~100 | ~0.6 GB (target+draft) | any prompt length |
| Phase 5 | ~100 | ~0.2 GB compressed | any prompt length |

## Verification (every phase)

- `cargo test --release -p hologram-ai --features e2e -- tinyllama`
- Manual: `hologram-ai run model.holo --prompt "..." → "Paris"`
- `cargo clippy -- -D warnings` clean
- No regression in non-LLM models (BERT, ResNet)
- Archive size check
- tok/s measurement vs checkpoint target

## Key Files

### hologram-ai
- `crates/hologram-ai/src/compiler.rs` — compile_components, archive assembly
- `crates/hologram-ai/src/commands/run_cmd.rs` — execution, KV cache, CLI
- `crates/hologram-ai-common/src/opt/pipeline.rs` — optimization pass ordering
- `crates/hologram-ai-common/src/opt/matmul_activation_fusion.rs` — NEW
- `crates/hologram-ai-common/src/lower/builder.rs` — Q4 quantization gating
- `crates/hologram-ai-common/src/lower/strategy.rs` — 0-sentinel lowering
- `crates/hologram-ai/src/speculative.rs` — NEW

### hologram base
- `hologram-exec/src/float_dispatch/matmul.rs` — AMX hybrid kernel
- `hologram-exec/src/float_dispatch/mod.rs` — resolve_size() fix
- `hologram-exec/src/tape.rs` — Q4 dispatch routing
