# Plan 040: CPU Inference Performance — 100-200+ tok/s on Llama 8B

## Workflow

- **Pre-step:** Merge current `feat/stable-diffusion-pipeline` → `main` in hologram-ai
- **Branch:** `feat/cpu-inference-perf` in both `hologram` and `hologram-ai` (from main)
- **Plan file:** `specs/plans/040-cpu-inference-performance.md` (hologram-ai)
- **SPRINT.md:** Update `specs/SPRINT.md` as work progresses
- **Two repos:** hologram base changes first (kernels), then hologram-ai (wiring/integration)
- **Next session:** Stable diffusion work continues after this
- **Priority order:** macOS first → WASM second → x86_64 third

### Progress
- [x] Tier 1: All items already implemented (KV quant CLI, epilogue fusion, sparse V)
- [x] 2.2: Flash attention SIMD — NEON/AVX2 dot product + V accumulation (hologram base commit ca66111)
- [x] 2.5: wasm32-simd128 table lookup — i8x16_swizzle (hologram base commit 193eac7)
- **Baseline benchmarks collected:** matmul M=1 (9.4ms), M=128 (17.9ms), transformer decode step (13.0ms)

## Context

**Target:** 100-200+ effective tokens/second for Llama 3.1 8B on CPU (Apple Silicon primary, x86_64 secondary, WASM supported).

**Inspiration:** Taalas HC1 (custom ASIC, 17k tok/s) — but they eliminate the memory hierarchy by converting the model into silicon. We work within the memory bandwidth wall but use every technique to push effective throughput as high as possible.

**The memory bandwidth wall** (every decode token reads ALL weights once):
| Precision | Weight Size | DDR5 60 GB/s | Apple M-series 200 GB/s |
|-----------|-------------|--------------|------------------------|
| Q4        | ~4 GB       | ~15 tok/s    | ~50 tok/s              |
| Q2        | ~2 GB       | ~30 tok/s    | ~100 tok/s             |
| Ternary   | ~1 GB       | ~60 tok/s    | ~200 tok/s             |

**Current best:** 39.1 tok/s on TinyLlama 1.3B (Q4 + epilogue fusion + SIMD).

**How we get to 100-200+:** Two multipliers stack: (a) extreme quantization (Q2/ternary) doubles raw bandwidth ceiling to ~100-200 tok/s, and (b) speculative decoding generates 2-3 effective tokens per forward pass. Combined: **100-200+ tok/s effective** on Apple Silicon is achievable.

---

## WASM Compatibility

Every optimization is designed to work across all targets. WASM has graceful fallbacks already built in:

| Capability | Native (macOS/Linux) | WASM |
|-----------|---------------------|------|
| SIMD | AVX2/NEON (256/128-bit) | wasm32-simd128 (128-bit) or scalar fallback |
| Threading | rayon parallel levels | Sequential (automatic fallback) |
| Memory | mmap + madvise prefetch + huge pages | Linear memory + Vec<u8> (fallback exists) |
| GPU | Metal / WebGPU | WebGPU (auto-detected on wasm32) |
| BLAS | Accelerate (macOS) | Generic k-outer matmul (fallback exists) |
| Tape dispatch | Same | Same (pure Rust, zero platform deps) |
| LUT-GEMM | Same | Same (pure Rust, psumbook is stack-allocated) |
| Speculative decode | Same | Same (pure logic, no platform deps) |

**Key principle:** All optimizations are pure Rust with platform-specific fast paths behind `#[cfg]` gates. The WASM path is always functional — just at reduced throughput (~30-50% of native due to no BLAS, no threading, scalar SIMD fallback).

**WASM-specific enhancement (new):** Add `wasm32-simd128` support for the SIMD table lookup path in `hologram-core/src/view/simd.rs` — currently x86_64/aarch64 only, WASM falls to scalar. This would recover ~60-70% of native SIMD performance for elementwise ops and LUT apply.

---

## Tier 1: Wire Existing Infrastructure (1.3-1.5x, 1-2 weeks)

Kernel code already exists in hologram base but hologram-ai doesn't use it. WASM-compatible: yes (all pure Rust).

### 1.1 Wire KV Cache Quantization
- **Impact:** 1.1-1.2x decode at long context; major memory savings
- **What exists:** `KvCacheConfig::asymmetric_q4()` — K at F32, V at Q4, boundary-layer protection, Walsh-Hadamard rotation. All tested in hologram-exec.
- **What's missing:** hologram-ai hardcodes `KvCacheState::new()` (all F32)
- **Work:** Add `--kv-quant` CLI flag, pass config to `KvCacheState::with_config()`
- **Files:** `hologram-ai/src/commands/run_cmd.rs`, `hologram-ai/src/cli.rs`
- **WASM:** Works as-is (pure Rust quantize/dequantize)

### 1.2 Wire Epilogue Fusion End-to-End
- **Impact:** 1.15-1.25x (eliminates intermediate buffers for MatMul+Activation pairs)
- **What exists:** Tape kernels `MatMulLut4Activation`, `InlineMatMulActivation`
- **What's missing:** hologram-ai lowering emits plain MatMul, not fused variants
- **Files:** `hologram-ai-common/src/lower/strategy.rs`, `hologram-ai-common/src/lower/builder.rs`
- **WASM:** Works as-is (fusion is compile-time, not runtime)

### 1.3 Wire Sparse V Decode
- **Impact:** +22.8% decode at 32K context (skip V accumulation for negligible attention weights)
- **What exists:** `SPARSE_V_THRESHOLD` skip logic in attention kernel (Plan 038)
- **Work:** Verify active in non-BLAS path; ensure BLAS path has equivalent
- **Files:** hologram-exec `float_dispatch/attention.rs`
- **WASM:** Works as-is (pure comparison + branch)

---

## Tier 2: Compute Kernel Optimizations (2-3x, 2-4 weeks)

### 2.1 Speculative Decoding — THE Equation Changer
- **Impact:** 2-4x effective throughput — the ONLY way past the bandwidth wall
- **How:** Small draft model (TinyLlama 1.1B or distilled 2-layer head) generates N candidate tokens at ~4x speed. Large model verifies all N in one batched forward pass. 60-70% acceptance → 2-3x net speedup.
- **Key insight:** Verification of N tokens costs ~same as generating 1 token (weights read once regardless of batch size). This is the single most impactful optimization.
- **Architecture fit:** `BatchConfig` exists in executor; `dispatch_matmul` handles M>1
- **Work:**
  1. `DraftModel` struct wrapping a smaller `.holo` archive
  2. Speculative generation loop: draft N tokens → batched verifier input
  3. Acceptance/rejection with adjusted probability sampling
  4. CLI: `--draft-model <path>` flag
- **Files:** New module in hologram-ai; touches tape.rs, matmul.rs
- **WASM:** Fully compatible (pure Rust logic, sequential execution fine)

### 2.2 Flash Attention SIMD on CPU
- **Impact:** 1.2-1.5x at long context (>4K tokens)
- **Current state:** Online softmax (Flash-style) exists. Missing: SIMD dot product vectorization + L2-tiled K/V blocking
- **Work:** Hand-vectorize dot product with NEON/AVX2; add KC-blocked tiling (tiles fit in L2 ~256KB)
- **Files:** hologram-exec `float_dispatch/attention.rs`
- **WASM:** Scalar fallback works; wasm32-simd128 variant possible (128-bit dot product)

### 2.3 AMX/BLAS Hybrid for Quantized MatMul (Apple Silicon)
- **Impact:** 1.3-2x on Apple Silicon for quantized paths
- **Current state:** f32 BLAS already uses AMX via Accelerate. Quantized paths bypass BLAS.
- **Approach:** Dequant KC-blocked panel to f16, call `cblas_hgemm` (Accelerate → AMX)
- **Files:** hologram-exec `float_dispatch/matmul.rs`
- **WASM:** N/A (macOS only); WASM uses generic fallback or WebGPU

### 2.4 AVX-512 VNNI Micro-Kernels (x86_64)
- **Impact:** 1.5-2x over AVX2 for quantized inner loops (Intel Ice Lake+/AMD Zen4+)
- **Work:** `_mm512_dpbusd_epi32` variants behind `#[target_feature]` gate
- **Files:** hologram-exec `float_dispatch/matmul.rs`
- **WASM:** N/A (x86 only); WASM uses generic fallback

### 2.5 wasm32-simd128 Support (NEW — WASM-specific)
- **Impact:** 1.5-2x for WASM builds (recovers SIMD for elementwise ops, LUT apply, dot products)
- **Current state:** SIMD only for x86_64 (AVX2/SSE) and aarch64 (NEON); WASM falls to scalar
- **Work:** Add `#[cfg(target_arch = "wasm32")]` paths using `core::arch::wasm32` intrinsics: `v128_load`, `f32x4_mul`, `f32x4_add`, `i8x16_swizzle` (for LUT apply)
- **Files:** `hologram-core/src/view/simd.rs`, hologram-exec `float_dispatch/elementwise.rs`, `matmul.rs`

---

## Tier 3: Extreme Quantization (2x additional, 4-8 weeks)

This tier is what gets us from ~60-80 tok/s to 100-200+ tok/s.

### 3.1 Q2 / Ternary Quantization (BitNet-style)
- **Impact:** 2x over Q4 — halves weight reads; compute becomes trivial
- **How:** Two bits per weight (sign + nonzero). Inner loop: add, subtract, or skip. Process 64 weights per u64 with bitmask ops. Zero multiplies.
- **Quality:** Ternary Llama 8B models exist (BitNet b1.58 approach). Perplexity trade-off ~2-5% vs Q4 depending on calibration quality.
- **Why this matters:** At ternary (~1GB weights), Apple M-series bandwidth ceiling becomes ~200 tok/s raw. With speculative decode (2.5x), effective rate approaches **500 tok/s**.
- **Files:** New kernel module in hologram-exec
- **WASM:** Fully compatible — bitmask ops are pure integer arithmetic, excellent for WASM

### 3.2 Continuous Batching / Multi-User Throughput
- **Impact:** Nx total throughput for N concurrent users
- **How:** Amortize weight reads across N users per forward pass
- **Prerequisites:** Paged attention (Plan 016)
- **Files:** New scheduling module; extends `executor.rs` `BatchConfig`
- **WASM:** Compatible (sequential batching, no threading needed)

### 3.3 Sliding Window + Sparse Attention
- **Impact:** 1.3-1.5x at very long context (>8K tokens)
- **How:** Local window + global tokens. O(n^2) → O(n*w).
- **Files:** hologram-exec `float_dispatch/attention.rs`
- **WASM:** Compatible (pure control flow)

---

## Tier 4: System-Level (1.1-1.3x, 1 week)

### 4.1 Huge Pages for Weight Buffers (native only)
- `mmap` with `MAP_HUGETLB` (Linux) / `VM_FLAGS_SUPERPAGE_SIZE_2MB` (macOS)
- **Files:** hologram-exec `buffer/mmap_buf.rs`
- **WASM:** N/A (no-op fallback)

### 4.2 Compile-Time Weight Reordering
- Pre-pack weights into KC-blocked, NR-grouped layout at compile time
- **Files:** Archive format + hologram-exec `matmul.rs`
- **WASM:** Compatible (archive format is cross-platform)

### 4.3 Multi-Level Prefetch Enhancement (native only)
- Extend prefetch to first 4KB of weight buffers (currently first cache line only)
- **Files:** hologram-exec `tape.rs`
- **WASM:** N/A (no-op fallback)

---

## Resource-Constrained Hardware Considerations

### Memory Budget
Running on resource-constrained hardware means memory is as important as throughput.

**Speculative decoding memory overhead:**
- Llama 8B Q4 (~4GB) + draft model must fit simultaneously
- TinyLlama 1.1B Q4 (~0.5GB) as draft: total ~4.5GB — fits 8GB devices
- Llama 3.2 1B Q4 (~0.5GB) as draft for Llama 3.1 8B: total ~4.5GB
- **Fallback:** If draft model doesn't fit, auto-disable speculative decode (memory-aware path)
- **Q2/ternary:** Llama 8B ternary (~1GB) + draft Q4 (~0.5GB) = ~1.5GB — fits 4GB devices

**KV cache memory at long context:**
| Context | F32 KV (32 layers) | Q4 V + F32 K | Savings |
|---------|-------------------|--------------|---------|
| 2K | 512 MB | 320 MB | 37% |
| 8K | 2 GB | 1.3 GB | 35% |
| 32K | 8 GB | 5 GB | 37% |

**Memory-aware runtime behavior:**
- Query available memory at startup (platform-specific: `sysctl hw.memsize` macOS, `/proc/meminfo` Linux, `navigator.deviceMemory` WASM)
- Auto-select quantization level: if <4GB available → force ternary; <8GB → force Q4; ≥8GB → allow F16
- Auto-select KV cache config: if <2GB headroom → asymmetric Q4 V; <1GB → Q4 K + Q4 V
- Auto-disable speculative decode if draft model doesn't fit
- Report memory budget and decisions at startup via tracing

### Draft Model Compatibility
Speculative decoding requires draft and target models to share the same tokenizer/vocabulary.

**Supported pairs:**
| Target Model | Draft Model | Shared Tokenizer |
|-------------|-------------|-----------------|
| Llama 3.1 8B | Llama 3.2 1B | Yes (Llama 3 tokenizer) |
| Llama 3.1 8B | TinyLlama 1.1B | No — different vocab, incompatible |
| Llama 2 7B | TinyLlama 1.1B | Yes (Llama 2 tokenizer) |

**Validation:** At load time, verify draft and target have identical `vocab_size` and tokenizer config. Fail fast with clear error if mismatched.

### Profiling-First Approach
Before diving into kernel work, profile TinyLlama decode to identify actual bottleneck breakdown:

1. **Step 0 (before any changes):** Run with `--profile` flag, collect per-op timing
2. **Expected breakdown:** ~70-80% matmul, ~10-15% attention, ~5-10% overhead (KV read/write, softmax, norm)
3. **Verify assumptions:** If attention dominates more than expected, reprioritize Flash attention SIMD (2.2) over speculative decoding (2.1)
4. **Tools:** `instruments` (macOS), `perf stat` + `perf record` (Linux), Chrome DevTools (WASM)

### Incremental Mergeability
Each tier is independently mergeable to `main`:

- **Tier 1:** Pure wiring, no kernel changes, zero risk of regression → merge fast
- **Tier 2:** Each item (2.1–2.5) is independently mergeable behind feature flags
- **Tier 3:** Each item is independently mergeable; Q2 kernel is additive (doesn't change Q4 path)
- **No tier depends on a later tier.** Partial implementation still delivers value.

### API Coordination: hologram base → hologram-ai

Each hologram base change introduces new APIs that hologram-ai must wire. Track these per-tier:

**Tier 2 base API changes → hologram-ai integration:**
| Base Change | New/Changed API | hologram-ai Integration |
|-------------|----------------|------------------------|
| Flash attention SIMD (2.2) | No API change (internal optimization) | None needed |
| AMX/BLAS hybrid (2.3) | No API change (internal dispatch) | None needed |
| AVX-512 VNNI (2.4) | No API change (internal dispatch) | None needed |
| wasm32-simd128 (2.5) | No API change (internal dispatch) | None needed |

**Tier 3 base API changes → hologram-ai integration:**
| Base Change | New/Changed API | hologram-ai Integration |
|-------------|----------------|------------------------|
| Q2/ternary kernel (3.1) | New `FloatOp::MatMulTernary` or `QuantFormat::Ternary` | Lowering in `strategy.rs` must emit ternary matmul when quant config requests it; `builder.rs` must handle new op |
| Sliding window (3.3) | New param on `FloatOp::Attention { window_size: Option<u32> }` or new `FloatOp::SlidingWindowAttention` | Lowering must pass window size from model metadata; `AiOp::GroupedQueryAttention` needs `window_size` field |

**Tier 4 base API changes → hologram-ai integration:**
| Base Change | New/Changed API | hologram-ai Integration |
|-------------|----------------|------------------------|
| Huge pages (4.1) | New `MmapConfig { huge_pages: bool }` | Pass through from `InferenceConfig` |
| Weight reordering (4.2) | New `WeightLayout::Packed` in archive format | Compiler must emit packed layout; archive version bump |
| Prefetch (4.3) | No API change (internal) | None needed |

### Feature Flag Propagation

New hologram base features must be exposed through hologram-ai's dependency:

```toml
# hologram-ai/Cargo.toml — additions needed
[dependencies.hologram]
features = ["std", "parallel", "accelerate"]  # existing
# Add as each tier lands:
# "avx512"    — Tier 2 (2.4)
# "simd128"   — Tier 2 (2.5)
# "ternary"   — Tier 3 (3.1)
```

### Archive Format (v1 — no backwards compatibility needed)

The `.holo` archive already has strong foundations for resource-constrained streaming. We extend v1 directly (can break APIs freely):

**Already exists and ready to use:**
- `SectionTable` — binary-indexed TOC for random access to any section
- `WeightIndex` with `group_byte_range(group)` — byte ranges per layer group (e.g., "layers.0", "embed")
- `FLAG_TENSOR_PAGE_ALIGNED` — 4KB-aligned tensors for zero-copy GPU + efficient madvise paging
- `HoloLoader::prefetch_range()` / `release_range()` — OS paging hints per layer
- `ModelMetaSection` — metadata-only reads (model info without loading weights)
- `LayerHeader` with `LayerDescriptor` — layer-by-layer execution schedule with plan offsets
- `TensorMetadata` with shape, dtype, quantization params per tensor

**New additions to v1 for this plan:**
1. **Per-layer-group weight compression** — currently whole-blob only; change to compress per layer group so layers can be decompressed independently during streaming
2. **`QuantizationScheme::Ternary`** — add variant for Q2/ternary weights (3.1)
3. **Packed weight layout marker** — `WeightLayout::Packed` in `TensorMetadata` for pre-reordered weights (4.2)
4. **`InferenceConfig` section** — embed recommended runtime config (KV quant, memory budget hints) in archive

**Layer-by-layer streaming execution (executor changes in hologram base):**
1. **Lazy constant seeding** — don't load all weights at startup; resolve from mmap on-demand during instruction dispatch
2. **Per-instruction eviction** — decrement consumer counts + `release_range()` after each instruction (not just level boundaries)
3. **Layer-boundary prefetch** — use `WeightIndex.groups()` to prefetch next layer while executing current layer

**Shape inference in archive:**
- `SerializedGraph.node_shapes` — compiled N-D shapes per node (0 = runtime-resolved symbolic dim)
- `SerializedGraph.constant_shapes` — weight tensor shapes
- `SerializedGraph.node_dtypes` — output dtype per node
- All readable without loading weights (graph section is independent)

### Unified InferenceConfig

All new runtime knobs should flow through a single config struct (not scattered CLI flags):

```rust
pub struct InferenceConfig {
    pub kv_cache: KvCacheConfig,          // 1.1
    pub speculative: Option<SpecConfig>,   // 2.1
    pub memory_budget: Option<usize>,      // auto-detect or override
    pub quant_format: QuantFormat,         // Q4, Q2, Ternary
    pub window_size: Option<u32>,          // 3.3
}
```

CLI flags map to this struct. Library/WASM API takes it directly. This keeps the API clean for all consumers.

### CI Benchmark Regression Gate
Add benchmark gate to prevent silent throughput regressions:

- **New CI job:** `benchmark-regression` in `.github/workflows/benchmarks.yml`
- **Mechanism:** Compare current bench results against stored baseline (committed as `benches/baseline.json`)
- **Threshold:** Fail CI if any key benchmark regresses >5% from baseline
- **Key benchmarks gated:** matmul M=1 (decode), matmul M=128 (prefill), lut_gemm Q4, tape executor
- **Baseline update:** Manual commit when intentional trade-offs are made (e.g., trading prefill speed for decode speed)

### Quality Metrics for Extreme Quantization
Q2/ternary quantization needs quality validation beyond numerical tolerance:

- **Perplexity benchmark:** Evaluate on WikiText-2 (standard LLM eval set)
- **Acceptable threshold:** <5% perplexity increase over Q4 baseline
- **Per-layer sensitivity analysis:** Measure perplexity impact of quantizing each layer individually to identify sensitive layers that should remain at higher precision (similar to boundary-layer protection in KV cache)
- **Mixed-precision fallback:** If ternary exceeds threshold, automatically keep sensitive layers (first 2, last 2, + any identified by sensitivity analysis) at Q4

---

## Performance Projections

### Apple M3 Pro (200 GB/s unified memory)

| Stage | Effective tok/s | Key Multiplier |
|-------|----------------|----------------|
| Baseline (Q4) | ~35-40 | — |
| + Tier 1 (wire existing) | ~45-55 | 1.3-1.5x |
| + Tier 2 (spec decode + kernels) | **~90-130** | 2.5x (spec decode) |
| + Tier 3 (Q2/ternary) | **~150-200+** | 2x (halved weights) |
| + Continuous batching (N=4) | **~400+ total** | Nx users |

### DDR5 Desktop (60 GB/s)

| Stage | Effective tok/s | Key Multiplier |
|-------|----------------|----------------|
| Baseline (Q4) | ~10-12 | — |
| + Tier 1 | ~14-17 | 1.3-1.5x |
| + Tier 2 (spec decode) | ~30-45 | 2.5x |
| + Tier 3 (Q2/ternary) | **~60-90** | 2x |

### WASM (browser, no threading, scalar or simd128)

| Stage | Effective tok/s | Notes |
|-------|----------------|-------|
| Baseline (Q4) | ~10-15 | No BLAS, sequential |
| + Tier 1 + simd128 | ~18-25 | SIMD recovery |
| + Spec decode | ~40-60 | Draft model in same WASM module |
| + Q2/ternary | ~60-100 | Bitmask ops fast even in WASM |
| + WebGPU offload | **~80-150** | Large matmuls on GPU |

---

## Work Split by Repo

### hologram base (`feat/cpu-inference-perf`)
| Item | File(s) | Description |
|------|---------|-------------|
| 2.2 | `hologram-exec/src/float_dispatch/attention.rs` | Flash attention SIMD + L2 tiling |
| 2.3 | `hologram-exec/src/float_dispatch/matmul.rs` | AMX/BLAS hybrid for quantized matmul |
| 2.4 | `hologram-exec/src/float_dispatch/matmul.rs` | AVX-512 VNNI micro-kernels |
| 2.5 | `hologram-core/src/view/simd.rs` | wasm32-simd128 support |
| 3.1 | `hologram-exec/src/lut_gemm/` (new) | Q2/ternary quantization kernel |
| 3.3 | `hologram-exec/src/float_dispatch/attention.rs` | Sliding window attention |
| 4.1 | `hologram-exec/src/buffer/mmap_buf.rs` | Huge pages |
| 4.2 | `hologram-exec/src/float_dispatch/matmul.rs` + archive | Compile-time weight reordering |
| 4.3 | `hologram-exec/src/tape.rs` | Multi-level prefetch |
| 5.1 | `hologram-archive/src/weight/quantize.rs` | `QuantizationScheme::Ternary` variant |
| 5.2 | `hologram-archive/src/weight/mod.rs` | `WeightLayout::Packed` in `TensorMetadata` |
| 5.3 | `hologram-archive/src/writer/holo_writer.rs` | Per-layer-group weight compression |
| 5.4 | `hologram-exec/src/tape.rs` | Lazy constant seeding + per-instruction eviction |
| 5.5 | `hologram-exec/src/tape.rs` | Layer-boundary prefetch using `WeightIndex.groups()` |

### hologram-ai (`feat/cpu-inference-perf`)
| Item | File(s) | Description |
|------|---------|-------------|
| 1.1 | `run_cmd.rs`, `cli.rs` | Wire KV cache quantization |
| 1.2 | `lower/strategy.rs`, `lower/builder.rs` | Wire epilogue fusion |
| 1.3 | (verification only) | Verify sparse V decode active |
| 2.1 | New module + `run_cmd.rs` | Speculative decoding |
| 3.2 | New scheduling module | Continuous batching |

## Implementation Order

```
Week 1-2 (Tier 1 — hologram-ai only, no base changes):
  [1.1] KV cache quant wiring (CLI + config passthrough)
  [1.2] Epilogue fusion wiring (lowering strategy)
  [1.3] Sparse V decode verification

Week 3-4 (Tier 2a — hologram base kernel work):
  [2.2] Flash attention SIMD (NEON/AVX2 dot product + L2 tiling)
  [2.5] wasm32-simd128 support (view/simd.rs + elementwise)
  [4.2] Compile-time weight reordering

Week 5-8 (Tier 2b — both repos):
  [2.1] Speculative decoding ← highest single impact, gets us to ~100 tok/s
       hologram base: batch-aware matmul/attention paths
       hologram-ai: DraftModel, generation loop, CLI
  [2.3] AMX/BLAS hybrid for quantized matmul (hologram base, Apple Silicon)
  [2.4] AVX-512 VNNI micro-kernels (hologram base, x86_64)

Week 9-12 (Tier 3 — both repos):
  [3.1] Q2/ternary quantization kernel (hologram base) ← gets us to 150-200+ tok/s
  [3.2] Continuous batching (hologram-ai)
  [3.3] Sliding window attention (hologram base)

Week 12+ (Tier 4 + archive streaming — hologram base):
  [4.1] Huge pages
  [4.3] Multi-level prefetch
  [5.1] QuantizationScheme::Ternary in archive (pairs with 3.1)
  [5.2] WeightLayout::Packed in archive (pairs with 4.2)
  [5.3] Per-layer-group compression in archive writer
  [5.4] Lazy constant seeding + per-instruction eviction in executor
  [5.5] Layer-boundary prefetch in executor
```

**Critical path to 100+ tok/s:** Tier 1 (wire existing) → Speculative decoding (2.1) → Q2/ternary (3.1). These three give the biggest jumps.

---

## Testing & Verification

### Comprehensive tests for ALL new code
Every item must have corresponding tests before merging:

- **Tier 1 tests (hologram-ai):**
  - KV cache quant: unit test exercising `KvCacheState::with_config()` through CLI, decode with Q4 V
  - Epilogue fusion: conformance test verifying fused MatMul+Activation matches unfused
  - Sparse V decode: test that sparse skip produces same output as dense (within tolerance)

- **Tier 2 tests (hologram base + hologram-ai):**
  - Flash attention SIMD: test SIMD path matches scalar path (exact bit-for-bit or within 1e-6)
  - wasm32-simd128: test SIMD matches scalar on representative inputs
  - Speculative decoding: test acceptance/rejection logic, test that verified output matches greedy decode
  - AMX/BLAS hybrid: test dequant+BLAS matches reference fused-dequant path
  - AVX-512 VNNI: test VNNI micro-kernel matches generic micro-kernel

- **Tier 3 tests (hologram base + hologram-ai):**
  - Q2/ternary: quantize → dequant round-trip test, matmul output matches f32 within tolerance
  - Continuous batching: test batch=1 matches sequential, test batch=N correctness
  - Sliding window: test window attention matches full attention for short sequences

### TinyLlama regression gate
- **Before each commit:** run TinyLlama ONNX end-to-end decode test
- **Verify:** coherent English output, tok/s not regressed, top-5 tokens match baseline
- **Test:** `cargo test --release -p hologram-ai -- tinyllama` (existing test suite)

### Benchmark verification
- **Before starting (baseline):** run `cargo bench` in hologram base (matmul, lut_gemm, executor, attention)
- **After each tier:** re-run benchmarks, record tok/s delta
- **Key benchmarks:**
  - `matmul.rs` — M=1 vecmat (decode), M=128 GEMM (prefill)
  - `lut_gemm.rs` — Q4/Q8 at representative sizes
  - `executor.rs` — full tape execution (transformer decode step)
  - End-to-end: TinyLlama decode tok/s
- **Target checkpoints:**
  - After Tier 1: ~45-55 tok/s (Apple Silicon) — 1.3-1.5x baseline
  - After Tier 2: ~90-130 tok/s — 2.5-3.5x baseline
  - After Tier 3: ~150-200+ tok/s — 4-5x baseline

### End-to-end benchmarks in hologram-ai
Add matching full-stack benchmarks (lowering + tape build + execution) alongside hologram base micro-benchmarks:
- `benches/decode_tok_s.rs` — TinyLlama and Llama 8B decode tok/s (measures real throughput including all overhead)
- `benches/prefill_tok_s.rs` — prefill throughput at various sequence lengths
- `benches/memory_peak.rs` — peak memory usage during inference
- These complement hologram base's micro-benchmarks (matmul, attention, etc.) by measuring full-stack impact

### Conformance
- `cargo test -p hologram-ai-conformance` — all op conformance tests pass
- `cargo clippy -- -D warnings` — no lint warnings
- `cargo fmt --check` — formatting clean

## Key Files
- `hologram-exec/src/float_dispatch/matmul.rs` — GEMM kernels, SIMD, BLAS, fused dequant
- `hologram-exec/src/float_dispatch/attention.rs` — attention, sparse V, online softmax
- `hologram-exec/src/tape.rs` — tape executor, prefetch, parallel dispatch
- `hologram-exec/src/lut_gemm/` — quantized GEMM, psumbook, fiber accumulation
- `hologram-exec/src/kv_cache.rs` — KV cache quantization
- `hologram-core/src/view/simd.rs` — SIMD dispatch (add wasm32-simd128 here)
- `hologram-ai/src/commands/run_cmd.rs` — CLI integration
- `hologram-ai-common/src/lower/strategy.rs` — lowering (fusion wiring)
- `hologram-ffi/src/wasm/mod.rs` — WASM bindings
