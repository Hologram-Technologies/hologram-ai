# Current Sprint — hologram-ai

## Sprint Goal

**Correct ONNX execution, then optimize.**

All ONNX ops compile and execute faithfully — every node matches ORT within
tolerance. No premature fusion or optimization until the baseline is proven.

**Design principle:** hologram-ai is a compiler only (ADR-0016). It ships
zero runtime code. All kernels belong in hologram base crate.

---

## Milestone: ONNX Correctness (DONE)

- [x] **Fully static shapes at compile time** — all dims concretized to
  `context_length` (seq-like) or 1 (batch-like). No 0-sentinels, no runtime
  shape walker needed. `--seq-len` CLI flag for user override.
- [x] **AttentionFusion** — 22 SDPA chains fused into GroupedQueryAttention
  ops. Conformance tests pass (GQA flat, expanded, scaled dot product).
- [x] **KvSlotInjection** — KV cache write/read ops injected after fusion.
  Pipeline archive: prefill (full seq) + decode (seq=1).
- [x] **Node-by-node inspector** — `tinyllama_node_inspector` test dumps all
  intermediate buffers; `compare_node_by_node.py` finds first divergent node.
- [x] **TinyLlama ONNX matches ORT** — all 1156 compared nodes pass.
  Top-5 token predictions identical. Zero failures.
- [x] **TinyLlama generates coherent English** — "I'm not a joke teller,
  I just like to laugh at them!" (causal model, KV cache decode, 1.4 tok/s)

---

## Active: Performance (make it fast)

### P1: Prefill speed (DONE — via --seq-len)
- [x] Compile with `--seq-len 32` for short prompts — prefill 15s → **170ms**
  (89x speedup). Variable-length prefill deferred to P5 (now done).
- [x] Verified: TinyLlama ONNX pipeline, 29.9 tok/s
- Note: chat models require the user to supply the full chat template in
  `--prompt` (e.g. `<|user|>\nTell me a joke</s>\n<|assistant|>` for
  TinyLlama-Chat). The CLI does not apply templates automatically.

### P2: Decode speed — wire Sprint 13 infrastructure (DONE)

#### P2a: Execution hot-path fast paths (DONE — hologram base)
- [x] SameAs(0) fast path in `propagate_level_shapes` — skip full shape
  resolution for elementwise ops (~60-70% of nodes, just copy input[0] shape)
- [x] Skip `input_shapes` gathering in `dispatch_level` for unary ops and
  non-broadcasting binary ops (eliminate per-input HashMap lookups)

#### P2b: Tape executor (DONE — hologram base)
- [x] `TapeBuilder` — pre-resolve kernel fn pointers + `output_elem_size`
  per node at graph-load time, with dynamic size resolution for Softmax,
  RmsNorm, LayerNorm, Reduce* (eliminates per-dispatch op match + HashMap
  lookups for `compiled_dtypes`)
- [x] `BoxedTape` execution loop with prefetch + pre-computed elem_size
- [x] Wire tape executor into hologram public API (`build_tape_from_plan`,
  `execute_tape`)

#### P2c: Integration (DONE — hologram-ai)
- [x] Wire tape executor from `HoloRunner` — build tape at load time, use
  `execute_tape` / `execute_tape_with_kv` for all execution. No fallback
  to legacy `execute_plan` (Plan 022)
- [x] Migrate `run_with_shape_context()` from `execute_plan` to tape API
  (Plan 023 — hologram Sprint 17 removed all `execute_plan*` functions)
- [x] Remove intermediate capture debug tests — API deleted in hologram
  Sprint 17 (Plan 023)

#### P2d: Remaining decode optimizations (DONE — Plan 020)
- [x] Wire `dispatch_float_into` — buffer reuse, wired into tape executor
  via `BoxedInstruction::FloatInto` (eliminates per-op allocations)
- [x] Wire `WeightCache` into tape executor — `TapeContext.weight_cache`
  caches deserialized quantized weights across dispatches
- [x] Level-aware tape execution — `Tape.level_offsets` splits execution by
  level; KvWrite/KvRead as `TapeKernel` enum variants; parallel level
  execution via rayon (`execute_parallel()`)
- Note: f32 ONNX decode at 13.6 tok/s is near memory bandwidth ceiling
  (4.1 GB weights × ~60 GB/s DDR ≈ 15 tok/s theoretical max). Further
  speedup requires weight quantization — see GGUF models section.

### P3: Compiler fusion passes (Plan 019 — TensorBend-inspired)
- [x] SwiGLU fusion pass — pattern-match `SiLU(gate) * up` into
  `FusedSwiGLU`. Implemented in `swiglu_fusion.rs`, wired into MVP pipeline.
  Eliminates 1 intermediate tensor + 1 dispatch per transformer layer.
- [x] Add+RMSNorm residual fusion — `AddRmsNormFusion` pass in
  `add_rmsnorm_fusion.rs`, wired into MVP pipeline; lowering maps to
  `FloatOp::AddRmsNorm`; kernel implemented in hologram base.
- [ ] QK-Norm + RoPE pre-attention fusion — **pass removed** (hologram base
  `dispatch_attention()` still ignores `qk_norm`/`rope` flags). AiOp fields
  and lowering stubs retained as forward-compatible placeholders. Re-add pass
  when hologram base wires kernel support.

### P4: Compilation speed (DONE — Plans 017, 020)
- [x] Release profile with LTO (`codegen-units = 1, lto = "thin"`)
- [x] Extract shared `post_concretization_repair` (was duplicated 3x in
  compiler.rs, now a single function with early convergence detection)
- [x] Early convergence detection in fixpoint loop (break when dynamic dims
  stop decreasing, saves up to 9 pass invocations)
- [x] Cache `topo_order` on AiGraph (was called ~40 times per compilation,
  each building 3 HashMaps; now cached with `RefCell` + invalidation)
- [x] Avoid double LLM compilation (clone AiGraph after MVP, concretize
  twice instead of re-importing from disk — ~50% LLM compile time savings)

### P5: Variable-length prefill (DONE)
- [x] **Blocker resolved:** hologram base now applies `resolve_size()` in
  both tape executor AND legacy `dispatch_float_ctx` paths. Softmax, RmsNorm,
  LayerNorm, Reduce*, InstanceNorm all resolve from runtime buffer sizes.
  MatMul uses `infer_matmul_k()` to re-derive k from buffers.
- [x] `mini_transformer_variable_seq_len_runs` test passes (seq=1, 7, 128)
- [x] `SeqMode::Variable` enabled as default in `run_cmd.rs`
- [ ] Wire `ShapeContextGraph` into `HoloRunner.execute()` — project shapes
  at runtime from actual input dimensions instead of compiled seq_len
- [x] Any prompt length without recompilation (via runtime size resolution)

### P6: Performance deep clean (Plan 024 — active)
- [x] Remove `hologram-ai-ggml` stub crate (entire crate was unimplemented)
- [x] Remove 3 unregistered fusion passes: `MatMulActivationFusion` (~233 lines),
  `ConcatMatMulFusion` (~213 lines), `PreAttentionFusion` (~415 lines) — hologram
  base has no corresponding kernels. AiOp variants + lowering stubs retained.
- [x] Remove `project_shapes()` dead method from HoloRunner
- [x] Remove unused dependencies: `async-stream`, `tokio-util` (zero usage)
- [x] `collect_weight_bytes()` single pre-allocated buffer — eliminates N
  intermediate `Vec<u8>` allocations (one per mmap'd param)
- [x] Uncompressed archives by default — enables `load_from_bytes_zero_copy()`
  (removed forced `compress_graph()` / `compress_weights()`)
- [x] `topo_order()` returns `Rc<Vec<NodeId>>` — eliminates clone on every call
  (was cloned 50+ times per compilation across all opt passes)
- [x] Eliminate weight clone in LLM pipeline — decode component passes `None`
  weights, registered under shared group via `WeightStore` group-reuse path.
  Same for multi-ONNX: removed `weight_cache` HashMap, non-first-in-group
  components pass `None`.
- [ ] Clone elimination — remaining `.clone()` calls via move semantics, `Cow`,
  shape reference folding
- [x] Path compression in `constant_fold.rs` remap chains — O(α(n)) amortized
  instead of O(chain_length) per resolution
- [x] Rayon parallelization — subgraph optimization via `into_par_iter()`,
  `Arc<Vec<NodeId>>` topo cache for Send compatibility
- [x] `eprintln!` → `tracing::info!/warn!` in `run_cmd.rs` (20+ calls migrated)
- [x] Parallelization — rayon `par_iter()` for multi-component compilation
  (import → optimize → concretize → memory plan per component in parallel)
- [ ] Parallelization — weight I/O, constant dedup hashing
- [ ] Clone elimination — remaining `.clone()` calls via move semantics, `Cow`,
  shape reference folding
- [ ] Worklist dtype fixpoint in shape_prop.rs
- [x] **Prefill fixed**: Transpose no-op and batched MatMul bugs fixed in
  hologram base. Prefill logits match ORT exactly (conformance tests pass).
- [ ] **BLOCKER: Decode produces gibberish** → **Solution: Plan 026 —
  Single-Model Rearchitecture.** Remove the prefill/decode dual-model
  split entirely. Compile ONE model, use it for both prefill and decode.
  The runtime KV cache (`write_pos == 0` check, `resolve_size()`,
  `TensorMeta`) already handles both modes. Pipeline format becomes
  universal (1 or N components). Removes ~400+ lines of dual-model code
  and eliminates the entire class of layout mismatch bugs.

### P7: LUT-GEMM Wiring + Compile-Time Quantization (Plan 030)
- [x] `--quantize q4_0` CLI flag on compile command (`cli.rs`)
- [x] `QuantStrategy` enum (`None`, `Auto`, `Q4_0`, `Q8_0`) threaded through
  `ModelCompiler.quant_strategy` → `LoweringOptions` → `lower()`
- [x] `try_convert_f32_to_lut4` in `builder.rs` — compile-time f32 → k-means
  Q4 quantization for ONNX MatMul weights. Skips small weights (<256 per dim).
  Uses `hologram::hologram_exec::lut_gemm::quantize::quantize_4bit` + rkyv serialize.
- [x] GGUF Q4_0 → LUT-GEMM interception already wired (`try_convert_q4_0_to_lut4`
  in builder.rs:340-377, intercepts `Gemm { quant_b: 1 }`)
- [x] Per-step timing instrumentation in `run_cmd.rs` (archive load, prefill,
  per-decode-step timing for first 5 steps)
- [x] Criterion benchmark harness (`benches/inference.rs` — TTFT, 20-token
  decode, single decode step for ONNX + GGUF)
- [x] E2E validation: compile with `--quantize q4_0` fires `MatMulLut4` conversions
  (26 weight matrices quantized). f32 baseline verified correct: logits match ORT
  to 5 decimal places (top-5 token IDs identical, max Δ=2.1e-5).
- [x] Variable-length execution: works at full context (compiled seq=2048,
  any prompt length). Intermediate seq (32/64/128) has Reshape meta ambiguity
  — tracked in Plan 033 (ShapeContextGraph post-fusion). Production path:
  compile at auto-detected context length.
- [x] Q4_0 output quality: ONNX `--quantize q4_0` produces correct output
  ("The capital of France is Paris.") via LUT-GEMM Psumbook path.
  GGUF Q4_0 has quality loss from double quantization (Q4→f32→k-means).
- [ ] Q4_0 performance: Psumbook kernel is 30× slower than f32 BLAS —
  **Plan 037** (active):
  - [x] Phase 1: Fused Q4_0 dequant-matmul — `matmul_dequant_q4_0` eliminates
    full K×N f32 materialization (hologram base). 3 bit-exact tests.
  - [x] Phase 1b: Fused Q6_K dequant-matmul — `matmul_dequant_q6_k` extends
    fused path to Q6_K 210-byte super-blocks. 3 bit-exact tests.
  - [x] Phase 6: Per-op profiling — `#[cfg(feature = "profile")]` tracing span
    on `dispatch_float_into` for SD pipeline analysis.
  - [x] Phase 2: Adaptive remainder micro-kernels — `m_remainder_tiled()`
    uses `micro_kernel_packed<2, 8>` and `<1, 8>` for leftover rows.
  - [x] Phase 3: Explicit SIMD micro-kernels — NEON (`vfmaq_f32`) on aarch64,
    AVX2+FMA (`_mm256_fmadd_ps`) on x86_64. Both packed and strided variants.
    Wired into all matmul paths (f32, fused Q4_0, fused Q6_K).
  - [x] Phase 4: SIMD Psumbook dot — NEON `dot_neon_256` + AVX2 `dot_avx2_256`
    (pre-existing, verified).
  - [x] Phase 5: Page-aligned tensors — `is_tensor_page_aligned()` reader +
    3 roundtrip tests for `page_align_weight_blob`.
  - [x] Phase 7: Static duty partitioning — `with_min_len(duty)` on all 3
    parallel matmul paths (f32, Q4_0, Q6_K).
- [x] Epilogue fusion wired: `MatMulRelu`/`MatMulGelu`/`MatMulSilu` →
  `GraphOp::FusedMatMulActivation` via `wrap_graph_op()` in strategy.rs.
  Tape builder maps to `InlineMatMulActivation`. Decode: 20.5 → 28.5 → 39.1 tok/s.
- [x] Persistent WeightCache: `HoloRunner.weight_cache` (RefCell) persists
  deserialized quantized weights across decode steps.
- [x] Decode prewarm skip: skip `prewarm_arena` for decode steps (write_pos > 0),
  reducing oversized buffer allocation. 2× decode throughput at large compiled seq.
- [x] Auto-detect causal LM: ONNX export script uses `AutoModelForCausalLM` for
  decoder models (LLaMA, GPT, Mistral, etc.) producing logits output directly.

---

## Medium Term: Multi-model support

### Any ONNX model
- [x] Test with ResNet-50 (vision, no attention) — **compiles and executes**
  (225 nodes after BatchNorm decomposition + constant folding, [1, 1000] output,
  all finite values). Conv2d conformance tests pass.
- [x] Test with BERT (encoder-only, bidirectional attention) — **compiles and executes**
  (507MB ONNX, bert-base-uncased, seq=32). Non-causal attention detected correctly,
  KV cache skipped, single-graph path used. Shape→Gather→Concat chains folded
  at compile time via `ForceConcretize` + `ConstantEvaluation` Shape-node eval.
  hologram base inline dispatch fixed (9 missing ops). Shape propagation hardened:
  never downgrade Concrete dims to Dynamic (prevents post-concretization shape
  regression in intermediate attention tensors). Output: [1, 32, 768] all finite.
- [ ] **Stable Diffusion support (Plans 027-035)**
  - [x] Phase 1: GroupNorm lowering — `FloatOp::GroupNorm` in hologram base + lowering.
  - [x] Phase 2: UNet compilation — compiles (1634 nodes, 3.4 GB, 0 warnings).
  - [x] Phase 3: Output type system — manifest `kind` field, `ModelKind::ImageGen`.
  - [x] Plan 028-029: Runtime shape resolution + compiler shape hardening.
  - [x] Plan 031: ONNX import parameter inference.
  - [x] Text encoder (CLIP) compilation — fixed Slice param resolution for dynamic
    ends + added `FloatOp::ArgMax` (rayon-parallel). 384 nodes, 492 MB.
  - [x] VAE decoder compilation — 290 nodes, 198 MB.
  - [x] All 3 SD v1.5 components compile successfully.
  - [x] Plan 035: Runtime acceleration — hologram base Conv2d BLAS sgemm (526x
    VAE speedup), liveness-based arena eviction (UNet RSS 31GB → <1GB), parallel
    float matmul, weight index + mmap prefetch/release, pipeline archive
    auto-detection (`load_auto`), alignment safety, binary_broadcast fix.
  - [x] UNet execution — 87s on CPU with BLAS Accelerate, <1GB RSS via mmap
    zero-copy weights + liveness eviction. Output empty (0 bytes) due to
    pre-existing shape inference bug in Resize→Conv chain.
  - [x] VAE decoder execution — 0.9s with BLAS Conv2d. Panics late on empty
    buffer (node 282 Reshape from Resize with wrong spatial dims).
  - [x] Resize scales shape inference — shape_prop now reads float scale
    constants from params and multiplies input spatial dims (was falling back
    to input shape, producing 2x2 spatial after concretization).
  - [x] Weight alignment padding — `collect_weight_bytes` pads each tensor
    to 4-byte boundary, preventing bytemuck cast failures on mmap'd weights.
  - [ ] VAE decoder correctness — verify Resize fix produces correct Conv2d
    spatial dims (128, 256, 512) and full [1,3,512,512] output.
  - [ ] Spatial scale compilation — `ModelCompiler::spatial_scale` divides input
    spatial dims for reduced-resolution compilation. Compilation succeeds but
    intermediate shapes are not fully re-derived (heuristic scaling misidentifies
    which 4D tensors are activations vs structural). Correct approach: scale only
    graph input shapes and let shape_prop recompute everything via Resize scales
    and Conv2d output formulas. Requires shape_prop to handle all ops correctly
    (Resize scales fix in shape_prop.rs is prerequisite).
  - [ ] Activation memory — full-res VAE uses 51GB (per-instruction eviction
    reduces to ~20GB via arena slot count drop, but system allocator fragmentation
    prevents RSS reduction). Need: explicit `Vec::shrink_to_fit()` or `madvise`
    on freed pages, or switch to arena allocator that returns pages to OS.
  - [x] Plan 036: Streaming executor — per-instruction eviction (Phase 2) moves
    eviction from level boundaries to after each instruction. RSS drops during
    execution as completed activations are freed immediately. Phases 1 (lazy
    constants) and 3 (mmap release) infrastructure exists but not wired;
    constants already zero-copy borrowed, OS handles page reclamation.
  - [x] All 3 components execute independently: text encoder (9.9s), UNet
    (12.7s per step), VAE decoder (7.75s). MmapBuffer arena eviction bounds
    RSS. Encoder-only detection prevents KV injection on CLIP.
  - [x] **Full pipeline generates 512×512 image** — tokenize (CLIP BPE) →
    text encoder → 20-step Euler-a denoising → VAE decode → PPM output.
    337 seconds total on CPU with Accelerate BLAS.
  - [ ] Phase 4: Multi-component pipeline archive via manifest (currently
    each component compiled separately).
  - [ ] Phase 5: Classifier-free guidance, proper scheduler, PNG output.
  - [x] **SD Performance (Plan 039)**
    - [x] Phase 1: GroupNorm `_into` + fused activation kernel — `dispatch_group_norm_into`
      and `dispatch_group_norm_activation_into` in hologram base. Fuses activation into
      normalize loop (1 pass instead of 3). `InlineGroupNormActivation` tape dispatch updated.
    - [x] Phase 2: Depthwise Conv2d fast path — `conv2d_depthwise` in hologram base.
      Auto-detected via `ic_per_group == 1 && oc_per_group == 1`. Direct nested loop.
    - [x] Phase 3: Winograd F(2,3) for 3×3 convolutions — `conv2d_winograd_f23` in
      hologram base. Weight transform + input tile transform + batched BLAS GEMM + output
      transform. Gated on `3×3, stride=1, dilation=1, pad=1, ic_per_group >= 16`.
    - [x] Phase 4: CLIP weight quantization — `try_convert_f32_to_lut8` + `QuantStrategy::Q8_0`
      wired in lowering. `--quantize q8_0` works for any model (not just LLMs).
      SD E2E test compiles Q8 CLIP variant and prefers it at runtime.
    - [x] Phase 5: Activation checkpointing — `checkpoint_enabled` flag on `EnumTape`.
      Force-evicts skip-connection buffers after first consumer; recomputes on demand.
      SD E2E test enables checkpointing for VAE decoder.
    - [x] Conv2d + activation epilogue fusion — `FusedConv2dActivation` / `FusedConv2dBiasActivation`
      GraphOps + `InlineConv2dActivation` / `InlineConv2dBiasActivation` TapeKernels in
      hologram base. Fusion pass fires automatically during `hologram::compile()`.
      Eliminates 335MB memory traffic per Conv2d+SiLU block at 512×512.
- [ ] Test with Whisper (encoder-decoder, audio)
- [ ] Fix any op dispatch failures discovered
- [ ] Goal: `hologram-ai compile -m model.onnx` works for top-20 HuggingFace models

### GGUF models
- [x] Verify GGUF TinyLlama causal logit consistency — `gguf_causal_logit_consistency`
  test passes (logits at position P identical for seq=P+1 and seq=P+2)
- [x] LUT-GEMM for Q4_0/Q8_0: `TapeKernel::MatMulLut4`/`MatMulLut8` with
  `WeightCache` and `psumbook` pre-computed partial sums (hologram base)
- [ ] Goal: GGUF generation at >1 tok/s

### Multi-component pipeline archives (Plan 021)
- [x] Phase 1: Generic N-component compilation — `compile_one_component()`,
  `compile_components()`, `LowerPhase::Named`, `OptProfile`, `MemoryPlan::empty()`,
  `ComponentSpec` with role + weight_group. LLM pipeline delegates to
  `compile_components` with 2 specs.
- [x] Phase 2: `MetaSection` with `ComponentDescriptor`, `ComponentRole`,
  `ComponentConnection` — rkyv zero-copy serialization, `EmbeddableSection`,
  `ExecContext` impl. Embedded in pipeline archive via `PipelineWriter::add_section()`.
  LLM pipeline creates 2 descriptors (Prefill + Decode) + 1 KV-cache connection.
  Roundtrip tests pass (LLM + 4-component CALM).
- [x] Phase 3: Weight deduplication — `WeightStore` primitive in hologram-base
  (content-addressable via BLAKE3), `SECTION_WEIGHT_DEDUP` (kind=4),
  `WeightDedupIndex` section keyed by component name. `LoadedPipeline`
  resolves dedup at load time (zero-indirection at runtime). Compiler
  skips weight embedding for duplicate components in same weight group
  (LLM decode no longer duplicates prefill weights).
- [x] Phase 4: `ModelSource::MultiOnnx` + `OptPipeline::generic()` — generic
  multi-ONNX compilation with per-component import, optimization (MVP for
  transformers, generic for others), concretization, and weight group tracking.
  Unlocks CALM, Whisper, Stable Diffusion, any multi-component ONNX model.

---

## Long Term: Production readiness

### Performance
- [x] Fused attention kernel — online softmax (Flash Attention-style) in
  hologram base `attention.rs`, avoids materializing full scores matrix
- [x] Parallel dispatch — rayon `execute_parallel()` with adaptive threshold
  (≥4 instructions per level), excludes shared-state ops (LUT-GEMM, KvCache)
- [x] Memory-mapped weight loading — mmap zero-copy execution with
  `MADV_RANDOM`/`MADV_SEQUENTIAL` page discipline
- [ ] KV cache with variable-length sequences (P5 blocker resolved)
- [ ] Multi-modal output trait (text, images, audio, etc.) — Plan 027 Phase 5
  adds image output via `--output` flag and `ModelKind::ImageGen` detection
- [x] MatMul + Activation fusion — `MatMulActivationFusion` pass creates
  fused AiOp variants. Lowering emits `GraphOp::FusedMatMulActivation` →
  `InlineMatMulActivation` tape kernel. **Wired end-to-end.** TinyLlama
  decode: 20.5 → 39.1 tok/s (Plan 034 Sprint B).
- [x] Concat + MatMul fusion — `ConcatMatMulFusion` pass in hologram-ai
  fuses Concat+MatMul into `ConcatMatMul`. AiOp variant + lowering added.
  Awaiting fused FloatOp kernel in hologram base.
- [ ] F16 compute kernels — deferred to GPU backend (CPU already uses mixed
  precision: F16 storage with F32 compute via dequant in cast.rs)
- [x] Online softmax benchmarked: row-based 2-4x faster standalone; online
  softmax's real win is in fused attention (avoids scores matrix). Current
  split (online in fused attention, row-based standalone) is optimal.
- [x] GPU backend: `ComputeBackend` trait + `BackendSelector` + auto-detection
  in hologram base (Sprint 16 Phases 1-7)
- [x] GPU backend: Metal elementwise (13 MSL kernels), tiled SGEMM matmul,
  softmax, RmsNorm, MTLBuffer-backed arena, zero-copy output path
- [x] GPU backend: Metal async command buffer batching — `Mutex<Option<CommandBuffer>>`
  with `flush()` at level boundaries (hologram base Phase 8.2)
- [x] GPU backend: WebGPU/wgpu compute shader path — cross-platform GPU,
  browser + native (hologram base Phase 8.3)
- [ ] GPU backend: Metal Attention kernel (fused QKV on GPU)
- [ ] GPU backend: CUDA kernel implementations
- [ ] GPU backend: WebGPU command encoder batching + buffer reuse (Phase 8.3d)

### P8: KV Cache Compression & Attention-Gated Decode (Plan 038)
- [ ] Phase 1: Sparse V decode — skip V accumulation for negligible attention weights
  (τ=1e-6) in hologram base attention kernel. Format-agnostic, works with f32/Q8/Q4
  KV cache. Expected +22.8% decode at 32K context, zero quality loss.
- [ ] Phase 2: Wire KV config — hologram base already has `KvCacheConfig` with
  asymmetric K/V bits (`Q8`/`Q4`), boundary-layer protection, and Walsh-Hadamard
  rotation for V. hologram-ai never uses it. Add `--kv-cache`, `--kv-boundary-layers`,
  `--kv-wht` CLI flags + `ModelMetaSection` KV fields + compiler threading.
- [ ] Phase 3: E2E validation — TinyLlama with q8, q4, asymmetric q8:q4, boundary
  layers, WHT, and sparse V. Quality + memory + performance benchmarks.

### P9: CPU Inference Performance — 100-200+ tok/s (Plan 040)

**Target:** 100-200+ effective tok/s for Llama 8B on CPU. WASM supported.
**Branch:** `feat/cpu-inference-perf` in both repos.

#### Tier 1: Wire existing infrastructure (hologram-ai only) — DONE (already implemented)
- [x] 1.1 Wire KV cache quantization — `--kv-cache`/`--kv-boundary-layers`/`--kv-wht`
  CLI flags parse into `KvCacheConfig`, passed to `KvCacheState::with_config()`
- [x] 1.2 Wire epilogue fusion end-to-end — `wrap_graph_op()` emits
  `GraphOp::FusedMatMulActivation` for MatMulRelu/Gelu/Silu (39.1 tok/s result)
- [x] 1.3 Sparse V decode active — `sparse_v: true` in all attention lowering paths

#### ONNX execution correctness (Plan 041)
- [x] **Root cause identified**: variable-length shape resolution. When compiled
  seq_len differs from runtime input length, ops like Reshape/Expand use compiled
  values while Softmax/RmsNorm infer from buffer lengths → shape corruption.
- [x] **Workaround**: compile with `--seq-len N` matching prompt token count.
  TinyLlama at seq_len=24: "The capital of France is Paris." — matches ORT exactly.
  KV cache decode (seq=1) works correctly regardless.
- [ ] **Proper fix**: hologram base `resolve_size()` + lowering should use 0-sentinels
  for all seq-dependent dimensions, resolving consistently at runtime.

#### Tier 2: Compute kernel optimizations (hologram base + hologram-ai)
- [ ] 2.1 Speculative decoding — draft model + batched verification (2-4x throughput)
- [x] 2.2 Flash attention SIMD — NEON `vfmaq_f32` / AVX2 `_mm256_fmadd_ps`
  dot product + V accumulation + online softmax correction. All 8 attention
  tests + 2 SIMD-vs-scalar primitive tests pass.
- [ ] 2.3 AMX/BLAS hybrid — dequant to f16, use Accelerate HGEMM (Apple Silicon)
- [ ] 2.4 AVX-512 VNNI micro-kernels — `_mm512_dpbusd_epi32` (x86_64)
- [x] 2.5 wasm32-simd128 support — `i8x16_swizzle` table lookup in
  `hologram-core/src/view/simd.rs`, wired into `apply_slice()`/`apply_to()`.
  Compiles for `wasm32-unknown-unknown` with `target-feature=+simd128`.

#### Tier 3: Extreme quantization (hologram base + hologram-ai)
- [ ] 3.1 Q2/ternary quantization (BitNet-style) — 2x over Q4, bitmask inner loop
- [ ] 3.2 Continuous batching — amortize weight reads across N concurrent users
- [ ] 3.3 Sliding window attention — O(n*w) instead of O(n²)

#### Tier 4: System-level + archive streaming (hologram base)
- [ ] 4.1 Huge pages for weight buffers
- [ ] 4.2 Compile-time weight reordering (pre-packed layout)
- [ ] 4.3 Multi-level prefetch enhancement
- [ ] 5.1-5.5 Layer-by-layer streaming (lazy constant seeding, per-instruction eviction)

### Precision & Information Theory (Plan 032)
- [x] `SemanticHint` enum on `TensorInfo` — classifies tensors by information
  content (Pixel ~24 bits, Latent ~4 bits, Token ~16 bits, Embedding ~12 bits,
  AttentionWeight ~8 bits, Residual ~16 bits, NormOutput ~12 bits, Position ~8 bits).
  `SemanticPropagation` pass infers hints from op types after fusion passes.
  GGUF importer seeds Token (input_ids) and Embedding (embed output).
  Based on thermodynamic precision framework (Landauer's principle).
- [x] Epilogue fusion — **fully wired end-to-end** (v0.3.0).
  hologram base: `InlineMatMulActivation`, `MatMulLut4Activation`,
  `FusedMatMulBiasActivation`, norm+activation fused kernels.
  hologram-ai: `wrap_graph_op()` emits `FusedMatMulActivation` for
  `MatMulRelu/Gelu/Silu`. Result: 39.1 tok/s TinyLlama decode.
- [ ] Mixed-precision attention — FP8 scores + f32 softmax (future, needs FP8 dtype)
- [ ] Calibration-based precision assignment — measurement-driven, not search (future)

### Architecture
- [x] Simplify post-concretization pipeline — extracted shared
  `post_concretization_repair()` with early convergence detection
- [x] Break up large functions — `compile()` reduced from 257→98 lines by
  extracting `log_post_repair_diagnostics()` (160 lines of diagnostics)
  and `post_concretization_repair()` (100 lines of fixpoint repair)

---

## Complete (this sprint)

### Node-by-node inspector tooling
- [x] `execute_plan_with_intermediates_and_shape_hints` in hologram base
  (**removed** in hologram Sprint 17 — Plans 014+015)
- [x] `tinyllama_node_inspector` conformance test (removed — depended on
  intermediate capture API; node-level debugging now requires probe output nodes)
- [x] `tinyllama_node_divergence_finder` conformance test (removed — same reason)
- [x] `compare_node_by_node.py` Python comparator
- [x] `ort_intermediates.py` ORT intermediate dumper

### Static shape compilation
- [x] `concretize_all_dims` uses `context_length` from model metadata
- [x] `--seq-len` CLI flag on compile command
- [x] `seq_len_override` field on `ModelCompiler`
- [x] `SeqMode::Variable` enabled (was FixedPad-only until P5 blocker resolved)
- [x] `HoloRunner::execute` uses `execute_tape` (EnumTape, Plan 022)
- [x] Post-concretization cleanup uses `Concrete(1)` not `Concrete(0)`

### ResNet-50 / multi-model support
- [x] `OpDecomposition` pass: BatchNorm inference decomposition
  (`scale/sqrt(var+eps)`, bias correction, NCHW broadcast via Unsqueeze)
- [x] ResNet-50 compiles: 582 → 225 nodes after BatchNorm decomposition +
  constant folding, 0 warnings
- [x] Conv2d conformance tests (ORT vs hologram): single Conv2d, stride=2,
  padding variants — all pass
- [x] Mini vision classifier conformance test (Conv+Relu+GlobalAvgPool+Flatten+Gemm)
- [x] `onnx_builder::conv2d()` and `mini_vision_classifier()` test builders
- [x] `position_ids` injection pass for KV cache decode

### Compilation speed (P4)
- [x] `[profile.release]` with `lto = "thin"`, `codegen-units = 1`
- [x] `post_concretization_repair()` — extracted from 3x duplication in
  compiler.rs, with early convergence detection (breaks when dynamic dim
  count stops decreasing)
- [x] `topo_order()` caching on AiGraph via `RefCell<Option<Vec<NodeId>>>`
  with `invalidate_topo_cache()` in all structural mutation passes
- [x] Avoid double LLM compilation — clone pre-concretized graph, re-concretize
  at seq=1 for decode instead of re-importing from disk (~50% savings)
- [x] `Clone` derived for `AiGraph` (cheap: large weights use `Mmap`)

### Compiler fusion passes (P3)
- [x] `SwiGluFusion` pass — fuses `SiLU(gate) * up` → `FusedSwiGLU`, wired
  into MVP pipeline after RmsNormFusion. Eliminates 1 intermediate tensor +
  1 dispatch per transformer layer (LLaMA, Qwen, Mistral, Gemma).
- [x] `AddRmsNormFusion` pass — fuses `Add(x, residual) → RmsNorm(sum, w, eps)`
  into `FusedLayerNormResidual`. Wired into MVP pipeline, lowering maps to
  `FloatOp::AddRmsNorm`. Kernel implemented in hologram base.

### Decode optimizations (P2d)
- [x] `dispatch_float_into` buffer reuse in tape executor
- [x] `WeightCache` wired into `TapeContext` for quantized weight caching
- [x] Level-aware tape execution with `level_offsets`, KvWrite/KvRead enum
  dispatch, and rayon parallel level execution

### Hologram base infrastructure (Sprint 13+)
- [x] Online softmax attention kernel (Flash Attention-style)
- [x] `FusedSwiGLU` kernel (binary elementwise: `silu(a) * b`)
- [x] `AddRmsNorm` kernel in `float_dispatch/norm.rs`
- [x] LUT-GEMM Q4/Q8 (`MatMulLut4`/`MatMulLut8`) with `WeightCache`
- [x] Rayon parallel level execution (`execute_parallel()`)
- [x] Memory-mapped weight loading with madvise page discipline
- [x] Conv2d BLAS sgemm — im2col + Accelerate `cblas_sgemm` on macOS,
  parallel `matmul_k_outer` fallback for non-BLAS platforms (WASM, Linux)
- [x] Parallel float matmul — row-level (M-tile) and batch-level rayon
  parallelism in `dispatch_matmul`, `dispatch_matmul_into`, `dispatch_batched_matmul`
- [x] Liveness-based arena eviction — `consumer_counts` per node, decremented
  per level, `arena.evict()` at zero. Output nodes protected with `u32::MAX`.
  Bounds peak memory to max live activation set (not sum of all outputs).
- [x] Weight index (`WeightIndex`) — per-tensor byte ranges + layer group
  annotations (`derive_layer_group`). `SECTION_WEIGHT_INDEX` embedded in archives.
- [x] Mmap prefetch/release — `HoloLoader::prefetch_range()` (`MADV_WILLNEED`)
  and `release_range()` (`MADV_DONTNEED`). Per-level weight byte ranges
  computed at tape build time for next-level prefetching.
- [x] Pipeline archive auto-detection — `load_auto()` transparently unwraps
  single-component pipeline archives. `LoadedPipeline::into_first_model()`.
- [x] Prewarm guard — skip `prewarm_arena` when total estimate > 2GB
- [x] Alignment safety — `safe_cast_f32` (Cow), empty-buffer guards in
  `get_f32`/`get_f32_unchecked`, misalignment copy in `insert_borrowed_with_elem_size`

### Variable-length prefill (P5)
- [x] hologram base `resolve_size()` applied to legacy `dispatch_float_ctx`
  path (Softmax, RmsNorm, LayerNorm, Reduce*, InstanceNorm)
- [x] `SeqMode::Variable` enabled as default in `run_cmd.rs`
- [x] `mini_transformer_variable_seq_len_runs` test passes (seq=1, 7, 128)

### Fusion pass infrastructure
- [x] `MatMulActivationFusion` pass — fuses MatMul+Relu/Gelu/Silu into
  `MatMulRelu`/`MatMulGelu`/`MatMulSilu` (AiOp variants + lowering added,
  awaiting fused FloatOp kernels in hologram base)
- [x] `ConcatMatMulFusion` pass — fuses Concat+MatMul into `ConcatMatMul`
  (AiOp variant + lowering added, awaiting fused FloatOp kernel)

### Architecture refactoring
- [x] `compile()` reduced from 257→98 lines by extracting
  `log_post_repair_diagnostics()` and `post_concretization_repair()`
- [x] `Clone` derived for `AiGraph` (cheap: Mmap weights not deep-copied)

### Performance benchmarking
- [x] Online softmax benchmark: row-based 2-4x faster standalone; current
  split (online in fused attention, row-based standalone) is optimal
- [x] GGUF causal logit consistency test passes

### Sprint 13 hologram correctness fixes
- [x] **Softmax precision**: restored `f32::exp()` — Sprint 13's `fast_exp()`
  (~1.5% error) compounded across 22 layers producing gibberish
- [x] **Shape-aware GlobalAvgPool**: `infer_nchw` heuristic failed for
  non-standard channel counts. Added `dispatch_global_avg_pool_with_shapes`
- [x] **KV cache overflow**: `read_k_through`/`read_v_through` clamped to
  buffer capacity. `set_advance_override` for padded prefill
- [x] **Clippy clean**: all warnings resolved in both repos

### Root causes found and fixed
- [x] **Shape bug**: seq-like dims set to 0-sentinel → RoPE slices produce `[32,1]`
  instead of `[1,4,5,32]` → 1051 of 1067 nodes fail
- [x] **Attention fusion bugs** (documented, fusion now works):
  - K^T not un-transposed (find_pre_transpose stops at Mul)
  - Scale applied on K path not detected (double-scaling)
  - Output shape `[1,1,5,2048]` instead of `[1,32,5,64]`
  - V tensor uses post-expansion 32 heads but kernel expects 4-head GQA

---

## Previous sprints

See git history for Plans 005-016. Plan 017: performance optimization.
