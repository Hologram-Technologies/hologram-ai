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
  when hologram base wires kernel support. **→ Plan 074 Patterns 1+3.**

### P4: Compilation speed (Plans 017, 020, 073)
- [x] Release profile with LTO (`codegen-units = 1, lto = "thin"`)
- [x] Extract shared `post_concretization_repair` (was duplicated 3x in
  compiler.rs, now a single function with early convergence detection)
- [x] Early convergence detection in fixpoint loop (break when dynamic dims
  stop decreasing, saves up to 9 pass invocations)
- [x] Cache `topo_order` on AiGraph (was called ~40 times per compilation,
  each building 3 HashMaps; now cached with `RefCell` + invalidation)
- [x] Avoid double LLM compilation (clone AiGraph after MVP, concretize
  twice instead of re-importing from disk — ~50% LLM compile time savings)
- [ ] **Plan 073 — Compilation Speed Optimization (next wave)**
  - [x] Phase 1: Tracing instrumentation on all pipeline stages —
    `info_span!` on import, optimize, weight collection, per-component
    prepare, lower, hologram_compile, and per-opt-pass in pipeline.rs
  - [x] Phase 2: Parallelize LLM 3-graph compilation (prefill/decode/verify
    via `std::thread::scope`) — preparation + compilation run in parallel.
    `AiGraph.topo_cache` changed from `RefCell` to `Mutex` for `Sync`.
    `prepare_llm_component()` extracted as reusable helper.
  - [x] Phase 3: Pass skip predicates (`should_run` on `Pass` trait) —
    implemented for KvSlotInjection, AddRmsNormFusion, SwiGluFusion,
    NormProjectionFusion, SwiGluProjectionFusion, PositionIdsInjection
  - [x] Phase 4: Parallel weight quantization — `rayon::par_iter` in chunks
    of 8 before node lowering loop. Pre-quantizes all Q4-eligible weights.
  - [x] Phase 5: `AiParam::Inline` data → `Arc<Vec<u8>>` — graph clones
    are now near-free (reference-counted, not deep-copied)

### P5: Variable-length prefill (active — Plans 045 + 058)
- [x] **Blocker resolved:** hologram base now applies `resolve_size()` in
  both tape executor AND legacy `dispatch_float_ctx` paths. Softmax, RmsNorm,
  LayerNorm, Reduce*, InstanceNorm all resolve from runtime buffer sizes.
  MatMul uses `infer_matmul_k()` to re-derive k from buffers.
- [x] `mini_transformer_variable_seq_len_runs` test passes (seq=1, 7, 128)
- [x] `SeqMode::Variable` enabled as default in `run_cmd.rs`
- [x] Wire `ShapeContextGraph` into `HoloRunner.execute()` — `resolve_shapes()`
  calls `walk_shape_context()` and passes result to `execute_tape_with_shapes()`.
  hologram base `execute_direct` populates `input_metas` from `shape_overrides`.
- [x] `execute_tape_with_kv_shapes_cached` — combines KV cache + shape overrides
  + persistent weight cache in single execution path.
- [x] Prune dead ShapeContextGraph entries after fusion (`retain_live_nodes`)
- [x] Persistent WeightCache in CLI `run` — eliminate per-step Q4 rkyv overhead
- [x] `concretize_all_dims` returns `seq_dim_positions: HashSet<(TensorId, axis)>`
  identifying seq-dependent dims before concretization (infrastructure for Plan 045)
- [x] **Post-fusion shape projections (Plan 059 Phases 1-2)** — `float_output_shape()`
  in hologram-core covers all FloatOp variants. `compute_post_fusion_shapes()` in
  `emit_stage()` walks post-fusion graph and produces 100% node_shapes coverage.
  **Finding:** Shape metadata overrides can't fix baked op parameters (MatMul m/k/n,
  Softmax size). Shape overrides tell the *next* op what shape a buffer has, but
  the *current* op still computes with its baked parameter values.
- [x] **0-sentinel op parameters (Plan 045)** — `zero_seq_dims_for_lowering()`
  now called in all compile paths including `compile_multi_onnx` (was missing).
  Fixed 75× UNet regression (>10 min → ~20s).
- [x] **Metal MatMul dispatch** — wired Metal SGEMM through tape executor for
  InlineMatMul, InlineMatMulActivation, InlineMatMulBiasActivation. 15× kernel
  speedup (8.6s → 0.57s). Conv2d 1×1 also routes through Metal.
- [x] **Online softmax for large attention** — SD self-attention (4096×4096)
  uses O(seq) online softmax instead of materializing O(seq²) score matrix.
- [x] **Streaming pipeline archives** — `PipelineWriter::build_to_file()` streams
  weights from disk. All archives are now proper pipeline format (SECTION_PIPELINE).
- [x] **GPU buffer chaining (Plan 065)** — `GpuBuffer` + `GpuInput` abstraction,
  Metal kernel coverage for all SD UNet ops, GPU-to-GPU op chaining. CPU kernel
  time reduced to 587ms but hybrid GPU/CPU execution creates 24s sync overhead.
- [ ] **ComputeBackend + ComputeMemory rewrite (Plan 067)** — replaces Plans
  065/066. New `hologram-backend` crate with device-native execution: all data
  lives on one device, all computation happens there, no CPU↔GPU transfers.
  Every backend implements full UOR computational model (ring ops, float ops,
  data movement). Target: 2-5s for SD UNet.
  - [ ] `zero_seq_dims_for_lowering()` — new function in compiler.rs
    - Zero `known_i64_values[axis]` on Reshape/Flatten output tensors
    - Zero seq dims on Expand shape input tensors
    - Set seq axis to `Dim::Dynamic` on MatMul/norm input tensor shapes
  - [ ] Remove prompt-length guard in `resolve_seq_mode()`
  - [ ] Conformance test (seq=24, prompts 10/18/24/36)
  **Workaround:** Compile at model's full context length (default). Variable-
  length works correctly for any prompt <= compiled seq_len.
  - [ ] Conformance test: compile at seq=24, run with 18 and 36 token prompts

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
- [x] **Model-size-aware quantization (Plan 074 Pattern 6)** — compiler auto-
  downgrades Q4 → f32 for models < 750M params. Adaptive Q4 error threshold
  scaled by `sqrt(total_params / 1B)`. Q8 uniform BLAS path added for small
  model fallback (partial coverage). TinyLlama 39 tok/s Q4, Qwen2 5.3 tok/s Q8.
- [ ] **Post-lowering quantization pass (Plan 076)** — refactor quantization
  from 9+ hooks scattered across builder.rs into a single graph pass that runs
  between `lower()` and `compile()`. Walks the Graph, converts eligible
  `Float(MatMul)` → `MatMulLut4/8` with pre-serialized weights. Enables:
  full Q8 coverage for small models (→ 10+ tok/s), per-layer sensitivity,
  mixed Q4/Q8, and future GPTQ/AWQ algorithms.

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
- [ ] **Vision-language model support (Plan 070 — Falcon-Perception analysis)**
  - [x] ONNX export of Falcon-Perception-300M — 3621 nodes, 22 op types, 957 MB.
    All 22 ops now supported. Export uses SDPA decomposition (FlexAttention
    not ONNX-exportable). Compilation succeeds (1.1 GB archive).
  - [x] `Trilu` op support — AiOp variant + ONNX mapping (`upper` attr) +
    constant folding in DataPropagation (materializes upper/lower triangular
    matrix from constant input). Lowering falls through to Identity (folded).
    Used for causal mask: `ConstantOfShape → Trilu → Unsqueeze → Cast → Where → Softmax`.
  - **hologram-ai changes (VLM architecture):**
    - [ ] `AttentionMaskKind` enum on AiOp — replace `causal: bool` with `Full |
      Causal | HybridCausalPrefix | SpatialWindow | BlockSparse` + lowering
    - [ ] `RoPE2D` AiOp variant + lowering (split temporal/spatial halves)
    - [ ] `VisionTokenConfig` in tokenizer — image placeholder, coord/size/seg
      token IDs for multimodal encoding
    - [ ] New DimVars: `IMAGE_HEIGHT`, `IMAGE_WIDTH`, `NUM_IMAGES`, `PATCH_DIM`
    - [ ] `SquaredReluGateFusion` pass (pattern: `relu(gate)² * up`)
    - [ ] SafeTensors weight loader crate (medium-term, for HF ecosystem)
    - [ ] `MultimodalTokenizer` trait + `encode_with_images()` API
  - **hologram base changes:**
    - [ ] `AttentionMaskKind` kernel support in `dispatch_attention` — hybrid
      bidirectional prefix + causal suffix, spatial window patterns
    - [ ] `FloatOp::RoPE2D` variant + kernel (2D spatial rotation with learned freqs)
    - [ ] `FloatOp::FusedSquaredReluGate` kernel (like FusedSwiGLU but relu²)
    - [ ] Sink token gating (`sigmoid(LSE - sink_param)`) fused into attention output
    - [ ] Paged KV cache in hologram-exec (Plan 016) — virtual page tables,
      LIFO free-page stack, `KvPagedWrite`/`KvPagedRead` dispatch
    - [ ] AnyUp-style windowed cross-attention kernel (P3, segmentation)
- [x] **Qwen2-0.5B cross-family validation (Plan 074)**
  - [x] Download + compile + decode Qwen2-0.5B ONNX (first non-LLaMA LLM)
  - [x] ByteLevel BPE tokenizer (BBPE, 151K vocab) — hologram-ai + hologram base
  - [x] config.json companion file reading for arch/vocab metadata
  - [x] Position IDs (PositionIdsInjection), arch detection (`"qwen2"`)
  - [x] Post-embedding RMSNorm verified correct (49 fusions)
  - [x] InlineTranspose 0-sentinel fix for M>1 prefill (hologram base)
  - [x] Vocab dimension inference from output buffer
  - [x] Model-size-aware quantization (auto Q4→Q8 for <750M params)
  - [x] HOLOGRAM_DUMP_DIR diagnostic infrastructure
  - [x] qwen2_e2e.rs test suite (compile, run, variable seq_len, logit comparison)
  - [ ] **Plan 076: Post-lowering quantization pass** — full Q8 coverage for
    10+ tok/s (currently 5.3 at partial coverage)
- [ ] **Architectural patterns from Qwen (Plan 074) — future**
  - [ ] Pattern 1: Fused RoPE + context scaling (NTK/YaRN for 128K+ context)
  - [ ] Pattern 2: LogN attention scaling (`log(n)/log(n_train)` for long ctx)
  - [ ] Pattern 3: QK-Norm (RMSNorm on Q/K before attention)
  - [ ] Pattern 4: KV cache quantization as first-class feature
  - [ ] Pattern 5: SwiGLU clamping for numerical stability at Q4
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
- [ ] GPU backend: CUDA backend (Plan 072) — `CudaMemory` + `CudaBackend` via
    cudarc 0.19 (dynamic loading, no toolkit at compile time). Pre-compiled PTX
    kernels matching Metal parity (~30 kernels). Hardware detection for SM 7.5-9.0.
    Future: tensor-core matmul, FlashAttention, INT8 (informed by kaio patterns).
- [ ] GPU backend: WebGPU command encoder batching + buffer reuse (Phase 8.3d)

### P8: KV Cache Compression & Attention-Gated Decode (Plans 038, 074B)
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

#### Path to 100-200 tok/s (Plan 042) — 5 phases

**Phase 1: Archive dedup + MatMulActivationFusion (→ ~5 tok/s)** — DONE
- [x] 1A. Archive weight dedup — skip shared blob for same-weight-group LLM
  pipeline. F32 archive: 9.4 → 4.4 GB. Q4: 9.8 → 5.0 GB (f32 originals still
  embedded alongside Q4 constants — further optimization needed).
- [x] 1B. MatMulActivationFusion pass — pattern-matches MatMul → SiLU/GeLU/ReLU
  into fused AiOp variants. Wired into MVP pipeline. No matches on TinyLlama
  (uses SwiGLU), but ready for GPT-2/BERT (GeLU). Lowering + kernel already
  handle fused variants.

**Phase 2: Single-path executor + zero-copy (→ 40 tok/s)** — DONE
- [x] 2A. `execute_direct` — single execution path, pre-allocated buffers, one
  dispatch match per instruction. Eliminated execute_inner, dispatch_kernel_par,
  execute_parallel. tape.rs: 5,829 → 4,188 lines (-28%).
- [x] 2B. Zero-copy input gathering — unsafe raw pointer aliasing for input slots,
  mutable output slot. No per-instruction clone. 370ms → 21.5ms/step.
- [x] 2C. Q4 BLAS hybrid in single path — `dispatch_lut_gemm_4` dequants centroids
  → f32 (cached in WeightCache), then cblas_sgemm. 870ms → 22ms/step.
- **Result: 2.4 → 40.9 tok/s (17x speedup), TinyLlama f32 + Q4, M4 Max.**

**Phase 3: Path to 100+ tok/s** — 3 feature branches, execute in order

3A. **Speculative decoding (Plan 043)** → 80-120 tok/s effective
- Branch: `feat/speculative-decoding`
- Draft model generates N candidates, target verifies in one batch
- 2-3x effective throughput at 40 tok/s base
- New: `crates/hologram-ai/src/speculative.rs`, CLI `--draft-model`
- [ ] SpeculativeDecoder struct + acceptance/rejection
- [ ] CLI wiring + draft model validation
- [ ] Tests: effective tok/s > 2x base

3B. **Layer-level parallelism (Plan 044)** → 60-80 tok/s
- Branch: `feat/layer-parallelism`
- Parallel attention heads (32 heads on 12 cores via rayon)
- Parallel FFN gate+up projections
- [ ] Parallel head loop in `dispatch_attention`
- [ ] Parallel group dispatch in `execute_direct`
- [ ] Tests: output matches sequential, tok/s improvement

3C. **Variable-length execution fix (Plan 045)** → any prompt length
- Branch: `feat/variable-length-fix`
- 0-sentinels for seq-dependent dims in lowering
- Wire ShapeContextGraph into `execute_direct`
- [ ] Compile at seq=64, run with 24 tokens → correct
- [ ] Non-LLM models unaffected

3D. **Archive compression + Q4 size** (backlog)
- Wire hologram-compression for --compress on compile
- Strip f32 originals from Q4 archive (constant offset remapping)
- Q4 archive: 5 GB → ~0.5 GB → ~0.2 GB compressed

#### Plan 051: Full Byte-Domain Q4 MatMul — 60+ tok/s
- Branch: `feat/hologram-lut-q4`
- **Core idea:** Quantize activations to int8 at runtime, making the entire
  Q4 matmul inner loop pure integer (int8×int8→int16→int32). f32 conversion
  happens ONCE per output element instead of per K-row.
- Strategy A: `vmull_s8` (int8×int8→int16) + `vaddw_s16` (int16→int32 accumulate)
- 18 NEON ops per 32 columns vs current 36 → 2x inner-loop reduction
- K-unroll by 4 for ILP → projected 2.6x total speedup
- [ ] Phase 1: `quantize_activation_row` helper (NEON-vectorized)
- [ ] Phase 2: `lut_gemm_4bit_neon_int8` kernel (replaces nibbleview)
- [ ] Phase 3: K-unroll by 4 + N-chunk 64
- [ ] Phase 4: Profile vs BLAS, lower/remove hybrid threshold
- [ ] Phase 5: Update scalar fallback + WASM path
- **Result: 43-44 tok/s — near Q4 bandwidth ceiling on Apple Silicon**
- [x] Phase 1-2: int8 activation quantization + NEON vmull/vaddw kernel
  - Pure LUT: 12.5 → 20 tok/s (1.6x from int8 activations)
  - Compile-time int8 centroid table, K-unrolled by 4 for ILP
- [x] Phase 3: Unified LUT+BLAS pipeline (no threshold branching)
  - LUT handles Q4 storage + dequant (hologram's core value)
  - BLAS/AMX handles matmul compute (hardware acceleration)
  - Single path: LUT dequant → BLAS sgemm on macOS; int8 LUT kernel on non-BLAS
- [x] NEON f32 vecmat for M=1 — **tested, AMX faster** (10 vs 43 tok/s)
  - Apple AMX hardware outperforms software NEON even for M=1 vecmat
  - Kept as fallback for non-BLAS platforms (WASM, Linux without Accelerate)
- [x] Profiling infrastructure (`HOLOGRAM_PROFILE=1`, per-kernel-type timing)
- [x] SharedInputProjectionFusion pass (Plan 052)
  - QKV: 3 MatMuls → 1 MatMul + 3 Slices (22 fusions fire on TinyLlama)
  - Gate+Up: 2 MatMuls → 1 MatMul + 2 Slices (22 fusions fire)
  - **Tested, currently slower on Apple Silicon** — AMX handles small matmuls
    efficiently; fused larger matmul + Slice overhead is net slower
  - Disabled for now; useful on non-AMX platforms where BLAS overhead is higher
  - Gate+Up fusion hits rkyv serialization overflow for large Q4 weights (11264 cols)
- **Profiling findings (21ms/step at 43 tok/s):**
  - MatMulLut4 (Q4 dequant→BLAS): 10ms/48% (45 calls)
  - MatMul f32 (down_proj+attn): 9.4ms/45% (110 calls)
  - Transpose: 0.3ms/1.5% (89 calls)
  - Everything else: 1.3ms/5%
  - **96% of time is in matmul** — AMX is at hardware throughput limit
- **Path to 60+ tok/s requires reducing data per step (Tier 3):**
  - Q2/ternary quantization — half the weight reads (2x bandwidth ceiling)
  - Speculative decoding — multiple effective tokens per forward pass
  - Sliding window attention — reduce KV cache reads at long context

#### Deep Decode Fusion (Plan 054) — eliminate intermediate buffers

Today's fusions are shallow (2-op chains). Deep fusion chains 3-4 ops into
single dispatches, eliminating all intermediate buffers through transformer
blocks. Decode-only (M=1); prefill falls back to separate BLAS calls. The
general fusion rule: if an op's output has exactly one consumer and no global
reduction, fuse them.

**Wave 1: Core deep fusions (hologram base + hologram-ai) — DONE**
- [x] `FloatOp::NormProjectionGemv` / `AddNormProjectionGemv` / `SwiGluProjectionGemv`
- [x] TapeKernel variants + tape_builder + dispatch_kernel wiring
- [x] M=1 fast path: normalize into caller Vec, skip arena allocation
- [x] `FusedNormProjection` + `FusedSwiGluProjection` AiOp variants
- [x] `NormProjectionFusion` (44 fusions on TinyLlama ONNX) — multi-output lowering:
  1 norm node + N MatMul nodes sharing norm output. No weight concatenation.
- [x] `SwiGluProjectionFusion` (22 fusions on TinyLlama ONNX)
- [x] Lowering + pipeline ordering
- [x] ShapeContextGraph serialization panic fixed (graceful fallback)
- [x] Fusion metrics: 6 → 2 nodes per FFN layer (67% reduction in unit tests)
- [x] TinyLlama ONNX E2E: compiles + generates "The capital of France is Paris."

**Wave 2: Shape chain elimination + GEMM absorption — DONE**
- [x] GQA Expand chain — already eliminated by DeadNodeElimination after AttentionFusion
- [x] `TransposeMatMulFusion` — absorb Transpose(swap-last-2) → MatMul into Gemm trans_b
- [x] `ScalarAbsorption` — fold MatMul → Mul(scalar) into Gemm alpha
- [x] Expand → Identity lowering (zero-copy pass-through instead of Reshape)
- [ ] Q/K/V Reshape+Transpose absorption into attention (future)

**Wave 3: Extended patterns (all model types)**
- [ ] Embed + Norm fusion (post-embedding normalization: Qwen, Gemma)
- [ ] Final Norm + LM Head fusion (single-consumer logit projection)
- [ ] LayerNorm + Projection (BERT, GPT-2, CLIP encoder models)
- [ ] Attention → Residual Add fusion (Plan 039 #5)
- [ ] Conv2d + GroupNorm + Activation chain (SD UNet)
- [ ] LUT4 variants (dequant preamble first, inline if profiling demands)

**Wave 4: General rule-based fusion walker**
- [ ] Replace individual fusion passes with single configurable walker
- [ ] Rules: elementwise fuse freely, norm fuses with Add or projection,
  max 1 GEMM per group, Softmax/Attention are barriers

#### Runtime Performance (Plan 039 — hologram base)

**Phase 1: Multi-core unlock (highest impact — 2-3x decode throughput)**
- [ ] Fix `dispatch_kernel_par` bug — missing arms for MatMulLut4/8, Conv2d*
  in parallel dispatch. Blocks all multi-core quantized inference. (Plan 039 #1)
- [ ] N-dimension parallelism for M=1 GEMM — partition n_tiles across rayon
  threads. Each thread writes non-overlapping output columns. (Plan 039 #2)

**Phase 2: Memory bandwidth**
- [x] Multi-level weight prefetch — madvise(WILLNEED/DONTNEED) wired in execute_direct
- [ ] Shared B-panel packing — restructure GEMM loop (Plan 039 #6)

**Phase 3: Fusion gaps (hologram base graph-level)**
- [x] AddRmsNorm + Activation fusion — already in hologram base
- [ ] SwiGLU fusion from Split → Silu → Mul pattern (Plan 039 #7)
- [ ] Attention + Residual Add fusion (Plan 039 #5)

**Phase 4: Memory & platform**
- [ ] Wire workspace buffer reuse (Plan 039 #8)
- [x] Adaptive sparse_v threshold — HOLOGRAM_SPARSE_V_THRESHOLD env var
- [ ] wasm32 SIMD128 micro-kernels (Plan 039 #11)

#### Path to 60+ tok/s (Plans 055-057) — RESULTS

**Achieved: 43.0 tok/s** on ONNX TinyLlama `--quantize q4_0` (from 2.7 f32 baseline).
This is the AMX hardware ceiling for TinyLlama 1.1B on Apple Silicon CPU.
See Plan 057 for full session summary.

**Phase 1: Per-row Q4 + early-quant + prewarm — DONE (43 tok/s)**
- [x] Per-row symmetric linear Q4 quantization (replaced global k-means)
- [x] Early weight quantization at param registration (before graph optimization)
- [x] Dequant cache prewarm at model load (zero warmup penalty)
- [x] SwiGluProjectionGemv decompose to FusedSwiGLU + MatMulLut4 for Q4 weights
- [x] `feeds_attention` fix: check K == hidden_size (not just N)
- Steady-state: 21ms/step, 93% in BLAS sgemm (AMX hardware ceiling)

**Phase 2: Speculative decoding infrastructure — DONE (needs draft model)**
- [x] 3-component pipeline: prefill (seq=N) + decode (seq=1) + verify (seq=8)
- [x] HoloRunner 3-tape loading with shared WeightCache + prewarm
- [x] Batch verify via execute_verify() — ONE forward pass for N tokens
- [x] `--speculative` + `--draft-steps N` CLI flags
- [x] KvCacheState.truncate_to() for draft rollback
- ⚠ Self-speculative: 0% acceptance (M=1 vs M=N numerical divergence)
- Needs: separate draft model or exact numerical agreement

**Phase 3: Q2 quantization — DONE (infrastructure only)**
- [x] Full Q2 stack: QuantizedWeights2, quantize_2bit, lut_gemm_2bit, MatMulLut2
- [x] `--quantize q2_0` CLI flag + compiler integration
- ⚠ Pure integer Q2 kernel slower than AMX BLAS (tested: 400ms vs 21ms)
- ⚠ Bandwidth reduction doesn't help when AMX uses cached f32 dequant

**What doesn't work for 60+ on CPU (Apple Silicon):**
- Any software kernel (NEON/scalar/int8) is 8-20x slower than AMX BLAS
- Bandwidth reduction doesn't help (AMX uses cached f32 regardless of Q level)
- Self-speculative fails (numerical divergence between M=1 and M=N)
- Level-parallel: BLAS already uses multi-core AMX internally

**Viable paths to 60+ (future work):**
- Separate draft model for speculative decoding (60-80 effective tok/s)
- Metal GPU (100+ tok/s, deferred per user preference)
- Smaller models (Phi-2, Gemma-2B — same architecture, fewer parameters)

#### Tier 2: Compute kernel optimizations (hologram base + hologram-ai)
- [ ] 2.1 Speculative decoding — see Plan 055 Phase 2 above
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

### Host-facing metadata (Plan 060)
Bake host-facing fields (chat template, sampling defaults, port names,
model card) into a new `HostMetaSection`, sibling to `ModelMetaSection`.
Closes the documented papercut where chat models require users to type
the full chat template into `--prompt` on every invocation.
- [x] Phase 1 (hologram base): `host_meta.rs` rkyv struct + `SECTION_HOST_META = 0x1003` + 6 tests
- [x] Phase 2: TOML `Manifest` `[host]` table + `--prompt-template`/`--chat-template`/
  `--temperature`/`--top-k`/`--top-p`/`--repetition-penalty`/`--stop`/`--author`/
  `--license`/`--source-url`/`--tag` flags, `build_host_meta()` precedence helper
  (manifest > CLI > imported), 8 unit tests covering every precedence path
- [x] Phase 3: `hologram inspect` summary prints "Host metadata" block via new
  `LoadedPlan::host_meta_from_bytes()`; `sections` detail names kind 4099
  as `host_meta`. E2E verified against `mini_transformer.onnx`: compile with
  flags → write → reload → print, all fields round-trip
- [ ] Phase 4 **DEFERRED**: no GGUF importer exists in the current tree
  (`ModelSource` only has ONNX variants). Hook is in place —
  `build_host_meta()` accepts `imported_chat_template: Option<String>`,
  currently `None`. When a GGUF importer lands it populates that parameter
  and the precedence rules apply unchanged.
- [ ] Phase 5 (follow-up, separate plan in hologram base): `run_cmd.rs` reads
  `HostMetaSection` and applies `chat_template` automatically
- [ ] Backlog: schemars-generated `specs/schemas/manifest.schema.json`

### Architecture
- [x] Simplify post-concretization pipeline — extracted shared
  `post_concretization_repair()` with early convergence detection
- [x] Break up large functions — `compile()` reduced from 257→98 lines by
  extracting `log_post_repair_diagnostics()` (160 lines of diagnostics)
  and `post_concretization_repair()` (100 lines of fixpoint repair)

---

## Complete (this sprint)

### SDXL ONNX compilation (Plans 061, 064)
- [x] ConstantOfShape op: AiOp variant + ONNX mapping + DataProp materialization
- [x] Dynamic Slice resolution: reads known_i64_values from DataProp when import-time resolution fails
- [x] Mmap constant reading: small constants from external data files read eagerly for Slice params
- [x] All 4 SDXL components compile: text_encoder, text_encoder_2, UNet (10.3 GB), vae_decoder
- [ ] **Streaming compilation (Plan 064)** — compile SDXL UNet within 2 GB RSS
  - [ ] Phase 1: Fix Q4 accumulation leak (`.get()` → `.remove()`)
  - [ ] Phase 2: Spill Q4 constants to temp file
  - [ ] Phase 3: Streaming archive write
  - [ ] Phase 4: OS page cache advise

### OutputBuffer + Mmap eviction (Plan 062)
- [x] OutputBuffer enum (Heap/Arena/Mmap) replaces &mut Vec<u8> in 44 kernels
- [x] Mmap eviction via munmap — RSS drops during SD UNet execution
- [x] Lazy allocation: buffers start empty, self-promote to Mmap on resize ≥ 256 KiB
- [x] heap_only_eviction flag for Conv2d-heavy models (VAE)
- [x] SD UNet live working set: 37 MiB (was 47 GiB)
- [x] TinyLlama: 42.5 tok/s unaffected

### Cleanup: GGUF removal (Plan 061 Stage 0)
- [x] Removed `hologram-ai-gguf` crate, workspace member, and all consumers
  (compiler.rs `ModelSource::GgufPath`, lib.rs re-export, cli.rs `inspect_gguf`,
  validate.rs `.gguf` branch, download/mod.rs `DownloadFormat::Gguf` +
  `download_gguf` + `convert_to_gguf`, tinyllama_e2e.rs gguf tests,
  mini_fixture.rs `gguf_causal_logit_consistency`, benches/inference.rs
  `gguf_holo_path` benchmarks).
- [x] **TinyLlama ONNX baseline gate cleared post-removal: 40.1 tok/s coherent**
  (`The capital of France is Paris.`) on `hologram-ai run` with
  `--temperature 0.0` and chat-formatted prompt.
- [x] Fixed UTF-8 char-boundary panic in `run_cmd.rs:517` streaming path —
  multi-byte codepoints (emoji) spanning two BPE tokens no longer panic.
- [x] Removed dead `nodes`/`inits` locals from `kv_expand_tinyllama_dims_matches_ort`
  test in conformance suite.

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
