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
- [ ] **BLOCKER: TinyLlama produces gibberish** — hologram base runtime regression:
  1. `build_with_shared_weights` offset bug: embedding Gather reads wrong
     data from shared weight blob → NaN. **Workaround in place**: embed
     weights directly per sub-archive (doubles archive size).
  2. Compute divergence: hologram's logits completely differ from ORT
     (hologram top-1="wig" id=9192 vs ORT top-1=id=529 for BOS input).
     Not caused by hologram-ai compiler changes (verified by reverting all
     changes). Root cause is in hologram base's float dispatch — likely
     introduced in Sprint 21 norm/attention refactoring.
     **Root cause FIXED** in hologram base: two bugs —
     (a) `FloatOp::Transpose` was a no-op passthrough (data never physically
     reordered). Added `InlineTranspose` TapeKernel with baked shapes.
     (b) Batched MatMul flattened all dims into single 2D multiply. Added
     per-batch loop. **Prefill logits now match ORT exactly** (conformance
     tests pass). Decode path produces gibberish — KV cache data is
     correctly populated (verified non-zero) but attention output is
     near-uniform (max logit 3.0 vs ORT's 22.7). Needs investigation
     in how cached K/V is combined with current Q during decode.

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
- [ ] Test with Stable Diffusion UNet (vision + attention + cross-attention)
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
- [ ] Multi-modal output trait (text, images, audio, etc.)
- [x] MatMul + Activation fusion — `MatMulActivationFusion` pass in
  hologram-ai fuses MatMul+Relu/Gelu/Silu into `MatMulRelu`/`MatMulGelu`/
  `MatMulSilu`. AiOp variants + lowering added. Awaiting fused FloatOp
  kernels in hologram base (currently lowers as plain MatMul).
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
