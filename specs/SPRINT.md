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
  (89x speedup). Variable-length prefill deferred to P4.
- [x] Verified: TinyLlama ONNX pipeline, 29.9 tok/s
- Note: chat models require the user to supply the full chat template in
  `--prompt` (e.g. `<|user|>\nTell me a joke</s>\n<|assistant|>` for
  TinyLlama-Chat). The CLI does not apply templates automatically.

### P2: Decode speed — wire Sprint 13 infrastructure (IN PROGRESS)

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
  for inference with fallback to `execute_plan`

#### P2d: Remaining decode optimizations (requires hologram base — Plan 020)
- [ ] Wire `dispatch_float_into` — buffer reuse, eliminate ~1000 per-op
  allocations per decode token (API exists in hologram base, not wired)
- [ ] Wire `WeightCache` into tape executor — cache deserialized quantized
  weights across dispatches (currently ~5-10x overhead for GGUF; impl exists
  at `kv/weight_cache.rs`, not wired)
- [ ] Level-aware tape execution for KV cache decode path — split tape
  around KvWrite/KvRead ops per level (design only)
- Note: f32 ONNX decode at 13.6 tok/s is near memory bandwidth ceiling
  (4.1 GB weights × ~60 GB/s DDR ≈ 15 tok/s theoretical max). Further
  speedup requires weight quantization — see GGUF models section.

### P3: Compiler fusion passes (Plan 019 — TensorBend-inspired)
- [x] SwiGLU fusion pass — pattern-match `SiLU(gate) * up` into
  `FusedSwiGLU`. Implemented in `swiglu_fusion.rs`, wired into MVP pipeline.
  Eliminates 1 intermediate tensor + 1 dispatch per transformer layer.
- [ ] Add+RMSNorm residual fusion — extend `rmsnorm_fusion.rs` to detect
  `Add(x, residual) → RmsNorm` and emit `FusedAddRmsNorm`. Needs new
  FloatOp + kernel in hologram base (cross-repo).
- [ ] QK-Norm + RoPE + KV-Store pre-attention fusion — fuse 5-7 nodes
  (Split/RmsNorm/RoPE/KvWrite) into extended `Attention` op. Design first,
  implement after tape executor is stable.

### P4: Compilation speed (Plans 017, 020)
- [ ] Release profile with LTO (`codegen-units = 1, lto = "thin"`)
- [ ] Extract shared `post_concretization_repair` (fixpoint code duplicated 3x)
- [ ] Early convergence detection in fixpoint loop (break when dynamic dims
  stop decreasing, saves up to 9 pass invocations)
- [ ] Cache `topo_order` on AiGraph (called ~40 times per compilation)
- [ ] Avoid double LLM compilation (clone AiGraph after MVP, concretize
  twice instead of re-importing from disk)

### P5: Variable-length prefill (deferred from P1 — BLOCKED)
- [ ] Wire `ShapeContextGraph` into `HoloRunner.execute()` — project shapes
  at runtime from actual input dimensions instead of compiled seq_len
- [ ] Add `SeqMode::Variable` — no padding, process actual prompt length
  (variant exists but disabled — see commit 07d6b40)
- [ ] **Blocker:** hologram executor must resolve baked FloatOp params (m/k/n
  in MatMul, size in Softmax) from runtime buffer sizes when they differ
  from compiled values. Requires hologram base changes.
- [ ] Expected: any prompt length without recompilation

---

## Medium Term: Multi-model support

### Any ONNX model
- [x] Test with ResNet-50 (vision, no attention) — **compilation works** (225 nodes
  after BatchNorm decomposition + constant folding). Conv2d conformance tests pass
  (single Conv2d, stride variants, Conv+Relu+GAP+Flatten+Gemm mini classifier).
- [ ] Test with BERT (encoder-only, bidirectional attention)
- [ ] Test with Stable Diffusion UNet (vision + attention + cross-attention)
- [ ] Test with Whisper (encoder-decoder, audio)
- [ ] Fix any op dispatch failures discovered
- [ ] Goal: `hologram-ai compile -m model.onnx` works for top-20 HuggingFace models

### GGUF models
- [ ] Verify GGUF TinyLlama matches ORT (same approach as ONNX)
- [ ] LUT-GEMM for Q4_0: `FloatOp::MatMulQ4` kernel
- [ ] Goal: GGUF generation at >1 tok/s

---

## Long Term: Production readiness

### Performance
- [ ] Fused attention kernel (proven correct via conformance)
- [ ] KV cache with variable-length sequences
- [ ] Parallel dispatch (rayon level scheduling)
- [ ] Memory-mapped weight loading
- [ ] Multi-modal output trait (text, images, audio, etc.)
- [ ] MatMul + Activation fusion (MatMulRelu, MatMulGelu — inline activation
  in matmul output write, avoid intermediate buffer)
- [ ] Concat + MatMul fusion (multi-head output projection — avoid
  materializing concatenated heads buffer)
- [ ] F16 compute kernels (most impactful with GPU backend; CPU uses mixed
  precision with F16 storage, F32 compute)
- [ ] Online softmax: benchmark vs BLAS for decode on macOS, make
  path selection runtime-configurable
- [ ] GPU backend: `trait Kernel` abstraction at tape level (Plan 019)
- [ ] GPU backend: Metal MatMul + Attention kernels
- [ ] GPU backend: CUDA MatMul + Attention kernels
- [ ] GPU backend: WebGPU via wgpu crate

### Architecture
- [ ] Simplify post-concretization pipeline (3 fixpoint iterations → 1)
- [ ] Break up large functions, apply Builder pattern

---

## Complete (this sprint)

### Node-by-node inspector tooling
- [x] `execute_plan_with_intermediates_and_shape_hints` in hologram base
- [x] `tinyllama_node_inspector` conformance test
- [x] `compare_node_by_node.py` Python comparator
- [x] `ort_intermediates.py` ORT intermediate dumper

### Static shape compilation
- [x] `concretize_all_dims` uses `context_length` from model metadata
- [x] `--seq-len` CLI flag on compile command
- [x] `seq_len_override` field on `ModelCompiler`
- [x] `SeqMode::FixedPad` only (removed `Variable` variant)
- [x] `HoloRunner::execute` uses `execute_plan` (no shape walker)
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

### SwiGLU fusion pass
- [x] `SwiGluFusion` pass — fuses `SiLU(gate) * up` → `FusedSwiGLU`, wired
  into MVP pipeline after RmsNormFusion. Eliminates 1 intermediate tensor +
  1 dispatch per transformer layer (LLaMA, Qwen, Mistral, Gemma).

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
