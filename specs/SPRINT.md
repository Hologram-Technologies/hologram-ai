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
- [x] **Disabled AttentionFusion** — the fused SDPA kernel had K^T routing,
  scale placement, and output shape bugs. Individual ops (MatMul, Softmax,
  Add, Transpose) are proven correct by 1156-node conformance comparison.
- [x] **Disabled KvSlotInjection** — depends on AttentionFusion.
- [x] **Node-by-node inspector** — `tinyllama_node_inspector` test dumps all
  intermediate buffers; `compare_node_by_node.py` finds first divergent node.
- [x] **TinyLlama ONNX matches ORT** — all 1156 compared nodes pass.
  Top-5 token predictions identical. Zero failures.

---

## Short Term: Performance (make it fast)

### P1: Release build + smaller seq compilation
- [ ] Add release build CI step
- [ ] Document `--seq-len` workflow: compile at prompt-length seq, not full context
- [ ] Benchmark: measure tok/s at seq=64 vs seq=2048

### P2: Re-enable RoPE + attention fusion (correctly)
- [ ] Write conformance tests for the fused attention kernel covering:
  - Scale on K (pre-MatMul) vs scale on scores (post-MatMul) vs split scale (Q and K)
  - GQA with expanded K/V (32 heads) vs raw K/V (4 heads) + group mapping
  - K^T input (transposed layout) vs K input (un-transposed)
  - Causal mask via flag vs explicit additive mask tensor
- [ ] Fix `AttentionFusion` to handle all ONNX SDPA export variants
- [ ] Re-enable `AttentionFusion` with conformance gate (only fuse when test passes)
- [ ] Verify: fused TinyLlama still matches ORT

### P3: KV cache for autoregressive generation
- [ ] Compile two graphs: prefill (seq=context_len) + decode (seq=1)
- [ ] Re-enable `KvSlotInjection` (requires working AttentionFusion)
- [ ] Prefill: run full prompt through prefill graph, cache K/V
- [ ] Decode: run single-token decode graph, read cached K/V
- [ ] Verify: multi-token generation produces coherent English

---

## Medium Term: Multi-model support

### Any ONNX model
- [x] Test with ResNet-50 (vision, no attention) — **compilation works** (225 nodes
  after BatchNorm decomposition + constant folding). Conv2d conformance tests pass
  (single Conv2d, stride variants). Execution blocked on naive Conv2d being too slow
  at full 224×224 resolution for practical e2e testing.
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

### Architecture
- [ ] Remove dead `ShapeContextGraph` walker code (replaced by static shapes)
- [ ] Remove `ShapeRecipeSection` / `ParamRecipe` (no more deferred dims)
- [ ] Simplify post-concretization pipeline (3 fixpoint iterations → 1)
- [ ] Break up large functions, apply Builder pattern
- [ ] Clean up clippy warnings

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

### Root causes found and fixed
- [x] **Shape bug**: seq-like dims set to 0-sentinel → RoPE slices produce `[32,1]`
  instead of `[1,4,5,32]` → 1051 of 1067 nodes fail
- [x] **Attention fusion bugs** (disabled, not fixed):
  - K^T not un-transposed (find_pre_transpose stops at Mul)
  - Scale applied on K path not detected (double-scaling)
  - Output shape `[1,1,5,2048]` instead of `[1,32,5,64]`
  - V tensor uses post-expansion 32 heads but kernel expects 4-head GQA

---

## Previous sprints

See git history for Plans 005-013.
