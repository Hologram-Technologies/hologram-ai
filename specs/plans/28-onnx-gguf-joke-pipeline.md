# Plan: ONNX + GGUF Compilation to Joke Output via Hologram

## Goal
Compile ONNX and GGUF models through hologram IR and execute a prompt-to-joke flow, keeping `hologram-ai-onnx` and `hologram-ai-gguf` as thin mappers and pushing fused/low-level ops into `/hologram`.

## Current Status
- **Phase 1 (Constant folding)**: in progress (ONNX focus; IR passes moved into compiler, CSE fixed + re-enabled)
- **Runtime debugging**: in progress (flattened MatMul weights handled; reshape kernel fixed for aliasing; tokenizer discovery updated; T5 run now completes for 1–5 tokens but still slow for 50; run-pipeline infers seq_len and slices logits; KV cache bootstrapping added but needs KV-enabled decoder compilation)
- **KV decoder compile**: merged decoder now compiles after treating past/cache dims as dynamic; release run currently aborts (Signal 6) and needs investigation.
- **Phase 3 (Fused ops + builders)**: pending
- **Phase 2 (Quant/dequant in hologram)**: pending

## Phase 1: Constant folding in hologram (Start Here, ONNX)
1. Run hologram IR optimization passes after ONNX → IR translation. **Done** (moved into `hologram` compiler pipeline).
2. Fix CSE rewiring and re-enable in compiler IR pass pipeline. **Done**.
3. Expand `/hologram/crates/ir/src/passes/constant_fold.rs` to cover ONNX-side folding currently done in `hologram-ai-onnx`.
4. Remove per-translator `constant_fold()` hooks in `crates/hologram-ai-onnx` and rely on IR pass.
5. Fix MatMul conversion to infer `N` from flattened weights (compiler `from_ir` + IR shape inference). **In progress** (code + tests added; test run pending).

## Runtime Performance (T5 ONNX)
1. Infer `max_length` from compiled plan metadata instead of hardcoded 512 in `run-pipeline`. **In progress** (code updated; test run pending).
2. Avoid copying full logits tensor each generation step (use slice). **In progress** (code updated; test run pending).
3. Add KV-cache plumbing in decoder generation loop (enable when decoder exposes past inputs). **In progress** (runtime support + initial zero-cache bootstrap added; requires KV-enabled decoder export with If).
4. Add fused last-token argmax kernel/op in hologram and wire compiler/backend support. **Done** (IR + compiler + kernel added; runtime uses ArgMaxLastToken runner).
5. Keep a baseline benchmark for `max_tokens=1` (capture wall time). **Done** (`scripts/bench-t5-max1.sh`).
## Phase 3: Fused ops + builders (Second)
1. Extend hologram IR to support fused attention features needed by GGUF path:
   - Add a fused op (or extend `NodeOp::Attention`) to cover RoPE and GQA.
   - Ensure compiler lowering and kernel mapping can accept the fused op.
2. Add/extend kernel implementations to execute fused attention in CPU backend.
3. Update `crates/hologram-ai-common` transformer builders to emit the fused op instead of decomposed subgraphs.
4. Validate GGUF compilation pipeline still emits valid .holo + weights and runs end-to-end for a prompt.

## Phase 2: GGUF quant/dequant in hologram (Third)
1. Add IR ops for GGML quant formats (Q4_0/Q4_K/Q8_0, etc.) or a generic dequant op.
2. Implement CPU kernels for dequantization in `/hologram/crates/backend` and map in compiler.
3. Update `crates/hologram-ai-gguf` to emit quantized constants plus dequant ops instead of CPU-side dequantization.
4. Validate GGUF inference end-to-end using hologram runtime.

## Notes
- Keep `hologram-ai-onnx` and `hologram-ai-gguf` as thin translators to hologram IR.
- New fused/low-level ops belong in `/hologram`.
- If AI-specific fused ops are needed, place them in `hologram-ai-common` (or a new `hologram-ai-operations` crate) and map them to hologram IR.
- Benchmark command (max-tokens=1): `scripts/bench-t5-max1.sh`
- ONNX `If` nodes are now lowered to `Where` (both branches computed) to unblock KV-enabled decoder compilation.
- Past/cache symbolic dims now resolve to dynamic to avoid broadcast errors in KV-enabled decoder subgraphs.
