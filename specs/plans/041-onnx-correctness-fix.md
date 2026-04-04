# Plan 041: ONNX Execution Correctness Fix

**Status:** Open (BLOCKER for Plan 040 performance work)
**Created:** 2026-04-01
**Branch:** `feat/cpu-inference-perf`

## Problem

TinyLlama ONNX (torch 2.11 export, opset 14 and 18) compiles successfully
(891 nodes, 0 warnings) but execution produces gibberish. Prefill step 0
top-1 token is "(" instead of a contextual response to "What is the capital
of France?". Logits are numerically valid (no NaN/inf, range -13 to +11)
but semantically wrong.

## Root Cause Hypothesis

The torch 2.11 `torch.onnx.export` produces SDPA (Scaled Dot Product Attention)
as a decomposed pattern with node names like `node_scaled_dot_product_attention`.
The hologram-ai `AttentionFusion` pass may not recognize this pattern, causing:

1. **No attention fusion** — SDPA chain not fused into `GroupedQueryAttention`
2. **No KV cache injection** — `KvSlotInjection` depends on fused attention nodes
3. **Wrong execution** — model runs as a single forward pass without proper
   autoregressive attention masking

The hologram base was updated with new tape executor, FloatOp variants, and
execution APIs. hologram-ai's compiler needs matching updates.

## Debugging Plan

### Phase 1: Diagnose (read-only)

1. **Verify AttentionFusion fires** — compile with `RUST_LOG=debug` and check
   if `AttentionFusion` fuses any nodes. If 0 fusions, the pattern doesn't match.
   ```
   RUST_LOG=debug cargo run --release -p hologram-ai -- compile ...
   grep "AttentionFusion" output
   ```

2. **Dump ONNX attention subgraph** — use Python to extract the first attention
   layer's op chain and compare to what `AttentionFusion` pattern-matches.

3. **Compare to working model** — if GGUF path works, compare the compiled
   graph structure (node types, shapes) between ONNX and GGUF compilations.

4. **Check node-by-node divergence** — compile TinyLlama ONNX with
   `compile_with_debug_info`, then run each node's output against ORT to
   find the first divergent node.

### Phase 2: Fix AttentionFusion for torch 2.11 SDPA

5. **Update SDPA pattern matching** — the torch 2.11 export may produce:
   - Different node naming conventions
   - Different decomposition of SDPA (MatMul Q*K^T vs batched MatMul)
   - Different causal mask application (Where vs masked_fill)
   - RoPE applied differently (Slice+Concat vs Rotate)

6. **Add new pattern variants** to `AttentionFusion` pass in
   `crates/hologram-ai-common/src/opt/attention_fusion.rs`

### Phase 3: Fix KV Cache Pipeline

7. **Verify KvSlotInjection** — after fixing AttentionFusion, verify that
   `KvSlotInjection` correctly injects KvWrite/KvRead around fused attention.

8. **Verify prefill+decode split** — ensure the compiled pipeline has both
   prefill graph (full prompt, seq=N) and decode graph (single token, seq=1).

### Phase 4: Validate

9. **Node-by-node validation** — compare every intermediate tensor against ORT
10. **End-to-end test** — TinyLlama generates coherent English response
11. **tok/s measurement** — verify we can measure real performance

## Key Files

- `crates/hologram-ai-common/src/opt/attention_fusion.rs` — SDPA pattern matching
- `crates/hologram-ai-common/src/opt/pipeline.rs` — optimization pipeline order
- `crates/hologram-ai-common/src/lower/strategy.rs` — AiOp → FloatOp lowering
- `crates/hologram-ai/src/compiler.rs` — compilation pipeline
- `crates/hologram-ai/src/commands/run_cmd.rs` — execution + KV cache wiring

## Verification

- `cargo test --release -p hologram-ai --features e2e -- tinyllama_onnx_runs`
- Manual: `hologram-ai run model.holo --prompt "..." --max-tokens 20` produces English
- Benchmark: tok/s matches expected range for model size
