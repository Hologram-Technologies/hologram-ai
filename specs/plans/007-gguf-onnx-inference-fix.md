# Plan 007: Fix GGUF + ONNX E2E Inference

## Context

Sprint `feat/tinyllama-e2e`. Both TinyLlama ONNX and GGUF models compile and
run but produce incoherent/degenerate output. Goal: identify root causes via
conformance tests (per AGENTS.md mandate), fix them, and get coherent English
output from both.

**Current state:**
- ONNX: compiles (1205 nodes, 4.1 GB), runs without error, but output
  incoherent — `inject_lm_head_if_needed` exists but unclear if the ONNX
  export includes `embed_tokens.weight` under that exact name.
- GGUF: compiles (333 nodes, 606 MB), runs, generates "Response:" then
  degenerates; most recent run showed Cyrillic garbage ("▁Див").

---

## Analysis

### ONNX Path

- `inject_lm_head_if_needed()` in `hologram-ai-onnx/src/lib.rs:131–210`
  injects `logits = last_hidden_state @ embed_tokens.weight^T` when the output
  is `last_hidden_state` and a param named `embed_tokens.weight` is present.
- If the HF ONNX export names it differently (e.g. `model.embed_tokens.weight`)
  the function silently no-ops → model outputs 2048-dim hidden state → garbage.
- A `model_causal.onnx` also exists in the model directory (265 MB manifest),
  which likely already outputs `logits` natively — usable as a fallback.
- The autoregressive run loop (`run_cmd.rs`) correctly pads to
  `compiled_seq_len` and extracts logits at `actual_len - 1`.

### GGUF Path — What is and isn't broken

| Component | Status | Evidence |
|-----------|--------|----------|
| lm_head | ✓ correct | `llama.rs:172–182` |
| Q4_0 dequant | ✓ correct | `q4_0.rs:19–29`, signed nibble verified |
| Causal mask | ✓ correct | upper-triangular neg-inf before softmax |
| RoPE positions | ✓ correct | `token_pos = chunk_idx / n_heads` (n_heads=32 for TinyLlama) |
| **GQA kernel** | ❓ PRIMARY SUSPECT | must reshape flat [B,S,2048]+[B,S,256] → heads, repeat KV 4×, SDPA |
| **SwiGLU** | ❓ SECONDARY | must be `silu(gate)*up`, not `gate*up` |

---

## Implementation Steps

### Step 0 (on execution start)
- Save this plan to `specs/plans/007-gguf-onnx-inference-fix.md` ← this file
- Add Phase 5 section to `specs/SPRINT.md`

### Step 1: Conformance tests (before any kernel fix)

Add to `hologram-ai-conformance/tests/exec_conformance.rs`:

**A. `gqa_matches_ort`**
- ONNX subgraph: Transpose + MatMul + Mul(1/sqrt(head_dim)) + Softmax + MatMul
  mimicking GQA for n_heads=8, n_kv_heads=2, head_dim=16, seq=4
- Compare hologram vs ORT output

**B. `swiglu_matches_ort`**
- ONNX: `Sigmoid(gate) * gate * up` (= silu(gate) * up)
- Compare vs `AiOp::FusedSwiGLU`

**C. `inject_lm_head_regression`** (unit test in `hologram-ai-onnx/src/lib.rs`)
- Synthetic ONNX with `last_hidden_state` output + `embed_tokens.weight` param
- Assert injection produces `logits` at `[B, S, vocab]`
- Assert no-op when param absent

### Step 2: Run conformance tests → identify failures

```bash
ORT_STRATEGY=system cargo test -p hologram-ai-conformance --features conformance \
  -- gqa swiglu inject --nocapture
```

Decision tree:
- GQA mismatch → fix `dispatch_attention` in hologram's `float_dispatch.rs`
- SwiGLU mismatch → fix `FusedSwiGLU` kernel
- inject no-op → extend to try alternate embedding weight names

### Step 3: Fix ONNX lm_head (if inject_lm_head_regression reveals gap)

Extend `inject_lm_head_if_needed` to try:
- `embed_tokens.weight`
- `model.embed_tokens.weight`
- `token_embd.weight`

Or switch e2e test to `model_causal.onnx` if it already has logits.

### Step 4: Fix kernel bugs

**GQA (if broken)** — `hologram/crates/hologram-exec/src/float_dispatch.rs`:
- Reshape Q: `[B, S, n_q*head_dim]` → `[B, n_q, S, head_dim]`
- Reshape K/V: `[B, S, n_kv*head_dim]` → `[B, n_kv, S, head_dim]`
- Repeat K/V heads: each KV head serves `n_q/n_kv` query heads (4× for TinyLlama)
- Scale: `1/sqrt(head_dim)` when `scale: None`
- Causal mask per query head

**SwiGLU (if broken)** — same file:
- `silu(gate) * up` = `gate * sigmoid(gate) * up`, NOT `gate * up`

### Step 5: Verify E2E

```bash
cargo test -p hologram-ai --features e2e -- tinyllama --nocapture
```

Update `tinyllama_e2e.rs` tests to assert coherent output (not just "no error").

### Step 6: Final SPRINT.md update

Tick all Phase 5 items as complete.

---

## Critical Files

| File | Role |
|------|------|
| `crates/hologram-ai-conformance/tests/exec_conformance.rs` | New GQA + SwiGLU tests |
| `crates/hologram-ai-onnx/src/lib.rs` | inject_lm_head unit tests |
| `hologram/crates/hologram-exec/src/float_dispatch.rs` | Fix site for GQA/SwiGLU |
| `crates/hologram-ai-gguf/src/arch/llama.rs` | Fix site if graph builder issue |
| `crates/hologram-ai/tests/tinyllama_e2e.rs` | Update assertions |
| `specs/SPRINT.md` | Track progress |

---

## Verification

```bash
# Conformance (pinpoints bug)
ORT_STRATEGY=system cargo test -p hologram-ai-conformance --features conformance \
  -- gqa swiglu inject_lm_head --nocapture

# E2E (validates fix)
cargo test -p hologram-ai --features e2e -- tinyllama --nocapture

# Full suite
cargo test && cargo clippy -- -D warnings
```
