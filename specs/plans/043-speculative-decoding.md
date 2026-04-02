# Plan 043: Speculative Decoding — 80-120 tok/s

**Status:** Open
**Created:** 2026-04-02
**Branch:** `feat/cpu-inference-perf`
**Depends on:** Plan 040 (performance), Plan 041 (variable-length fix for verification)
**Baseline:** 40.9 tok/s (TinyLlama f32/Q4, M4 Max, single-path executor)

## Motivation

Autoregressive decode is bandwidth-limited: each token requires reading ALL model
weights once. At 40 tok/s, each step reads 4.1 GB in 22ms. Speculative decoding
breaks this barrier by generating multiple tokens per weight read.

## How It Works

1. **Draft model** (small, fast) generates N candidate tokens autoregressively
2. **Target model** (large, accurate) verifies all N candidates in ONE batched
   forward pass — reads weights once for N tokens instead of N times
3. **Accept/reject** — compare draft vs target distributions per position:
   - Accept: draft token matches target distribution (60-70% rate typical)
   - Reject: sample from adjusted target distribution at first mismatch
4. **Net effect**: 2-3 accepted tokens per target forward pass = 2-3x throughput

## Architecture

```
SpeculativeDecoder {
    target: HoloRunner,        // TinyLlama 1.1B (or larger)
    draft: HoloRunner,         // Smaller model (same tokenizer)
    draft_steps: usize,        // Candidates per batch (4-8)
    target_kv: KvCacheState,   // Target KV cache
    draft_kv: KvCacheState,    // Draft KV cache
}
```

### Generation Loop

```
for each batch:
    1. Draft generates N tokens (sequential, fast): ~N × 5ms each
    2. Build batched input: [prompt + draft tokens] for target
    3. Target verifies: ONE forward pass at seq=N (~22ms for any N ≤ 8)
    4. Compare logits at each position:
       - If draft[i] matches target[i]: accept, advance both KV caches
       - If mismatch at position j: sample from target[j], discard j+1..N
    5. Net accepted: ~0.6-0.7 × N tokens per target forward pass
```

### Key Insight: Verification is Cheap

Target verification of N tokens costs ~same as generating 1 token because:
- MatMul: `[N, 2048] × [2048, 2048]` reads the same 16 MB weight matrix as
  `[1, 2048] × [2048, 2048]` — bandwidth-limited, not compute-limited
- BLAS is optimized for small M (N ≤ 8) — near-zero overhead vs M=1
- Attention: `[N, heads, 64] × [64, seq_k]` — N queries share cached K/V

## Implementation

### New Files

**`crates/hologram-ai/src/speculative.rs`**

```rust
pub struct SpeculativeDecoder {
    target: HoloRunner,
    draft: HoloRunner,
    draft_steps: usize,       // default 6
    temperature: f32,
    top_k: usize,
}

impl SpeculativeDecoder {
    pub fn new(target: HoloRunner, draft: HoloRunner) -> Self;

    /// Generate tokens with speculative decode.
    /// Returns (tokens, stats) where stats includes acceptance rate.
    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        tokenizer: &TokenizerSection,
    ) -> (Vec<u32>, SpecStats);
}

pub struct SpecStats {
    pub tokens_generated: usize,
    pub draft_calls: usize,
    pub target_calls: usize,
    pub acceptance_rate: f32,
    pub effective_tok_s: f32,
}
```

### CLI Changes

**`crates/hologram-ai/src/commands/run_cmd.rs`**

```rust
#[arg(long, value_name = "PATH")]
pub draft_model: Option<PathBuf>,

#[arg(long, default_value = "6")]
pub draft_steps: usize,
```

When `--draft-model` is provided, wrap the target+draft HoloRunners in a
`SpeculativeDecoder` and use it for the generation loop.

### Draft Model Requirements

- Same tokenizer/vocabulary as target (validated at load time)
- Same architecture family (both LLM with KV cache)
- Smaller: draft should be ~4-10x faster per token than target
- Q4 quantization recommended for both draft and target

**Supported pairs:**
| Target | Draft | Speed Ratio |
|--------|-------|-------------|
| TinyLlama 1.1B | TinyLlama 1.1B Q4 | 1x (self-speculative) |
| Llama 3.1 8B | Llama 3.2 1B | ~8x |

### Acceptance/Rejection Algorithm

Standard speculative sampling (Leviathan et al., 2023):

```
for i in 0..N:
    p = target_logits[i]  // target distribution at position i
    q = draft_logits[i]   // draft distribution at position i
    r = random(0, 1)
    if r < min(1, p[draft_token[i]] / q[draft_token[i]]):
        accept draft_token[i]
    else:
        sample from adjusted: (p - q).clamp(min=0).normalize()
        reject all subsequent tokens
        break
```

## Performance Projection

| Metric | Without Spec | With Spec (N=6, 65% accept) |
|--------|-------------|---------------------------|
| Target forward passes | 1 per token | 1 per ~4 tokens |
| Draft forward passes | 0 | ~6 per batch |
| Effective tok/s | 40 | ~100-120 |
| Memory | 4.1 GB | 4.1 GB target + ~0.5 GB draft |

## Prerequisites

- **Variable-length execution fix (Plan 041 / Phase 3C)** — verification pass
  sends N tokens but model was compiled at seq=1. Need dynamic seq_len support.
  OR: compile a separate verification graph at seq=N.
- **Batched matmul (M > 1 during decode)** — already supported by BLAS dispatch.

## Testing

- Unit test: acceptance/rejection matches reference implementation
- Test: speculative output matches greedy decode for deterministic (temp=0) case
- Test: effective tok/s > 2x base tok/s with self-speculative
- Memory: target + draft fit in memory budget

## Verification

- `cargo test --release -p hologram-ai -- speculative`
- Manual: `hologram-ai run model.holo --draft-model draft.holo --prompt "..."`
- tok/s measurement: effective rate accounting for rejected tokens
