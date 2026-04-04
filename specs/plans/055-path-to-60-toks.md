# Plan 055: Path to 60+ tok/s

## Context

Current: 38 tok/s GGUF Q4, 2.7 tok/s ONNX f32. Goal: 60+ tok/s.
AMX is saturated at ~43 tok/s — further single-thread optimization
won't move the needle. Need to reduce data AND multiply effective tokens.

## Phase 1: Strip f32 from Q4 Archives (hologram-ai)

**Problem:** ONNX `--quantize q4_0` produces 4.5 GB archives because f32
originals are registered as constants BEFORE Q4 conversion, and never removed.

**Fix:** Pre-scan which MatMul weights are Q4-eligible, skip registering
their f32 bytes in the parameter loop.

**Implementation:**
1. Before the parameter registration loop in `builder.rs`:
   - Collect `attn_dims` from GroupedQueryAttention nodes (existing logic)
   - Scan all MatMul nodes, check Q4 eligibility (strategy==Q4_0, shape 2D,
     dims>=256, not `feeds_attention`)
   - Build `HashSet<TensorId>` of Q4-eligible weight tids

2. In the parameter loop (lines 130-212):
   - If `q4_eligible_weights.contains(&tid)`: skip `param_bytes_owned()`,
     register as `ConstantData::Bytes(vec![])` placeholder (zero bytes)
   - Else: register normally

3. The Q4 interception (lines 407-438) already creates the real Q4 constant
   via `builder.matmul_lut_4bit()` — no change needed there.

4. Empty placeholder constants are harmless — they occupy no archive space
   and are never referenced by any node.

**Files:** `crates/hologram-ai-common/src/lower/builder.rs`
**Impact:** Archive 4.5 GB → ~0.5 GB. ONNX Q4 matches GGUF Q4 at 38-43 tok/s.
**Effort:** Low (~50 lines)

## Phase 2: Speculative Decoding (hologram-ai + hologram base)

**Approach:** Generate N draft tokens, verify in one forward pass. Accept/reject
per Leviathan et al. 2023. Self-speculative (first N/4 layers as draft) or
separate draft model.

**Key architectural insight:** HoloRunner already supports 2 tapes (prefill +
decode) with shared weights. Add a 3rd tape: verification (compiled at seq=N).

**Implementation:**

### 2a. Verification tape compilation (hologram-ai compiler)
- Add `--verification-seq N` flag (default 8)
- Compile 3 components: prefill (seq=prompt_len), decode (seq=1),
  verify (seq=N) — same model, different concretized seq_len
- Pipeline archive stores 3 components

### 2b. HoloRunner 3-tape support (hologram-ai)
- Add `verify_plan` + `verify_tape` fields alongside existing decode fields
- `execute_verify(&inputs, &mut kv)` method uses verification tape
- Shared `WeightCache` across all 3 tapes

### 2c. SpeculativeDecoder (hologram-ai, new file)
```rust
pub struct SpeculativeDecoder {
    runner: HoloRunner,           // Handles all 3 tapes
    draft_steps: usize,           // N candidates (default 4-6)
}

impl SpeculativeDecoder {
    pub fn generate_step(&mut self, kv: &mut KvCacheState) -> Vec<u32> {
        // 1. Draft: run decode tape N times (single token each)
        let drafts = self.generate_drafts(kv);
        // 2. Verify: run verification tape on [drafts] batch
        let target_logits = self.verify(kv, &drafts);
        // 3. Accept/reject
        self.accept_reject(&drafts, &target_logits, kv)
    }
}
```

### 2d. Acceptance/rejection sampling
- Add `sample_with_rejection(p_target, p_draft)` to sampling logic
- Adjusted distribution: `(p_target - p_draft).clamp(0).normalize()`

### 2e. CLI integration
- `--speculative` flag enables speculative decode
- `--draft-steps N` controls candidates per batch

**Files:**
- `crates/hologram-ai/src/speculative.rs` (new)
- `crates/hologram-ai/src/commands/run_cmd.rs` (generation loop)
- `crates/hologram-ai/src/compiler.rs` (3-component compilation)
- hologram base: pipeline archive 3-component support

**Impact:** 2x effective throughput → 38 × 2 = 76 effective tok/s
**Effort:** High (new subsystem)

## Phase 3: Q2 Quantization (hologram base + hologram-ai)

**Approach:** 2-bit weights (4 centroids) = half the data per matmul vs Q4.
LUT-GEMM with `Psumbook2` (16 bytes, 4 f32 slots).

**Implementation:**

### 3a. Q2 types (hologram base)
- `QuantizedWeights2` struct (4 centroids, 2-bit packed indices)
- `quantize_2bit()` in `lut_gemm/quantize.rs`
- `Psumbook2` partial sum accumulator

### 3b. Q2 kernel (hologram base)
- `lut_gemm_2bit()` inner loop: 4 centroids, 2 bits per weight
- NEON: `vqtbl1q_s8` with 4-entry table, pack 4 weights per byte
- `dispatch_lut_gemm_2()` in tape.rs

### 3c. Graph + tape wiring (hologram base)
- `GraphOp::MatMulLut2(ConstantId)` + `MatMulLut2Activation`
- `TapeKernel::MatMulLut2(ConstantId)`
- Tape builder mapping

### 3d. Compiler integration (hologram-ai)
- `QuantStrategy::Q2_0` enum variant
- `--quantize q2_0` CLI flag
- `try_convert_f32_to_lut2()` in builder.rs
- Quality gating: coherent English output validation

**Files:**
- hologram base: `lut_gemm/`, `tape.rs`, `tape_builder.rs`, `graph/mod.rs`
- hologram-ai: `lower/builder.rs`, `compiler.rs`, CLI

**Impact:** Base tok/s 38 → 50-60 (half the weight reads)
**Effort:** Medium (follows Q4 pattern exactly)

## Combined Impact

| Configuration | Base tok/s | With Speculative (2x) |
|--------------|-----------|----------------------|
| ONNX f32 (current) | 2.7 | 5.4 |
| ONNX Q4 (Phase 1) | 38-43 | 76-86 |
| ONNX Q2 (Phase 1+3) | 50-60 | 100-120 |

**Phase 1 alone hits 38-43.** Phase 1+2 hits **76+ (exceeds 60 target).**
Phase 1+2+3 hits **100+.**

## Implementation Order

| Step | Phase | What | Depends On |
|------|-------|------|------------|
| 1 | P1 | Pre-scan Q4 eligibility, skip f32 bytes | — |
| 2 | P1 | Test: archive < 1 GB, tok/s matches GGUF | Step 1 |
| 3 | P2a | Verification tape compilation (3-component) | — |
| 4 | P2b | HoloRunner 3-tape loading | Step 3 |
| 5 | P2c | SpeculativeDecoder struct + draft/verify loop | Step 4 |
| 6 | P2d | Accept/reject sampling | Step 5 |
| 7 | P2e | CLI + E2E test | Steps 5-6 |
| 8 | P3a | Q2 types + quantize_2bit() | — |
| 9 | P3b | Q2 LUT-GEMM kernel (NEON/AVX2) | Step 8 |
| 10 | P3c | Graph + tape wiring | Step 9 |
| 11 | P3d | Compiler integration + quality gate | Steps 10, 1 |

Steps 1-2 (Phase 1) and 8-9 (Phase 3a-b) can run in parallel.
Phase 2 is the longest path but gives the 2x multiplier.

## Verification

- **Phase 1:** `ls -lh` archive < 1 GB; tok/s >= 38 on ONNX Q4
- **Phase 2:** effective tok/s > 60; output matches greedy baseline
- **Phase 3:** Q2 archive < 0.3 GB; base tok/s > 50; coherent output
- **Combined:** effective tok/s > 100 on TinyLlama Q2 + speculative
