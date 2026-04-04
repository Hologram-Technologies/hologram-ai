# Plan 054: Deep Decode Fusion

## Context

Profiling shows 96% of decode time is matmul (AMX-bound at 43 tok/s). The
remaining 4% (~1ms) is kernel launch overhead, intermediate buffer allocations,
and memory traffic between ops that could share data in-register. More
importantly, each intermediate buffer is a full hidden-dim allocation that
pollutes the cache hierarchy, degrading matmul performance indirectly.

Today hologram has **shallow fusions** — each pass fuses 2 adjacent ops:
- `AddRmsNorm` = Add + RmsNorm
- `FusedSwiGLU` = SiLU + Mul
- `MatMulActivation` = MatMul + SiLU/GeLU/ReLU

This plan introduces **deep fusions** — chaining 3-4 ops into single kernel
dispatches that eliminate all intermediate buffers through the entire
transformer block. The pattern generalizes: any chain of ops where the
intermediate result is consumed exactly once can be fused into a single
dispatch.

### Fusion depth vs transformer block

A standard transformer decode step has this structure:
```
residual ──→ Add ──→ RmsNorm ──→ QKV projection ──→ Attention ──→ Output projection
                                                                        │
residual ──→ Add ──→ RmsNorm ──→ Gate+Up projection ──→ SwiGLU ──→ Down projection
                                                                        │
                                                                    (next layer)
```

Today each box is a separate kernel dispatch. Deep fusion collapses adjacent
boxes into single dispatches:

```
[Add + RmsNorm + QKV projection]  ──→  Attention  ──→  Output projection
                                                              │
[Add + RmsNorm + Gate+Up projection]  ──→  [SwiGLU + Down projection]
                                                              │
                                                          (next layer)
```

This eliminates 4 intermediate buffers per layer × 22 layers = 88 buffer
allocations per decode step, plus 4 kernel launches per layer × 22 = 88
fewer dispatch calls.

---

## Design

### General Fusion Rule

All deep fusions follow one pattern:

> **If an op's output has exactly one consumer, and both ops can be computed
> in a single pass without global synchronization (no reductions across the
> fused dimension), fuse them.**

Barriers that prevent fusion:
- **Softmax/Reduce** — global reduction requires seeing all elements
- **Multiple consumers** — output must be materialized for both
- **Attention** — complex multi-pass kernel, fusion boundary

This rule subsumes all existing fusions and all new ones in this plan.

### Decode-Only Application

Deep fusions only help at **M=1** (decode). For M>1 (prefill), separate
BLAS `sgemm` calls are faster because AMX/BLAS is optimized for large
matrix multiplies. The fused kernels use GEMV (matrix-vector) which is
only competitive at M=1.

**Implementation approach:** Apply deep fusion passes to all graphs, but the
kernel implementations branch at runtime on M. For M=1: fused single-pass
kernel. For M>1: decompose and dispatch sub-ops separately (norm, then BLAS
gemm, etc). The M=1 check is a single comparison on buffer length.

### Quant-Aware Variants

Each fused kernel needs both f32 and LUT4 variants. The LUT4 variants read
quantized weights directly via `WeightCache`, applying dequant inside the
GEMV loop (same pattern as existing `dispatch_lut_gemm_4`).

### Known Risks and Mitigations

#### Risk 1: AMX vs fused GEMV on Apple Silicon

SharedInputProjectionFusion (Plan 052) was **tested and disabled** because AMX
handles separate `sgemm` calls faster than a single fused larger `sgemm` +
Slice overhead. NEON vecmat for M=1 was also tested — AMX wins 43 vs 10 tok/s.

This means NormProjectionGemv's fused GEMV kernel will likely be slower than
`cblas_sgemv` on Apple Silicon for the projection portion. The fusion win must
come from **eliminating the norm output buffer**, not from fusing the GEMV.

**Mitigation:** The fused kernel should call `cblas_sgemv` (not hand-written
GEMV) for the projection portion. The fusion benefit is:
1. Norm output stays in L1/stack (never written to arena → read back)
2. One fewer dispatch call
3. No arena allocation for the norm intermediate

If profiling shows this is still net negative on AMX, the fusion should be
gated behind a platform flag (enabled on non-AMX: WASM, Linux, x86).

#### Risk 2: rkyv serialization overflow for concatenated Q4 weights

Gate+Up fusion already hits rkyv overflow for 11264-col Q4 weights (SPRINT.md).
NormProjectionGemv concatenates QKV weights (2048+256+256=2560 cols for
TinyLlama, larger for bigger models).

**Mitigation:** Two approaches:
1. **Pre-check:** Validate serialized weight size before rkyv encoding. If
   exceeds threshold (e.g., 50 MB), skip fusion for that layer.
2. **Deferred concat:** Don't concatenate at compile time. Instead, store
   original weights separately and concatenate at tape-build time (or use
   multi-weight input to the fused kernel). This avoids the serialization
   issue entirely but requires the fused kernel to accept N weight inputs.

Option 2 is cleaner — the fused kernel reads 3 weight matrices directly
and iterates them in the GEMV inner loop. No compile-time weight mutation.

#### Risk 3: Slice allocation negates buffer elimination

The fused NormProjectionGemv outputs a single [M, N_total] buffer. Three
Slice nodes then allocate new buffers for Q, K, V. This partially negates
the "eliminate intermediate buffers" benefit.

**Mitigation:** Two approaches:
1. **Slice elision:** If a Slice's output feeds directly into a single
   consumer, the consumer can read from the parent buffer with an offset
   instead of copying. Requires `TapeKernel::SliceView` (zero-copy borrow
   with offset+length into parent arena slot).
2. **Multi-output kernel:** The fused kernel writes Q, K, V into separate
   pre-allocated arena slots directly (3 outputs instead of 1). This
   eliminates Slice nodes entirely but requires multi-output support in
   the tape executor (currently instructions have 1 output slot).

Option 1 is simpler; option 2 is more impactful. Start with option 1.

#### Risk 4: Two-level fusion ordering

hologram base runs graph-level fusions (`float_fusion.rs`) AFTER hologram-ai
lowers AiOp → FloatOp → GraphOp. If hologram-ai creates a `NormProjectionGemv`
FloatOp, hologram base's `try_fuse_matmul_activation` won't see the inner
MatMul (it's already consumed into the fused op). This is correct — hologram-ai
fusions take priority. But if hologram base adds its own deep fusions in the
future, they must not conflict.

**Mitigation:** Document in hologram base that deep fusions (multi-op chains
involving MatMul) are the compiler's responsibility. Base-level fusions are
limited to shallow patterns (2-op epilogue fusions, elementwise chains).

#### Risk 5: Profiling methodology

The plan claims 88 fewer buffer allocations but doesn't specify how to measure
the actual impact on cache pressure and tok/s.

**Mitigation:** Before implementation, establish baselines:
1. `HOLOGRAM_PROFILE=1` per-kernel timing at current tok/s
2. Arena allocation count per decode step (add counter to tape executor)
3. Peak RSS during decode (measures cache pressure indirectly)
4. After each wave, re-measure all three to validate improvement.

---

## Phase 1: Fused Kernels (hologram base)

### 1a. New FloatOp Variants

**File:** `hologram-core/src/op/float_op.rs` — append to enum (never insert mid-enum)

```rust
// ── Deep decode fusions ─────────────────────────────────────
// Norm → multi-output projection (QKV or Gate+Up)
NormProjectionGemv {
    norm_size: u32, epsilon: u32,
    k: u32,                    // input hidden dim
    split_sizes: [u32; 3],     // [n_q, n_k, n_v] or [n_gate, n_up, 0]
    n_splits: u8,              // 2 or 3
},

// Add + Norm → multi-output projection
AddNormProjectionGemv {
    norm_size: u32, epsilon: u32,
    k: u32,
    split_sizes: [u32; 3],
    n_splits: u8,
},

// SwiGLU + down projection in single pass
SwiGluProjectionGemv { k: u32, n: u32 },
```

Using split_sizes array makes these ops general — they work for QKV (3-way),
Gate+Up (2-way), or any future multi-output projection pattern.

### 1b. Kernel Implementations

**New file:** `hologram-exec/src/float_dispatch/fused_decode.rs`

Each kernel:
1. Check M from input buffer size
2. If M=1: execute fused single-pass (norm in stack buffer → GEMV)
3. If M>1: decompose to separate ops (call existing `dispatch_rms_norm_into` + `cblas_sgemm`)

```
NormProjectionGemv (M=1):
  stack_buf = rmsnorm(x, weight, eps)     // norm_size floats on stack
  output = gemv(stack_buf, W_concat)      // single GEMV, no intermediate alloc
  // caller slices output into [q, k, v] or [gate, up] via Slice nodes

AddNormProjectionGemv (M=1):
  stack_buf[i] = x[i] + residual[i]      // fused add
  rmsnorm_inplace(stack_buf, weight, eps) // norm in-place
  output = gemv(stack_buf, W_concat)

SwiGluProjectionGemv (M=1):
  // activated values computed in-register, never materialized:
  for col in 0..n:
    acc = 0.0
    for i in 0..k:
      activated = silu(gate[i]) * up[i]   // in-register
      acc += activated * W_down[i, col]
    output[col] = acc
```

The SwiGluProjectionGemv kernel is the most impactful — it eliminates the
full hidden-dim activation buffer that SwiGLU normally materializes.

### 1c. TapeKernel + Tape Builder + Dispatch

Follow exact pattern of existing `InlineAttention`, `InlineAddRmsNorm`:
- Add `TapeKernel::InlineNormProjectionGemv { ... }` etc. to tape.rs enum
- Map FloatOp → TapeKernel in tape_builder.rs
- Wire dispatch in `dispatch_kernel()` match

### 1d. M-Threshold Enhancement (bonus)

In `dispatch_matmul()` (matmul.rs), add GEMV fast path for M=1:
```rust
if m == 1 {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, k, 1.0, b, k, a, 1, 0.0, out, 1);
} else {
    cblas_sgemm(/* existing path */);
}
```

This benefits ALL M=1 matmuls, not just fused ones.

---

## Phase 2: Fusion Passes (hologram-ai)

### 2a. New AiOp Variants

**File:** `hologram-ai-common/src/ir/op.rs` — append to enum

```rust
/// Fused: [Add +] RmsNorm → multi-output projection (decode-optimized)
/// Inputs: [x, weight, W_concat] or [x, residual, weight, W_concat]
FusedNormProjection {
    epsilon: f64,
    split_sizes: Vec<usize>,  // output column splits
    has_residual_add: bool,   // true = Add+Norm+Proj, false = Norm+Proj
},

/// Fused: SwiGLU → down projection (decode-optimized)
/// Inputs: [gate, up, W_down]
FusedSwiGluProjection,
```

Two AiOp variants instead of five — the `has_residual_add` flag handles
both Norm→Proj and Add+Norm→Proj in one variant. The `split_sizes` vec
generalizes to any number of output splits.

### 2b. NormProjectionFusion Pass (new)

**New file:** `hologram-ai-common/src/opt/norm_projection_fusion.rs`

**Patterns matched:**

Pattern A — Norm → multi-way projection:
```
RmsNorm(x, weight, eps)
  → MatMul(normed, W_a) → a
  → MatMul(normed, W_b) → b
  [→ MatMul(normed, W_c) → c]   // optional 3rd output
```

Pattern B — Add + Norm → multi-way projection:
```
FusedLayerNormResidual(x, residual, weight, eps)  // or Add → RmsNorm
  → MatMul(normed, W_a) → a
  → MatMul(normed, W_b) → b
  [→ MatMul(normed, W_c) → c]
```

Also matches `Add(x, residual) → RmsNorm(sum, weight, eps) → ...` when
AddRmsNormFusion hasn't fired (e.g., cross-layer residual from previous
block that AddRmsNormFusion didn't see as same-node).

**Algorithm:**
1. For each RmsNorm or FusedLayerNormResidual node in topo order:
2. Collect all MatMul consumers of the norm output
3. If ≥2 MatMuls share the norm output as their first input:
   - Concatenate weight matrices at compile time: `W_concat = [W_a; W_b; W_c]`
   - Create FusedNormProjection node with split_sizes
   - Add Slice nodes for each original output (zero-cost in tape)
   - Remove original MatMul nodes
4. Single-consumer norm+MatMul: skip (no benefit over separate ops)

**Constraint:** Norm output must have no other consumers besides the
projections (otherwise it must be materialized).

### 2c. SwiGluProjectionFusion Pass (new)

**New file:** `hologram-ai-common/src/opt/swiglu_projection_fusion.rs`

**Pattern:**
```
FusedSwiGLU(gate, up) → MatMul(activated, W_down) → down_out
```

**Constraint:** FusedSwiGLU output has exactly 1 consumer (the down MatMul).

**Result:** `FusedSwiGluProjection` with inputs `[gate, up, W_down]`

### 2d. Lowering

**File:** `hologram-ai-common/src/lower/dispatch.rs`
- `FusedNormProjection` → `FloatNeedsShape`
- `FusedSwiGluProjection` → `FloatNeedsShape`

**File:** `hologram-ai-common/src/lower/strategy.rs`
- Resolve shapes: norm_size from x's last dim, k from weight, split_sizes from AiOp
- Map to `FloatOp::NormProjectionGemv` / `AddNormProjectionGemv` / `SwiGluProjectionGemv`
- For LUT4 weights: detect `ConstantId` on weight param, emit through
  `GraphOp::MatMulLut4` variant (or new fused LUT4 graph op)

### 2e. Pipeline Ordering

**File:** `hologram-ai-common/src/opt/pipeline.rs`

Insert after AddRmsNormFusion, before AttentionFusion:

```
...
6.  AddRmsNormFusion
7.  NormProjectionFusion      ← NEW (consumes RmsNorm/AddRmsNorm + MatMul)
8.  SwiGluProjectionFusion    ← NEW (consumes FusedSwiGLU + MatMul)
9.  SharedInputProjectionFusion  (catches remaining unfused projections)
10. PositionIdsInjection
11. AttentionFusion
...
```

---

## Phase 3: Shape Chain Elimination

Shape manipulation ops (Reshape, Transpose, Unsqueeze, Expand, Squeeze) move
data without computing anything. When they sit between compute ops, they create
intermediate buffers that waste bandwidth and pollute cache.

### 3a. GQA Head Expansion (Unsqueeze → Expand → Reshape)

ONNX models expand K/V from `[batch, num_kv_heads, seq, dim]` to
`[batch, num_q_heads, seq, dim]` via 3 synthetic ops. The attention kernel
already handles GQA internally via `num_kv_heads != num_q_heads`, but these
nodes still exist and execute.

**Fix:** Extend `AttentionFusion` to mark the Unsqueeze→Expand→Reshape chain
for dead-node elimination after fusion absorbs the expansion. The
`trace_past_expand()` helper already traces through these — just need to mark
them as removable when their only consumer is the fused attention.

### 3b. Q/K/V Reshape + Transpose

ONNX models reshape flat projections into heads then transpose:
```
MatMul(hidden, Wq) → Reshape [seq, n_q, head_dim] → Transpose [n_q, seq, head_dim]
```

Three nodes per Q/K/V × 22 layers = 198 shape-only nodes.

**Fix:** Two options:
1. **Absorb into attention:** The attention kernel already accepts different
   `heads_first` layouts. If we can infer the layout from the projection
   output shape, skip the Reshape+Transpose entirely.
2. **Fused ReshapeTranspose:** New zero-copy op that combines both without
   intermediate buffer. Only needed if option 1 doesn't cover all cases.

### 3c. Expand as Metadata (not Reshape)

`dispatch.rs:126` lowers `Expand` as `FloatOp::Reshape`, which copies data.
For broadcast expansion (e.g., KV cache), Expand should be a pure metadata
operation — the runtime just changes the shape without touching the buffer.

**Fix:** In lowering, emit `Identity` instead of `Reshape` when Expand's
input and output have the same number of elements. The tape executor already
handles Identity as a zero-copy passthrough.

---

## Phase 4: GEMM Prologue/Epilogue Absorption

Ops adjacent to MatMul that can be folded into GEMM parameters or applied
in-register during the matmul writeback loop.

### 4a. Transpose → MatMul (use trans_a/trans_b)

Pattern: `Transpose(perm=[..., 1, 0]) → MatMul(transposed, W)` appears in
attention K^T computation and some model architectures.

**Fix:** In lowering, detect when a MatMul input is produced by Transpose.
If the transpose is a simple last-2-dims swap, set `trans_a=true` or
`trans_b=true` on the `FloatOp::Gemm` variant and skip the Transpose node.
Avoids materializing the transposed buffer entirely.

Already partially done in `AttentionFusion` (`find_pre_transpose_with_scale`),
but not generalized to all MatMul sites.

### 4b. MatMul → Mul(scalar) (GEMM alpha)

Pattern: `MatMul(A, B) → Mul(result, scale)` where scale is a scalar constant.

**Fix:** Fold into `FloatOp::Gemm { alpha: scale, ... }`. The BLAS `sgemm`
already supports alpha scaling at zero cost. Eliminates a full-tensor
scalar multiply per occurrence.

### 4c. MatMul → Add(bias) → Activation (3-node fusion)

hologram base already has `FusedMatMulBiasActivation` at the graph level.
Ensure hologram-ai's lowering emits it when the pattern is detected. This
may already work via `try_fuse_matmul_bias_activation` in hologram base's
`float_fusion.rs` — verify the hologram-ai pipeline produces the right
graph structure for the base-level fusion to fire.

### 4d. Cast → MatMul / MatMul → Cast (mixed precision)

Pattern: `Cast(F16→F32) → MatMul` or `MatMul → Cast(F32→F16)`

**Fix (future):** When F16 compute is added (GPU backends), the cast can
be absorbed into the matmul kernel's input/output type parameters. On CPU
this is lower priority since everything runs in F32 anyway. Worth tracking
for when Metal/CUDA backends handle mixed-precision GEMM.

---

## Phase 5: Extended Fusion Patterns (all model types)

### 5a. Embed + Norm (LLM model entry)
```
Embed(token_ids) → RmsNorm(embeddings, weight, eps)
```
Eliminates the full embedding buffer. Applies to models with post-embedding
normalization (Qwen, Gemma). Fuses into `NormProjectionGemv` machinery with
Embed replacing the projection's input source.

### 5b. Final Norm + LM Head (LLM model exit)
```
RmsNorm(hidden, weight, eps) → MatMul(normed, W_lm_head) → logits
```
Always single-consumer. Fuses into `NormProjectionGemv` with `n_splits=1`.
This is the last fusion in the decode path — removes the final intermediate.

### 5c. Attention Output Projection
```
Attention(Q, K, V) → [Reshape →] MatMul(attn_out, W_o) → projected
```
The attention output is always single-consumer. Requires the attention kernel
to write to a pre-allocated buffer that the projection reads from — or
integrate the output projection into the attention dispatch. Deeper
integration than norm+projection but high impact (eliminates the attention
output buffer which is `n_heads × head_dim` per layer).

### 5d. LayerNorm + Projection (encoder models)

BERT/GPT-2/CLIP use LayerNorm instead of RmsNorm:
```
LayerNorm(x, weight, bias, eps) → MatMul(normed, W_q) → q
                                → MatMul(normed, W_k) → k
                                → MatMul(normed, W_v) → v
```

The `NormProjectionFusion` pass should match LayerNorm in addition to
RmsNorm. The kernel needs a `LayerNormProjectionGemv` variant (or a
`norm_type` flag on the existing `NormProjectionGemv`).

### 5e. Vision Model Chains

**Conv2d → GroupNorm → Activation → Conv2d** (SD UNet residual block):
Already partially handled by `FusedConv2dActivation` and
`FusedGroupNormActivation`. The missing piece is chaining all three:
`Conv2d → GroupNorm → SiLU` as a single dispatch.

**BatchNorm → ReLU** (ResNet):
BatchNorm is decomposed to 6 nodes by `OpDecomposition`. A dedicated
`FusedBatchNormActivation` FloatOp would collapse this back to a single
dispatch. Lower priority since ResNet is not the primary target.

### 5f. Scalar Broadcast Absorption

Pattern: `Mul(x, scalar_constant)` or `Add(x, scalar_constant)` where the
constant is shape `[1]` or `[]`.

**Fix:** When a scalar op is the sole consumer of a compute op (MatMul, Norm),
fold into the compute op's epilogue as a scale/bias parameter. This avoids a
full-tensor loop for what is effectively `output[i] *= scale`.

Already handled for attention scaling (`find_pre_transpose_with_scale`), but
not generalized.

---

## Phase 6: General Rule-Based Fusion Walker

Once the specific passes in Phases 1-5 prove correct, consolidate into a
single pass. This reduces maintenance as fusion count grows.

```
for each node in reverse topo order:
    if node.output has exactly 1 consumer:
        if can_fuse(node.op, consumer.op):
            merge into fused group
    emit deepest matching fused op for group
```

**Fusion rules:**
- Elementwise (Add, Mul, activation) → always fusable into predecessor
- Norm (RmsNorm, LayerNorm, GroupNorm) → fusable with preceding Add (residual)
  OR following single-consumer projection
- Projection (MatMul/GEMV) → max 1 per group (fusion barrier after this op)
- Reduce/Softmax/Attention → hard fusion barrier
- Multi-consumer → materialization point (fusion barrier)
- Shape-only ops (Reshape, Transpose, Squeeze, Unsqueeze) → absorb into
  adjacent compute ops when possible, otherwise zero-copy passthrough

**Fusion group → op selection:**
The walker produces groups. Each group maps to the deepest available fused op:
- `[Add, RmsNorm, MatMul]` → `AddNormProjectionGemv`
- `[RmsNorm, MatMul]` → `NormProjectionGemv`
- `[SiLU, Mul, MatMul]` → `SwiGluProjectionGemv`
- `[MatMul, SiLU]` → `MatMulSilu` (existing)
- `[Add, RmsNorm]` → `AddRmsNorm` (existing)
- `[SiLU, Mul]` → `FusedSwiGLU` (existing)
- Fallback: `FusedFloatChain` for pure elementwise groups (existing in base)

---

## Phase 4: LUT4 Variants (hologram base)

Each fused kernel needs a quantized variant that reads Q4 weights directly.
Two approaches:

**Option A (recommended): Dequant preamble.** The fused kernel calls
`weight_cache.get_dequantized_f32()` for the weight, then runs the same
f32 GEMV. This reuses the existing dequant cache and adds zero new quantization
code. The fusion benefit (eliminated intermediate buffers for norm/activation)
still applies.

**Option B (maximum performance): Inline LUT-GEMV.** The fused kernel
integrates the Q4 nibble unpacking into the GEMV inner loop, using the
`tiled_vecmat_q4` pattern. Higher performance but more code to maintain.

Start with Option A; profile to determine if Option B is needed.

---

## Implementation Order

### Wave 1: Core deep fusions (Phases 1-2)

| Step | Repo | What | Depends On |
|------|------|------|------------|
| 1 | hologram base | FloatOp variants (3 new) | — |
| 2 | hologram base | fused_decode.rs kernel implementations | Step 1 |
| 3 | hologram base | TapeKernel + tape_builder + dispatch | Steps 1-2 |
| 4 | hologram base | M=1 GEMV fast path in dispatch_matmul | — |
| 5 | hologram-ai | AiOp variants (2 new) | — |
| 6 | hologram-ai | NormProjectionFusion pass | Step 5 |
| 7 | hologram-ai | SwiGluProjectionFusion pass | Step 5 |
| 8 | hologram-ai | Lowering (dispatch.rs + strategy.rs) | Steps 1, 5 |
| 9 | hologram-ai | Pipeline ordering | Steps 6-7 |
| 10 | both | Integration tests | Steps 1-9 |

Steps 1-4 (hologram base) and steps 5-7 (hologram-ai) can proceed in parallel.

### Wave 2: Shape chain elimination + GEMM absorption (Phases 3-4)

| Step | Repo | What | Depends On |
|------|------|------|------------|
| 11 | hologram-ai | GQA Expand chain → dead node elimination | Wave 1 |
| 12 | hologram-ai | Transpose → MatMul (emit trans_b in lowering) | — |
| 13 | hologram-ai | MatMul → Mul(scalar) → Gemm alpha absorption | — |
| 14 | hologram-ai | Expand → Identity lowering (same-element-count) | — |
| 15 | hologram-ai | Q/K/V Reshape+Transpose absorption into attention | Wave 1 |

### Wave 3: Extended patterns (Phase 5)

| Step | Repo | What | Depends On |
|------|------|------|------------|
| 16 | hologram-ai | Embed + Norm fusion | Wave 1 |
| 17 | hologram-ai | Final Norm + LM Head fusion | Wave 1 |
| 18 | hologram-ai | LayerNorm + Projection (encoder models) | Wave 1 |
| 19 | both | LUT4 variants (Option A: dequant preamble) | Wave 1 |
| 20 | both | Conv2d + GroupNorm + Activation chain (vision) | — |
| 21 | both | Scalar broadcast absorption | — |

### Wave 4: Consolidation (Phase 6)

| Step | Repo | What | Depends On |
|------|------|------|------------|
| 22 | hologram-ai | General rule-based fusion walker | Waves 1-3 proven |

---

## Verification

### Per-wave gates

**Wave 1 (core deep fusions):**
1. Unit tests per fusion pass: construct mini AiGraph → run pass → assert
   fused ops present with correct split_sizes
2. Kernel correctness: compare fused kernel output vs sequential execution
   within f32 tolerance (1e-5 relative error)
3. TinyLlama decode regression: measure tok/s before/after, verify
   identical top-5 token predictions
4. M>1 fallback: verify prefill (M=32) still works — fused kernels
   decompose to separate ops
5. LUT4 path: Q4 model produces same tokens with fused vs unfused kernels

**Wave 2 (shape chain elimination):**
6. Node count regression: count total nodes in compiled TinyLlama graph
   before/after — expect ~200 fewer shape-only nodes
7. ONNX attention path: verify Q/K/V Reshape+Transpose nodes removed
   after fusion, attention output unchanged
8. Transpose absorption: verify Gemm trans_b=true emitted when input is
   Transpose, and Transpose node eliminated

**Wave 3 (extended patterns):**
9. Embed+Norm: single kernel dispatch at model entry, output matches
   separate Embed then RmsNorm within tolerance
10. Final Norm+LM Head: single kernel dispatch at model exit, logits match
11. BERT/CLIP encoder: LayerNorm+Projection fires on encoder models
12. SD UNet: Conv2d+GroupNorm+SiLU chain count reduced

**Wave 4 (general walker):**
13. Regression: walker produces identical fused graph as individual passes
    on TinyLlama, BERT, ResNet, SD UNet
14. No new fused ops needed — walker selects from existing vocabulary

---

## Key Files

### hologram base (to modify)

**Wave 1:**
- `hologram-core/src/op/float_op.rs` — FloatOp enum (append 3 variants)
- `hologram-exec/src/float_dispatch/fused_decode.rs` — new kernel file
- `hologram-exec/src/float_dispatch/mod.rs` — wire dispatch
- `hologram-exec/src/tape.rs` — TapeKernel enum + dispatch_kernel match
- `hologram-exec/src/tape_builder.rs` — FloatOp → TapeKernel mapping
- `hologram-exec/src/float_dispatch/matmul.rs` — M=1 GEMV fast path

**Wave 3:**
- `hologram-exec/src/float_dispatch/norm.rs` — LayerNorm+Projection variant
- `hologram-exec/src/float_dispatch/conv.rs` — Conv2d+GroupNorm chain
- `hologram-graph/src/fusion/float_fusion.rs` — extend graph-level fusion

### hologram-ai (to modify)

**Wave 1:**
- `hologram-ai-common/src/ir/op.rs` — AiOp enum (append 2 variants)
- `hologram-ai-common/src/opt/norm_projection_fusion.rs` — new pass
- `hologram-ai-common/src/opt/swiglu_projection_fusion.rs` — new pass
- `hologram-ai-common/src/opt/mod.rs` — register passes
- `hologram-ai-common/src/opt/pipeline.rs` — ordering
- `hologram-ai-common/src/lower/dispatch.rs` — DispatchTarget mapping
- `hologram-ai-common/src/lower/strategy.rs` — shape resolution + lowering

**Wave 2:**
- `hologram-ai-common/src/opt/attention_fusion.rs` — mark GQA expand chain
  for removal, absorb Q/K/V Reshape+Transpose
- `hologram-ai-common/src/lower/dispatch.rs` — Expand → Identity when
  element count unchanged
- `hologram-ai-common/src/lower/strategy.rs` — Transpose → Gemm trans_b
  absorption, scalar Mul → Gemm alpha

**Wave 3:**
- `hologram-ai-common/src/opt/norm_projection_fusion.rs` — extend to
  LayerNorm, Embed+Norm, Final Norm+LM Head patterns

**Wave 4:**
- `hologram-ai-common/src/opt/general_fusion.rs` — new rule-based walker
