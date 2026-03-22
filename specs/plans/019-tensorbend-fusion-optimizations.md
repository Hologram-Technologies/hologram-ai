# Plan 019: TensorBend-Inspired Fusion Optimizations

## Context

TensorBend is an in-browser LLM inference engine using WebGPU with ~50 hand-written WGSL compute shaders. Analyzing its fusion patterns reveals optimization opportunities for hologram-ai's compiler passes and hologram's execution layer.

---

## GPU Backend Strategy

### Current state: CPU-only, optional macOS Accelerate

Hologram has no GPU backend infrastructure today. All kernels are pure Rust in `float_dispatch/`, with optional `cblas_sgemm` on macOS via the `accelerate` feature flag.

### Can we wrap TensorBend's WGSL shaders into hologram?

Not directly — they're obfuscated JS bundles, not standalone shader files. But TensorBend's kernel suite is an excellent **reference for what fused GPU kernels should look like** and what fusion boundaries to target.

### Recommended GPU backend approach

Hologram has **three clean abstraction points** for plugging in GPU backends (CUDA, Metal, WebGPU):

**Option 1 — Tape-level interception (recommended)**
The emerging tape executor (`tape.rs`) pre-resolves `KernelFn` pointers per node. Replace the scalar `KernelFn = fn(&[&[u8]], ...) -> Vec<u8>` with a `trait Kernel` that can hold GPU state:

```rust
// New trait in hologram-exec
pub trait Kernel: Send + Sync {
    fn execute(&self, inputs: &[&[u8]], ctx: Option<&ExecutionContext>) -> ExecResult<Vec<u8>>;
}

// CPU kernels: implement Kernel with existing dispatch functions
// GPU kernels: implement Kernel with device pointers, async dispatch, etc.
```

- Tape instructions hold `Box<dyn Kernel>` instead of `KernelFn`
- TapeBuilder accepts a `BackendFactory` that vends kernels per FloatOp
- GPU backend keeps weights + KV cache on device; only stages outputs to CPU at graph outputs
- **Key files**: `crates/hologram-exec/src/tape.rs`, `tape_builder.rs`

**Option 2 — CustomOpRegistry (works today, limited)**
Register GPU kernel handlers via existing `register_op!()` macro. Proven pattern, but lacks device memory management and requires CPU↔GPU staging per op.

**Option 3 — dispatch_float_ctx replacement (invasive)**
Replace the category-based dispatch in `float_dispatch/mod.rs` with backend-aware routing. Most invasive, touches every kernel.

### What a GPU backend needs to implement

Priority kernels (covers ~90% of inference compute):

| Priority | FloatOp | GPU Kernel | Notes |
|----------|---------|------------|-------|
| P0 | `MatMul { m, k, n }` | cuBLAS/Metal MPSMatrixMultiplication/WGSL | Dominates compute |
| P0 | `Attention { head_dim, ... }` | FlashAttention / custom | Fused QKV + softmax |
| P1 | `RmsNorm { size, eps }` | Custom | Simple reduction |
| P1 | `RotaryEmbedding { dim, base }` | Custom | Position-dependent |
| P1 | `KvWrite / KvRead` | Device memory management | Keep cache on-device |
| P2 | Elementwise (Add, Mul, SiLU, etc.) | Fused chains | Low arithmetic intensity |
| P2 | `Softmax { size }` | Custom | Online softmax preferred |
| P3 | `Conv2d`, `Gather`, `Concat` | Standard | Less common in LLM inference |

### Implementation prerequisites
1. **Tape executor must be wired first** (Plan 018 tasks C-F)
2. `BufferArena` needs a GPU memory bank extension (device pointers alongside CPU `Cow<[u8]>`)
3. KV cache needs device-side storage (avoid CPU↔GPU round-trip per token)
4. Shape metadata already available on `SerializedGraph` nodes — no graph changes needed

### Effort: Large (multi-sprint), but incremental
- Phase 1: `trait Kernel` + CPU implementation (no behavior change)
- Phase 2: Metal backend for MatMul + Attention (macOS, replaces Accelerate)
- Phase 3: CUDA backend for MatMul + Attention
- Phase 4: WebGPU backend (wgpu crate, cross-platform)
- Phase 5: Device-resident KV cache + weight caching

---

## Task 1: SwiGLU Fusion Pass

**Gap**: `FusedSwiGLU` exists in both `AiOp` (line 91 of `op.rs`) and `FloatOp` (line 263 of `float_op.rs`) with a working kernel in hologram base, but **no optimization pass creates it**.

**Pattern to match**:
```
gate = MatMul(x, gate_weights)
up   = MatMul(x, up_weights)
act  = SiLU(gate)
out  = Mul(act, up)
```
→ `FusedSwiGLU(x, gate_weights, up_weights)`

**Files**:
- New: `crates/hologram-ai-common/src/opt/swiglu_fusion.rs`
- Modify: `crates/hologram-ai-common/src/opt/pipeline.rs` (register pass after AttentionFusion)
- Modify: `crates/hologram-ai-common/src/opt/mod.rs` (export)

**Approach**: Walk graph looking for `Mul(SiLU(MatMul_gate), MatMul_up)` where both MatMuls share an input. Replace 4 nodes with 1 `FusedSwiGLU`. Follow the same pattern-matching style as `attention_fusion.rs` (build `tid_to_node` + `consumers` maps).

**Impact**: Eliminates 3 intermediate tensors per transformer layer. Every LLaMA/Qwen/Mistral/Gemma layer has this pattern.

---

## Task 2: Add+RMSNorm Residual Fusion

**Gap**: `FusedLayerNormResidual` declared in `AiOp` but no pass creates it. TensorBend has `add_rmsnorm` and `three_way_add_rmsnorm`.

**Pattern to match**:
```
residual = Add(x, attn_output)
normed   = RmsNorm(residual, weight, eps)
```
→ `FusedAddRmsNorm(x, attn_output, weight, eps)` with residual as pass-through output

**Files**:
- Modify: `crates/hologram-ai-common/src/opt/rmsnorm_fusion.rs` (extend existing pass)
- May need: new `FloatOp` variant `AddRmsNorm { size, epsilon }` in hologram base
- May need: kernel in hologram base `crates/hologram-exec/src/float_dispatch/norm.rs`

**Prerequisite**: hologram base needs an `AddRmsNorm` FloatOp + kernel (cross-repo change).

**Impact**: Eliminates 1 intermediate tensor + 1 dispatch per residual connection (2x per transformer layer = 2 × N_layers savings).

---

## Task 3: Fused QK-Norm + RoPE + KV-Store

**Gap**: TensorBend's most aggressive fusion — `fused_split_qknorm_kvstore` — combines QKV split, Q/K RMSNorm, RoPE, and KV cache write in one dispatch. hologram-ai handles these as 5-7 separate graph nodes.

**Pattern to match**:
```
[q, k, v] = Split(qkv_proj)          // or separate MatMuls
q_norm    = RmsNorm(q, q_weight)      // optional (Qwen has this, LLaMA doesn't)
k_norm    = RmsNorm(k, k_weight)      // optional
q_rope    = RotaryEmbedding(q_norm)
k_rope    = RotaryEmbedding(k_norm)
kv_write  = KvSlotWrite(k_rope, v)
attention = GQA(q_rope, k_cached, v_cached)
```

**Recommended approach**: Extend `FloatOp::Attention` with optional pre-processing flags rather than a new op:
```rust
Attention {
    head_dim, num_q_heads, num_kv_heads, scale, causal, heads_first,
    // New fields:
    qk_norm: bool,        // Apply RMSNorm to Q/K before attention
    rope: bool,           // Apply RoPE before attention
    rope_base: f32,
}
```

**Files**:
- New: `crates/hologram-ai-common/src/opt/attention_preprocess_fusion.rs`
- Modify: `crates/hologram-ai-common/src/ir/op.rs` (extend GQA metadata)
- Modify: hologram base `FloatOp::Attention` (add optional fields)
- Modify: hologram base `crates/hologram-exec/src/float_dispatch/attention.rs` (inline norm+rope in kernel)

**Impact**: Eliminates 5-7 intermediate tensors + dispatches per layer. Highest per-layer savings of all three tasks.

---

## Task 4: MatMul + Activation Fusion

**Status**: NOT implemented. `FusedFloatChain` only handles unary elementwise ops (`is_elementwise_unary()` explicitly excludes MatMul).

**Pattern**: `MatMul(a, b) → ReLU/GELU/SiLU` → fused `MatMulRelu`/`MatMulGelu`

**Why**: Avoids materializing MatMul output buffer; apply activation inline during output write. Better cache locality.

**Files**:
- New FloatOp variants: `MatMulRelu { m, k, n }`, `MatMulGelu { m, k, n }` in hologram base `crates/hologram-core/src/op/float_op.rs`
- Kernel: hologram base `crates/hologram-exec/src/float_dispatch/matmul.rs` — apply activation in inner loop
- Fusion pass: new file in hologram-ai-common or extend float_fusion in hologram base

**Impact**: Medium — applies to FFN layers outside the SwiGLU path (e.g., output projections in some architectures).

---

## Task 5: Concat + MatMul Fusion

**Status**: NOT implemented. Concat and MatMul are independent ops.

**Pattern**: `Concat([h1..hN]) → MatMul(_, W_out)` → `ConcatMatMul` that avoids materializing concatenated heads buffer.

**Files**:
- New FloatOp variant in hologram base `crates/hologram-core/src/op/float_op.rs`
- Kernel in hologram base `crates/hologram-exec/src/float_dispatch/matmul.rs`

**Impact**: Low-medium — once per attention layer (multi-head output projection).

---

## Task 6: F16 Compute Variants

**Status**: NOT implemented. F16 is used only for weight storage/scale factors. All arithmetic is F32.

**What TensorBend does**: `_f16` variants for all major kernels using `vec4<f16>` — 50% bandwidth reduction.

**Where to add**:
- New: `float_dispatch/matmul_f16.rs`, `norm_f16.rs`
- Extend dispatch routing in hologram base `crates/hologram-exec/src/float_dispatch/mod.rs` by output dtype
- Consider `half` crate for CPU F16 arithmetic

**Note**: CPU-side F16 has limited SIMD support (no native F16 arithmetic on x86). Most impactful when GPU backend exists. On CPU, mixed precision (F16 storage, F32 compute with F16↔F32 conversion) is more practical.

**Impact**: Medium on CPU (bandwidth savings), high on GPU (native F16 compute).

---

## Task 7: Online Softmax Benchmarking

**Status**: Implemented but platform-conditional. Online softmax in hologram base `crates/hologram-exec/src/float_dispatch/attention.rs` only used on non-macOS (`#[cfg(not(all(feature = "accelerate", target_os = "macos")))]`). On macOS, falls back to BLAS-based path that materializes full scores matrix.

**Action**: Benchmark online softmax vs BLAS path for decode (seq=1). For single-token decode, online softmax avoids allocating `[1, seq_k]` scores buffer and may be faster than BLAS dispatch overhead.

**Files**:
- hologram base `crates/hologram-exec/src/float_dispatch/attention.rs` — make path selection runtime-configurable or always use online softmax for seq_q=1
- Add benchmark in `hologram-bench/benches/`

**Impact**: Low effort, potentially significant for decode throughput on macOS.

---

## Task 8: Additional TensorBend-inspired observations

- **PARO quantization**: Learned orthogonal rotation before/after quantized matmul to reduce error (similar to QuIP#/AQLM). Track for future quantization work, not urgent.
- **DeltaNet kernel**: Linear attention variant — shows non-transformer architectures are viable targets.
- **Batch-of-2 kernel variants**: Brute-force speculative decoding optimization. Relevant when hologram supports batch inference.

---

## SPRINT.md Updates

Add new section under **Active: Performance**:

```markdown
### P4: Compiler fusion passes (TensorBend-inspired)
- [ ] SwiGLU fusion pass — pattern-match gate/up/silu into FusedSwiGLU
- [ ] Add+RMSNorm residual fusion — extend rmsnorm_fusion.rs
- [ ] QK-Norm + RoPE + KV-Store pre-attention fusion (design)
```

Add to **Long Term: Performance**:

```markdown
- [ ] MatMul + Activation fusion (MatMulRelu, MatMulGelu)
- [ ] Concat + MatMul fusion (multi-head output projection)
- [ ] F16 compute kernels
- [ ] Online softmax: benchmark vs BLAS for decode, make configurable
- [ ] GPU backend: trait Kernel abstraction at tape level
- [ ] GPU backend: Metal MatMul + Attention kernels
- [ ] GPU backend: CUDA MatMul + Attention kernels
- [ ] GPU backend: WebGPU via wgpu crate
```

---

## Implementation Order

1. **SwiGLU fusion pass** — lowest risk, ops exist on both sides, pure compiler change
2. **Add+RMSNorm fusion** — extends existing pass, needs hologram base FloatOp
3. **Online softmax benchmarking** — small investigation, informs attention kernel work
4. **QK-Norm+RoPE+KV-Store fusion** — design first, implement after tape executor is wired
5. **MatMul+Activation fusion** — after tape executor proves execution model
6. **F16 compute** — after GPU backend trait is in place
7. **Concat+MatMul fusion** — lowest priority, smallest impact
8. **GPU backend** — multi-sprint initiative, starts with `trait Kernel` refactor

## Verification

- `cargo test` passes after each change
- `cargo clippy -- -D warnings` clean
- TinyLlama compilation shows fused ops replacing decomposed patterns (log node counts before/after)
- Conformance tests still pass (node-by-node matches ORT)
- No `println!`/`eprintln!` — use tracing
- No `.unwrap()` — use `.expect("descriptive message")`
