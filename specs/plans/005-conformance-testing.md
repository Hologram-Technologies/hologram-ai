# Plan 005: Production-Ready Conformance Testing & Validation System

## Context

hologram-exec has 2,600 lines of hand-written numerical kernels (57 FloatOp
variants, 42 dispatch functions) with **zero external validation**. The only
test strategy is 12 inline unit tests checking properties like "softmax sums
to 1.0" — no comparison against any reference implementation.

Every bug found so far (Q4_0 nibble ordering, Q6_K indexing, RoPE convention,
attention transposes) was discovered by running a full model end-to-end and
comparing against llama.cpp via print statements. This doesn't scale.

**Goal**: Build a permanent testing architecture that validates **every kernel
in hologram-exec** against a reference implementation — not just AI ops but any
`FloatOp` variant, present or future. The architecture is kernel-agnostic:
Layer A covers all ops via pure Rust. Layer B uses ONNX Runtime as an oracle
for ONNX-mappable ops. Alternative reference oracles can be plugged in for
non-AI ops via the same comparator trait.

## Architecture: Three Layers

### Layer A — Pure Rust Reference Tests (hologram-exec, no new deps)

Expand inline `#[cfg(test)]` in `float_dispatch.rs` from 12 → ~80 tests.
Three categories per op:

1. **Known-answer**: Hand-computed expected outputs for small inputs
2. **Property**: Mathematical invariants (softmax sums to 1, relu >= 0, etc.)
3. **Numerical stability**: NaN, inf, subnormals, zero-length edge cases

Exhaustive match ensures new FloatOp variants can't be added without tests.

**Files:**
- `hologram/crates/hologram-exec/src/float_dispatch.rs` — expand `mod tests`
- `hologram/crates/hologram-exec/tests/float_conformance.rs` — property/stability

### Layer B — ORT Cross-Validation (new crate, `ort` dev-dep)

New crate: `hologram-ai/crates/hologram-ai-conformance/`

For each ~60 ONNX-mappable FloatOp variant:
1. Build single-op ONNX model in memory (no files)
2. Run through ONNX Runtime via `ort` crate
3. Run through `dispatch_float()` with the same inputs
4. Compare outputs with per-op-category tolerances

For non-ONNX ops (RmsNorm, FusedSwiGLU, RotaryEmbedding, Attention), build
composite ONNX models from primitives implementing the same math.

**Tolerance strategy** (numpy `allclose` semantics):
- Elementwise unary/binary: atol=1e-6, rtol=1e-5
- Boolean/compare ops: exact match
- MatMul/Gemm/Conv: atol=1e-4, rtol=1e-3
- Softmax/Norm: atol=1e-5, rtol=1e-4
- Attention (multi-op): atol=1e-3, rtol=1e-2

### Layer C — Model-Level Validation (`validate` CLI command)

Fill existing stub at `hologram-ai/crates/hologram-ai/src/validate.rs`.

```
hologram-ai validate --model resnet18.onnx [--tolerance normal]
```

Flow: Import ONNX → run ORT (capture all intermediates) → run hologram
node-by-node → report first divergence + summary.

### Quantization Conformance

- **Tier 1**: Cross-validate hologram-ai-quant vs hologram-exec dequantize
- **Tier 2**: Pre-computed golden vectors from ggml (Python script → JSON)

## CI Strategy

| Tier | Trigger | Time | What |
|------|---------|------|------|
| 1 | Every PR | <30s | 80+ inline tests, property tests, quant cross-validation |
| 2 | Every PR | <2min | ORT single-op conformance (~60 tests) |
| 3 | Nightly | <30min | Full model validation (MobileNet, TinyLlama ONNX) |

## Implementation Sequence

1. Expand hologram-exec inline tests (Layer A)
2. Create hologram-ai-conformance crate (Layer B)
3. Complex ops + quantization conformance
4. Validate CLI command (Layer C)
5. CI + model-level tests
