# ADR-0004: Quantization is first-class in `AiGraph`; dequantization is explicit in the IR

- Status: Accepted
- Date: 2026-03-06
- Owners: Architecture

---

## Context

GGUF models store weights in quantized formats (Q4_0, Q4_K_M, Q6_K, etc.).
ONNX models may use INT8 or INT4 quantization. Quantized inference is
necessary for practical memory and performance targets on edge devices.

The core design question: should quantization be stripped at import time
(all weights dequantized to f32/f16), or should it be preserved throughout
the pipeline?

**Option A — Strip quant at import:** All weights eagerly dequantized to f32
during import. Simple pipeline; all ops work on f32. High memory cost;
loses the ability to use quantized kernels.

**Option B — Preserve quant throughout IR:** Quantized tensors carry their
`QuantDescriptor` through `AiGraph`. Dequantization is explicit as `AiOp::Dequantize`.
The lowering pass decides whether to keep explicit dequant nodes or fuse them
into quantized GEMM kernels.

---

## Decision

Adopt Option B: quantization is preserved as first-class data throughout `AiGraph`.

Each `TensorInfo` carries both:
- `storage_dtype` — the quantized storage format (Q4_0, INT8, etc.)
- `logical_dtype` — the arithmetic format (F32 or F16)
- `quant: QuantDescriptor` — scale, zero-point, block size, scheme

`AiOp::Dequantize` is an explicit IR node. It appears in the graph wherever
quantized weights must be converted to float for arithmetic.

The `hologram-ai-opt` pass `QuantMatMulFusion` may fuse `Dequantize → MatMul`
into `AiOp::QuantizedMatMul` when the backend supports it.

At lowering time, unfused `Dequantize` nodes remain as explicit plan steps.
No silent upcasting occurs anywhere in the pipeline.

---

## Consequences

**Positive:**
- Memory efficiency: 70B Q4 model loads as Q4 throughout, not upcast to f32
- Correct representation: the IR accurately reflects what the model actually stores
- Backend choice: backends that support quantized GEMM use it; others fall back
- Auditable: dequantization is visible in the IR, not hidden in import code
- Future-safe: adding new quant schemes adds entries to `QuantScheme` and a
  dequant implementation; nothing else changes

**Negative:**
- More complex IR: two dtype fields per tensor instead of one
- Importers must correctly map every quant format to a `QuantDescriptor`
- `hologram-ai-quant` must implement correct dequant for every scheme (correctness risk, see R-02)

**Neutral:**
- The f32 fallback (eager dequant at plan start) remains available as a
  `LoweringOptions::quant_strategy = EagerDequant` option for debugging

---

## Alternatives Considered

**Option A: Strip quant at import**
Rejected. Makes large model inference impractical due to memory cost.
Eliminates any possibility of quantized-kernel dispatch.
Loses the ability to inspect quantized model structure.

**Option C: Represent quant as metadata only, not in op graph**
Rejected. Making dequantization invisible leads to silent precision upgrades
that are hard to reason about and hard to test.
