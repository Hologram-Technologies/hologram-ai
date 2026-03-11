# 004 — ONNX Last Mile: Full Op Coverage + Subgraph Support

**Status:** In Progress
**Branch:** `feat/onnx-last-mile`
**Date:** 2026-03-11

## Context

hologram-ai compiles TinyLlama 1.1B end-to-end (ONNX → .holo), but the ONNX
importer maps only 75 of ~170 standard ops. This blocks CNN/vision models
(no Conv/Pool/Resize), quantized pipelines, and control flow models
(If/Loop/Scan). Goal: **any ONNX model imports and compiles**.

Secondary goal: map ONNX subgraphs to hologram's native `SubgraphDef` +
`CallSubgraph` + `flatten_subgraph` mechanism.

## Key Architectural Decisions

- All new ops are **AiOp variants** that lower to hologram's `FloatOp`. hologram-ai
  is a compiler only (ADR-0016) — emits `GraphOp::Float(FloatOp::...)`, never
  implements kernels.
- ONNX subgraphs map to hologram's **native `SubgraphDef`** — not PipelineWriter
  (that's for independent multi-model archives).
- Models without subgraphs are **unaffected** — `AiGraph.subgraphs` is empty by
  default. Zero runtime cost.
- **ShapePropagation preserved** — all new AiOp variants get `OpCategory::Custom`
  forcing explicit shape inference rules. Settled-shape protection unchanged.

---

## Phase 1: Vision-Critical Ops

New AiOp variants (all `OpCategory::Custom`):

| AiOp | Key params | Shape rule |
|------|-----------|------------|
| `Conv` | kernel_shape, strides, pads, dilations, group, auto_pad | `floor((in + pad - dilation*(k-1) - 1) / stride + 1)` |
| `ConvTranspose` | kernel_shape, strides, pads, output_padding, dilations, group | `stride * (in - 1) + out_pad + dilation*(k-1) - pad + 1` |
| `MaxPool` | kernel_shape, strides, pads, dilations, auto_pad, ceil_mode | Same as Conv |
| `AveragePool` | kernel_shape, strides, pads, count_include_pad, auto_pad, ceil_mode | Same as Conv |
| `GlobalAveragePool` | — | Spatial dims → 1 |
| `Resize` | mode, coordinate_transform_mode, nearest_mode | From scales/sizes input |
| `Pad` | mode (constant/reflect/edge) | Add pad amounts per dim |
| `InstanceNorm` | epsilon | Shape-preserving |
| `LRN` | alpha, beta, bias, size | Shape-preserving |

ONNX mappings: `Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`,
`GlobalAveragePool`, `Resize`, `Upsample`, `Pad`, `InstanceNormalization`, `LRN`.

Add `attr_s()` string accessor to `OpContext`.

Dynamic param resolution for Pad (pads/constant_value from inputs, opset 11+)
and Resize (scales/sizes from inputs, opset 11+).

Lowering: `D::Unsupported` until hologram base adds `FloatOp::Conv2d` etc.

**Files:** `ir/op.rs`, `op_map.rs`, `graph_builder.rs`, `shape_prop.rs`,
`data_prop.rs`, `dispatch.rs`

---

## Phase 2: Utility Ops

| AiOp | Shape rule | Lowering |
|------|-----------|---------|
| `ReduceProd` | Same as ReduceSum | Needs FloatOp |
| `ReduceL1` | Same as ReduceSum | Decompose: Abs + ReduceSum |
| `ReduceL2` | Same as ReduceSum | Decompose: Mul + ReduceSum + Sqrt |
| `TopK` | Axis dim → K | Needs FloatOp |
| `ScatterND` | Output = data shape | Needs FloatOp |
| `CumSum` | Shape-preserving | Needs FloatOp |
| `NonZero` | `[rank, num_nonzero]` (dynamic) | Needs FloatOp |
| `OneHot` | `indices_shape + [depth]` | Decompose: Scatter |
| `DepthToSpace` | `[N, C/bs², H*bs, W*bs]` | Decompose: Reshape + Transpose |
| `SpaceToDepth` | Inverse of DepthToSpace | Decompose: Reshape + Transpose |
| `Compress` | Dynamic on compressed axis | Needs FloatOp |
| `ReverseSequence` | Shape-preserving | Needs FloatOp |

Quantization: `QuantizeLinear` → `Quantize`, `DequantizeLinear` → `Dequantize`,
`MatMulInteger` → `QuantizedMatMul`.

---

## Phase 3: Proto/Type Gaps

- **F64 dtype**: Add to `DType`, map ONNX type 11, cast to F32 at lowering
- **Widening casts**: UINT16→INT32, INT16→INT32, UINT32→INT64, UINT64→INT64
- **Opset validation**: Parse `opset_import`, enforce `max_opset`, version-aware
  op semantics (Squeeze axes attr vs input at opset 13, Reduce axes at opset 18)
- **Optional inputs**: Document `filter(|name| !name.is_empty())`, track
  position-sensitive optional inputs for Clip/Pad

---

## Phase 4: Subgraph Support (If/Loop/Scan)

### hologram already provides:
- `Graph.subgraphs: Vec<SubgraphDef>` — subgraph template storage
- `GraphOp::CallSubgraph(SubgraphId)` — invocation from a node
- `flatten_subgraph(parent, id, input_bindings)` — inlining before scheduling
- `LayerEntrypoint::Subgraph(u32)` — declared in archive format (not yet implemented)

### What we build:
- `AiGraph.subgraphs: HashMap<String, AiGraph>` — named subgraph registry
- `AiOp::If { then_branch, else_branch }`, `Loop { body, max_trip_count }`,
  `Scan { body, num_scan_inputs }`
- Recursive ONNX import: extract `AttributeProto.g`, build child AiGraph,
  convert lexical scope capture to explicit inputs
- Shape prop recurses into subgraphs; optimization passes recurse into subgraphs
- Lowering: child AiGraph → `lower()` → `SubgraphDef` → `register_subgraph()` →
  `CallSubgraph(id)`. Static trip count → `flatten_subgraph()` unrolling.
- Control flow metadata via `LayerDescriptor` properties or dedicated section

---

## Phase 5: Long-Tail + Conformance

- Map RNG, ML, and linear algebra ops to Opaque with warnings
- Verify multi-output ops (TopK, Split, BatchNorm training)
- ONNX conformance test runner (node test suite)

---

## External Dependencies (hologram base)

| Phase | Needed from hologram |
|-------|---------------------|
| 1 | `FloatOp::Conv2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool`, `Resize`, `Pad`, `InstanceNorm`, `LRN` |
| 2 | `FloatOp::TopK`, `CumSum`, `NonZero`, `ScatterND`, `ReduceProd` |
| 4 | Implement `LayerEntrypoint::Subgraph(u32)`, conditional/loop dispatch |

## Verification

| Phase | Test |
|-------|------|
| 1 | ResNet-50, EfficientNet-B0 import + shape prop |
| 2 | BERT (TopK), T5 (Scatter) import + shapes |
| 3 | DOUBLE weights model, opset-7 model |
| 4 | Beam search seq2seq with If/Loop |
| 5 | ONNX backend node test suite |
| All | `cargo test`, `cargo clippy -- -D warnings` |
