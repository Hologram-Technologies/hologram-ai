# Architecture — hologram-ai

> **Status: authoritative.** This document describes the *correct* architecture
> of `hologram-ai` as a UOR-native client of hologram 0.5.0. It is the spec the
> implementation conforms to — not a migration plan. Conformance is enforced by
> [`CONFORMANCE.md`](../../CONFORMANCE.md) and verified by `just vv`
> ([`VERIFICATION.md`](../../VERIFICATION.md)).

## 1. System context

hologram-ai is the **AI front-end** for the hologram runtime. hologram 0.5.0 is
a declarative, UOR-native compute substrate: it has zero knowledge of AI model
formats and operates **only over canonical forms** — the closed
`hologram_ops::OpKind` catalog, `ConstrainedTypeShape` dtypes, content-addressed
(`uor-addr` κ-label) buffers. That canonicality is the basis of its performance
guarantees (content-addressed compute elision, zero-movement buffers,
compile-time weight layout, structural fusion).

hologram-ai owns everything *above* hologram's graph layer and nothing below it:

| hologram-ai owns | hologram owns |
|---|---|
| model file parsing (ONNX, GGUF, safetensors) | tensor arithmetic + kernels |
| the AI IR (`AiGraph` / `AiOp`) and optimization passes | graph representation, scheduling, compilation |
| lowering `AiGraph` → canonical hologram `Graph` | execution (`InferenceSession`) + buffer pool |
| tokenization, sampling, the generation loop | archive format + content addressing |
| model-architecture knowledge (LLaMA, Mistral, …) | fusion, elision, warm-start, weight layout |
| quantization decisions | dtype/shape/op canonical semantics |

**Principle:** hologram-ai *declares*, it does not *dispatch*. It translates a
model into canonical hologram terms and hands them over. It never reimplements
compute, never carries a parallel op/tensor representation, and never inserts a
shim to bridge an impedance mismatch — a mismatch means the translation is
wrong, not that a bridge is needed.

## 2. Structural contract

hologram compiles to wasm, embedded (bare-metal), and platform targets, and its
applications are `no_std`, allocation-free at runtime, and
zero-copy/zero-movement/zero-cost. hologram-ai upholds the same contract, split
the way hologram splits its own lib stack from its CLI:

- **Runtime core** (`no_std` + `alloc`, on-device) — the pieces that run during
  inference on wasm / embedded / platform targets: `hologram-ai-quant` (block
  dequantization) and the tokenizer **encode/decode** path. These build on
  `wasm32-unknown-unknown` and `thumbv7em-none-eabi`. The inference compute
  itself is hologram's `InferenceSession` (a `no_std`, zero-movement,
  content-addressed pool) operating on the compiled `.holo`; hologram-ai adds no
  allocating runtime layer over it.
- **Host shells** (`std`, compile/author-time) — model import (ONNX protobuf,
  GGUF), the **lowering / optimization / compilation** pipeline
  (`hologram-ai-common`, the bulk of the AI front-end), the downloader,
  tokenizer training, the `hologram-ai` CLI, the desktop app, and the
  conformance harness. Compilation produces the `.holo`; it never runs
  on-device, so it may use `std` and allocate freely.

The boundary is enforced by V&V classes NS (portability), ZA (zero alloc), ZM
(zero movement), and CF (canonical-forms-only).

## 3. Crate structure

```
hologram-ai-common   runtime core — AiGraph IR, optimization passes, lowering, memory plan
hologram-ai-quant    runtime core — Q4_0/Q8_0 dequant kernels (GGML-conformant)
hologram-ai-tokenizer runtime core (encode/decode) + host shell (train/json load)
hologram-ai-onnx     host shell — ONNX protobuf → AiGraph (Grounding at the byte boundary)
hologram-ai          host shell — CLI + facade: import → optimize → lower → compile → run
hologram-ai-conformance host shell — V&V harness (external authorities + structural checks)
```

hologram-ai depends on the individual hologram member crates — there is no
umbrella `hologram` crate in 0.5.0:

- `hologram-types` — canonical dtypes (`DTypeKind`, marker structs).
- `hologram-ops` — the closed `OpKind` catalog (81 variants, fieldless).
- `hologram-graph` — `Graph`, `Node`, `GraphOp`, `ConstantStore`, shape/dtype registries, per-node attr tables.
- `hologram-compiler` — `compile(graph, BackendKind, WittLevel) → archive`.
- `hologram-exec` — `InferenceSession::load(archive).execute(inputs)`.
- `hologram-archive` — `.holo` format, `uor-addr` content addressing.
- `hologram-backend` — CPU/Metal/WebGPU kernels (host-shell concern).

## 4. The IR: `AiGraph` / `AiOp`

The AI IR stays — it is hologram-ai's model-level representation, richer than the
runtime graph (named tensors, dynamic dims, model metadata, format-specific
attributes). Import produces an `AiGraph`; optimization passes rewrite it;
lowering translates it to a canonical hologram `Graph`. The IR is *not* a mirror
of hologram's graph — it is a higher-level surface that compiles down to it.

## 5. Lowering: `AiGraph` → canonical `Graph`

This is the load-bearing translation and the core of the UOR-native design.

### 5.1 Ops are canonical and fieldless

Every `AiOp` lowers to one or more `GraphOp::Op(OpKind)` nodes. `OpKind` is a
fieldless tag; **its parameters are never carried on the op**. They are supplied
by exactly three canonical mechanisms, and lowering chooses the right one:

1. **Derived from operand shapes** — matmul `m/k/n`, softmax/normalization axis,
   reduction axes, conv output geometry. hologram-ai supplies each node's
   concrete `output_shape` (a `ShapeId` interned in the graph's shape registry)
   and operand shapes; hologram's compiler derives the op params from them. This
   is why hologram-ai no longer computes or carries these params.
2. **Sparse per-node attribute tables** — params not recoverable from shape:
   quantization (`QuantAttrs`), convolution stride/pad/kernel (`ConvAttrs`),
   GEMM α/β (`GemmAttrs`), LRN window (`LrnAttrs`). Set via
   `graph.set_*_attrs(node_id, …)`.
3. **Extra operands** — values that are tensors in their own right: normalization
   gamma/beta, attention Q/K/V, RoPE cos/sin tables, Clip lo/hi. Passed as
   additional `Node.inputs`.

### 5.2 Every op is fully realized — there is no failure path

hologram-ai is fully UOR-native: **every `AiOp` has a complete canonical
realization.** There are no unsupported ops, no `TODO`s, and no runtime
failure/fallback path. The `OpKind` catalog is closed and has no `Cast`,
`Gather`, `Embed`, `Split`, `Range`, `ArgMax`, `TopK`, `Scatter`, … — so
hologram-ai realizes each of these by **desugaring into a canonical `OpKind`
pipeline**:

- `Embed` / `Gather` / `GatherND` → row/element selection via `Slice`+`Concat`;
- `Split` → N `Slice`s; `Tile` → repeated `Concat`; `Cast` → the numeric
  primitives / `Dequantize`;
- `ArgMax`/`ArgMin` → `ReduceMax` + `Equal` + index-`iota` selection;
  `TopK` → unrolled argmax-and-mask rounds; `Scatter` → masked `Where`;
- `Einsum` → `Transpose`+`MatMul`+`Reduce`; `BatchNorm` → affine over the
  primitives; `ReduceL1/L2`, `DepthToSpace`/`SpaceToDepth`, `OneHot`,
  `ReverseSequence`, `Compress`, `NonZero` → their canonical primitive forms;
- `Shape`, `Range`, `CausalMask`, `AlibiSlope`, `ConstantOfShape`, `Constant`
  → compile-time constants materialized into the `ConstantStore` (their values
  are known once shapes are concrete, §5.1);
- hologram-ai's own legacy fusions (`MatMulRelu`, `ConcatMatMul`,
  `FusedNormProjection`, …) → **unfused** back into the plain canonical ops, so
  hologram performs the fusion structurally (§5.3).

Data-dependent-shape ops (`NonZero`, `Compress`, dynamic `TopK`) are realized
with statically-bounded output shapes (bounded by the input extent), keeping
them within the canonical, concrete-shape model. The KV-cache ops
(`KvSlotWrite`/`KvSlotRead`) do not exist in the pipeline at all — the KV
injection pass is removed; reuse is content-addressed elision (§5.3).

There is no identity stand-in for a real computation and no pass-through shim:
a desugaring is an exact canonical expansion held to the same external
reference as the op it replaces (V&V class LW).

### 5.3 What lowering no longer does

The canonical model moves three responsibilities into hologram, so the
corresponding hologram-ai machinery does not exist in the correct architecture:

- **Op-parameter resolution** — params come from shapes/attrs/operands (§5.1),
  so there is no `AiOp → FloatOp{m,k,n,size,epsilon,…}` resolver, no
  multi-strategy shape solver, and no runtime shape-projection/recipe context.
  hologram-ai supplies concrete shapes; the compiler does the rest.
- **Operator fusion** — hologram fuses `matmul→activation`, residual adds, etc.
  structurally (content-addressed, the FU invariant). hologram-ai emits the
  plain canonical ops and never emits a pre-fused op.
- **KV-cache** — there is no KV-cache. Autoregressive reuse is **content-addressed
  elision**: a node's output κ-label is derived from its op signature and operand
  labels, so an unchanged decode prefix is recognized and its compute elided
  (the SG/CE invariant). hologram-ai emits no KV read/write ops and plans no KV
  buffer; the legacy KV-cache optimization is abandoned.

### 5.4 Control flow

hologram's graph has no subgraph/call mechanism. Control-flow ops (`If`, `Loop`,
`Scan`) are resolved at compile time: branches are selected or loops unrolled
into the flat graph when statically determinable, and fail loud otherwise. There
is no runtime subgraph dispatch.

## 6. Weights, quantization, and constants

- Small constants (norms, biases) are inline `ConstantEntry { bytes, dtype, shape }`.
- Large weights live in the archive's weight layer (content-addressed by bytes
  via `uor-addr`), borrowed zero-copy at load — not cloned into the constant
  store.
- Quantized weights carry `QuantAttrs` on their consuming node and are
  dequantized by hologram's canonical `Dequantize` op (or LUT-GEMM path).
  hologram-ai-quant's dequant is held to the GGML/ONNX reference (class QZ).

## 7. Compilation and execution

Lowering yields a hologram `Graph`. hologram-ai calls
`hologram_compiler::compile(graph, BackendKind, WittLevel)` to get a `.holo`
archive (the compiler desugars composites, elides invariants, schedules, lowers,
and folds the constant-only cone for warm-start). At inference time hologram-ai
loads the archive into a `hologram_exec::InferenceSession` and calls `execute`
with input buffers. There is no tape-builder API and no `GraphInputs`/`Outputs`
marshalling layer — the session owns the content-addressed buffer pool.

## 8. Model addressing

A model addresses to a `uor-addr` κ-label via hologram's `address_ring`;
multi-component models compose with `compose_model` (order-independent E₈
product). This is the model identity used for dedup and warm-start (class MA).
Byte-level format parsing is confined to `Grounding` impls at the import
boundary; nothing mid-pipeline parses raw bytes.

## 9. Contracts

- **Depends on** hologram 0.5.x member crates (path/version) and the uor stack
  (`uor-foundation`, `uor-prism`, `uor-addr`) transitively.
- **Provides** model compilation (`model → .holo`) and inference to downstream
  consumers (CLI, desktop app, servers).
- Inter-repo contracts are declared in `hologram.repo.yaml`.
