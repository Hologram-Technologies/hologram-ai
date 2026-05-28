# ADR-0018: UOR-native declarative ingestion

**Status:** Accepted
**Date:** 2026-05-28
**Relates to:** ADR-0002 (canonical IR), ADR-0003 (import boundary), ADR-0005 (runtime boundary), ADR-0016 (compiler-only); upstream ADR-055 (UOR-native op taxonomy, no fallbacks), ADR-056 (UOR-native completion of layout/indexed/gradient ops); CONFORMANCE classes **IM**, **LW**, **CF**, **EE**

---

## Context

hologram-ai's ONNX→Graph translation is currently a stack of imperative
passes — 24 fusion/normalization/injection passes under
`crates/hologram-ai-common/src/opt/` plus per-op desugars in
`lower/builder.rs`. Each pass is hand-tuned for patterns from a specific
ONNX exporter or a specific architecture (Llama-style unbiased QKV vs
Qwen2-style biased QKV, GQA ratio 3 vs 7, with/without QK-norm, …). A new
architecture surfaces a new gap; a new gap is a new code change. This is
the bespoke imperative trap and it is not UOR-native.

UOR-addr 0.2.0 demonstrates the alternative: recursively-grammared formats
declare a canonical form derived from the format's authoritative spec, a
shared format-independent ψ-tower consumes the canonical form, and new
formats land by writing the canonical form — not by writing a parser or
downstream consumer.

## Decision

hologram-ai's ONNX→Graph translation is a **UOR-native declarative
ingestion**: a canonical-input form, a canonical typed view, a
declarative rule set, and a canonical target. There are no
architecture-specific passes. There are no order-dependent pass
pipelines. Every rule is verified against an external authoritative
source.

### Canonical-input form

ONNX bytes are ingested via `uor_addr::onnx::canonicalize` into the
flat skeleton defined by uor-addr's onnx realization. The skeleton
collapses protobuf field-order, topological-ordering, raw-data
vs. typed-data freedom, and node-name freedom into one byte-identical
form. (`uor_addr::onnx::CANONICAL_FORM_VERSION = 2`.) The skeleton is
the boundary at which raw bytes become structured content.

### Canonical typed view

The skeleton walks into a typed, structurally-bounded `OnnxModel` —
an algebraic data type mirroring the ONNX IR spec at the level the
spec itself defines: nodes, attributes, initializers, value-info,
opset imports. No fusion, no inference, no normalization — a typed
view of the canonical bytes.

### Canonical target form

`hologram_graph::Graph` over the closed `hologram_ops::OpKind` catalog.
This is hologram's invariant (ADR-055). hologram-ai emits only members
of this catalog; the type system enforces it.

### Declarative rule set

`OnnxModel → hologram_graph::Graph` is a rule set, not a pass pipeline.

- **Op-name rules** map ONNX op-type strings (`"MatMul"`,
  `"LayerNorm"`, `"GroupQueryAttention"`, …) to canonical `OpKind`
  constructions. Each rule carries: the ONNX spec link it implements,
  the input/output shape contract declared (not inferred), and the V&V
  witness — a test in `onnx_spec_conformance.rs` that compares the
  lowered output to the ONNX backend node-test corpus.

- **Pattern rules** match canonical sub-graphs and rewrite them to a
  canonical fused `OpKind`. Each pattern declares: the ONNX-spec-level
  shape of the input pattern, the canonical replacement (an `OpKind`
  with explicit attrs), and the V&V witness — a test that verifies
  ONNX Runtime logit parity on a model using this pattern.

  Architecture-specific variants (biased vs unbiased projections,
  with/without QK-norm, GQA ratio computed from declared kv_heads,
  …) are declared alternates in the rule schema, not separate code.

The rule engine applies rules to fixed-point: each rule either
matches and rewrites, or doesn't. The result is independent of rule
order; rules are confluent on the canonical form. A rule that breaks
confluence is rejected at rule-set load time.

### External verification

Every rule's correctness is established against an external
authoritative source — never against hologram-ai's own output:

- **Op-name rules** ← the ONNX operator spec + the official ONNX
  backend node-test corpus (`onnx_spec_conformance.rs`).
- **Pattern rules** ← ONNX Runtime logit parity (CONFORMANCE class
  EE-3) on a real pretrained model that uses the pattern.
- **Closed catalog** ← `lower::dispatch` exhaustive over `AiOp`
  (compile-time enforced); the closed `OpKind` set (ADR-055). The
  rule engine emits only canonical `OpKind`s.

A rule without an external witness is not a rule. A failing external
witness means the rule is wrong — never that the witness is wrong.

### No silent fallback

Like ADR-055: a rule matches correctly or the ingestion **refuses**.
An unknown ONNX op or an unmatched pattern is a hard error citing the
ONNX spec and the rule-set. No approximation, no opaque pass-through,
no "default behaviour."

## Surface

```
crates/hologram-ai-onnx/
  src/
    canonical.rs   // OnnxModel — typed walk of uor_addr::onnx skeleton
    lib.rs         // ONNX bytes → canonicalize → OnnxModel

crates/hologram-ai-common/
  src/
    rules/
      mod.rs           // RuleSet + matcher + confluence check
      op_rules.rs      // op-name rules + ONNX spec citations
      pattern_rules.rs // pattern rules + ORT-parity citations
    ingest.rs          // OnnxModel → Graph via RuleSet (replaces lower/builder.rs + opt/*)
```

`hologram-ai-common::opt/` is deleted. `hologram-ai-common::lower/`
collapses to the rule engine (dispatch + builder are folded into
the rule-driven path).

## Consequences

- A new architecture is supported by adding rules and witnesses,
  never by editing engine code. The Qwen2 `SlotOutOfRange` failure
  becomes "the biased-QKV variant of the attention pattern is not
  yet declared" — a missing rule, not a code defect.
- The V&V framework drives the implementation: a model is supported
  iff its ORT logit-parity test passes; the test passes iff the rule
  set covers every pattern in the model. The rule set grows by
  closing V&V gaps, not by reading new exporters' source code.
- The closed-OpKind catalog (CF-2), the ONNX-spec conformance
  (IM-1/LW-1/LW-2), the f64-reference checks (LW-2), and the ORT
  logit-parity (EE-3) remain the load-bearing witnesses. No new V&V
  axis is required; the existing axes become both the spec **and**
  the gate.
- hologram-ai-common loses its imperative pass machinery. The shape
  / value propagation passes (`shape_inference`, `shape_prop`,
  `data_prop`, `const_eval`, `dead_node`) stay — they implement spec
  semantics (ONNX shape inference + constant folding), which are
  themselves declarative relative to the spec — but they run inside
  the rule engine, not as a sequential pass list.

## What this is not

- Not e-graph saturation. The rule engine is a confluent fixed-point
  rewrite over a typed canonical form — the same paradigm as
  `Graph::desugar_composites` (upstream ADR-055) and uor-addr's
  ψ-tower, applied to architecture-pattern matching. An e-graph
  matcher backend (`egg`) is an implementation choice inside the
  engine; it is not the architectural decision.
- Not a new format. ONNX stays the input format. GGUF will fall
  out of the same machinery once GGUF gets its rule set
  (`uor_addr::gguf` already provides the canonical form).
- Not a port of upstream. hologram remains the compile + execute
  runtime (ADR-055/056). This ADR is about hologram-ai's lowering
  layer.

## References

- uor-addr 0.2.0 — `uor_addr::onnx::canonicalize` + the ψ-tower
  realization for protobuf/JSON/CBOR/CCMAS/GGUF/ONNX.
- Upstream ADR-055 — UOR-native op taxonomy, no silent fallbacks.
- Upstream ADR-056 — UOR-native completion of layout/indexed/gradient ops.
- CONFORMANCE.md — IM, LW, CF, EE classes and their witnesses.
- specs/docs/lowering.md — the existing lowering's authoritative
  description; aligned with the rule engine here.
