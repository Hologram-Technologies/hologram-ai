# ADR-0002: Introduce a canonical AI model IR (`AiGraph`) above raw Hologram graph IR

- Status: Accepted
- Date: 2026-03-06
- Owners: Architecture

---

## Context

`hologram-ai` must import AI model formats (ONNX, GGUF, GGML) and make them
executable on the Hologram runtime. The key design question is at what level
of abstraction the compilation pipeline should operate.

**Option A:** Import formats directly into `hologram::ExecutionPlan` nodes.
**Option B:** Import into a semantic `AiGraph` IR first, then lower to `ExecutionPlan`.

The primary motivation for Option B is that AI model formats carry semantic
structure — attention heads, normalizations, positional encodings, KV-cache
semantics — that is valuable to preserve through optimization passes before
conversion to primitive execution nodes.

---

## Decision

Introduce `hologram-ai-ir::AiGraph` as the canonical intermediate representation
for all AI model formats in `hologram-ai`.

All format importers (ONNX, GGUF, GGML) emit `AiGraph`. No format-specific
types escape the importer boundary. All optimization passes, memory planning,
and quantization logic operate on `AiGraph`. Lowering to `hologram::ExecutionPlan`
is a single, explicit final step.

---

## Consequences

**Positive:**
- Enables AI-semantic optimization passes (attention fusion, FFN fusion) that
  would require expensive pattern matching on lower-level execution nodes
- Single representation → single optimization and validation target
- Format-specific bugs are contained to importer crates
- Clear architectural boundary: hologram knows nothing about AI formats
- Quantization descriptors, shape annotations, and metadata travel cleanly
  through the pipeline without format-specific leakage

**Negative:**
- Additional translation layer between import and execution
- `AiGraph` must be expressive enough to cover all three formats without
  becoming format-biased; this requires careful op set design

**Neutral:**
- `hologram-ai-ir` becomes the most critical crate in the workspace; changes to
  it affect all other crates

---

## Alternatives Considered

**Option A: Direct lowering from each format to ExecutionPlan**
Rejected. Would scatter format-specific lowering logic throughout the system.
Would make shared optimization passes impossible without format detection.
Would expose hologram to AI-format concerns.

**Option C: Use an existing AI IR (e.g. MLIR, IREE, ONNX as internal IR)**
Rejected. All are external dependencies with their own build requirements and
semantic models. They add complexity without fitting the Hologram-native
philosophy. ONNX as an internal IR would bias the system against GGUF models.
