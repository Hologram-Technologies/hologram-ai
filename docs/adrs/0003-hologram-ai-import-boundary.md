# ADR-0003: Format-specific logic is fully contained within importer crates

- Status: Accepted
- Date: 2026-03-06
- Owners: Architecture

---

## Context

`hologram-ai` must handle three model formats: ONNX (protobuf graph),
GGUF (binary metadata + raw tensors, no graph), and GGML (legacy binary).

These formats are structurally different. ONNX contains an explicit operation
graph. GGUF stores weights and metadata but requires architecture recognition
to reconstruct the graph topology. GGML uses a fixed, hardcoded structure.

The risk is that format-specific concerns (protobuf types, GGUF KV metadata
structures, architecture-specific tensor naming conventions) leak into shared
code — optimization passes, the memory planner, the lowering pipeline.

---

## Decision

Format-specific logic is fully contained within its importer crate
(`hologram-ai-onnx`, `hologram-ai-gguf`, `hologram-ai-ggml`).

Each importer exposes exactly one public interface:

```rust
pub fn import_*(input: ..., opts: ...) -> Result<AiGraph>
```

After this function returns, no downstream crate has any knowledge of which
format was used. The `AiGraph` carries no format provenance information in
any type that matters for downstream processing.

Format-specific types (ONNX `ModelProto`, GGUF `GgufMetadata`, etc.) are
private to their importer crate.

---

## Consequences

**Positive:**
- Optimization passes, memory planner, and lowering are format-agnostic
- A new format (e.g. SafeTensors, PyTorch) adds a crate without touching
  anything else in the pipeline
- Format bugs are isolated to one crate
- Testing is simpler: import produces `AiGraph`, test the `AiGraph` downstream

**Negative:**
- Some format metadata (GGUF context_length, rope config) is needed by the
  session layer; this must travel as `AiGraph::metadata` (a generic KV map)
  rather than typed format-specific structs
- The `AiGraph::metadata` map is less type-safe than format-specific types

**Neutral:**
- Importer crates may be large (ONNX has a lot of op coverage); this is
  acceptable given they are independent modules

---

## Alternatives Considered

**Keep format types accessible via a `FormatInfo` enum on `AiGraph`**
Rejected. Creates a dependency on format-specific types in all downstream
crates. Adding a new format would require updating the enum everywhere.

**Share format parsers across importers via a common parsing crate**
Not rejected — acceptable to share byte-level utilities (varint, little-endian
readers) in a private utility crate. But semantic interpretation must remain
per-format.
