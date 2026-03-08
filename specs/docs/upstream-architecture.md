# Upstream Architecture Reference — hologram-ai

## Source of Truth

Architecture decisions for the Hologram ecosystem are maintained in:

```
hologram-architecture/
specs/adrs/          — Architecture Decision Records
specs/projects/      — Per-project planning docs
specs/research/      — Research reports
```

The ADRs in that repository are **authoritative**. Any constraint documented
there takes precedence over local conventions in `hologram-ai`.

---

## Relevant ADRs

| ADR | Decision | Impact on hologram-ai |
|-----|----------|------------------------|
| ADR-0001 | Repository boundaries, one-way architecture flow, cross-repository isolation | hologram-ai must not modify hologram-architecture or sibling repos; specs/docs/ is read-only; changes to other repos require specs/plans/ prompts |
| ADR-0002 | Canonical AI model IR (`AiGraph`) above raw Hologram graph IR | All format importers must emit `AiGraph`; no format-specific types escape importers; lowering to `ExecutionPlan` is explicit final step |
| ADR-0003 | Format-specific logic contained within importer crates | ONNX, GGUF, GGML importers expose only `import_*() -> Result<AiGraph>`; downstream crates are format-agnostic |
| ADR-0004 | Quantization is first-class in `AiGraph`; dequantization is explicit | `TensorInfo` carries `storage_dtype`, `logical_dtype`, and `QuantDescriptor`; `AiOp::Dequantize` is an explicit IR node |
| ADR-0005 | `InferenceSession` owns plan and KV-cache; `hologram` owns execution | Session manages AI-specific state; hologram remains AI-agnostic; clean handoff via `KvExecutor` API |
| ADR-0006 | MVP scope is GGUF import + CPU backend + single forward pass | Initial implementation targets LLaMA-family GGUF models with Q4_0 quantization on CPU |

---

## Local Interpretation

hologram-ai applies upstream constraints by maintaining strict layer separation:

1. **Import layer** (`hologram-ai-onnx`, `hologram-ai-gguf`, `hologram-ai-ggml`) — Each importer is self-contained and emits only `AiGraph`. Format-specific types (protobuf structs, GGUF metadata) are private to their crate.

2. **IR layer** (`hologram-ai-ir`) — The canonical `AiGraph` representation is the single optimization target. All passes operate here before lowering Jean.

3. **Lowering layer** (`hologram-ai-lower`) — Translates `AiGraph` to `hologram::Graph + ExecutionSchedule`. This is the only point where hologram types are constructed.

4. **Session layer** (`hologram-ai-session`) — Owns KV-cache, present length, and session options. Delegates execution to `hologram::KvExecutor` without reaching into hologram internals.

5. **Execution boundary** — hologram-ai never bypasses `KvExecutor`; all kernel dispatch flows through hologram's public API and `CustomOpRegistry`.

---

## Constraints This Repo Must Respect

1. **No hologram internals access** — Must use only public `KvExecutor` API, `CustomOpRegistry`, and `BufferArena`; no direct manipulation of hologram's internal scheduling or memory structures.

2. **Explicit dequantization** — Quantized tensors must carry `QuantDescriptor` through the pipeline; dequantization must appear as `AiOp::Dequantize` nodes in the IR, never as silent upcasting.

3. **Format isolation** — No ONNX protobuf types, GGUF metadata structures, or GGML-specific logic may appear outside their respective importer crates.

4. **Single IR target** — All optimization passes must operate on `AiGraph`; no format-specific optimization paths are permitted.

5. **Read-only specs/docs/** — Architecture documentation synced via `holoarch pull` must not be modified locally; changes require upstream proposals in hologram-architecture.

6. **Cross-repo isolation** — Agents working in hologram-ai must not modify hologram, hologram-sandbox, or hologram-architecture directly; required changes must be documented as specs/plans/ prompts.

7. **AI-agnostic hologram** — hologram must have no knowledge of KV-cache, attention heads, tokens, or model formats; all AI semantics terminate at the hologram-ai boundary.
