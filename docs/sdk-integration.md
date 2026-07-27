# SDK integration

Public branding is Hologram (ADR-0007). `hologram-ai` provides the Rust
implementation; Hologram's FFI and generated SDK infrastructure expose it.

## Namespaces

```text
Rust:       hologram_ai
Python:     hologram.ai
TypeScript: @tryhologram/sdk (AI namespace)
Node:       @tryhologram/native
Browser:    @tryhologram/wasm (future — same binding protocol; ADR-0008)
```

## Binding protocol

Hologram's FFI gains `hologram_ai_*` exports (feature-gated; see
`docs/upstream/hologram-contract.md`):

- lifecycle: `hologram_ai_app_load_path`, `hologram_ai_app_load_bytes`,
  close;
- acquisition/compilation: `hologram_ai_download`,
  `hologram_ai_compile_huggingface`, `hologram_ai_compile_source`;
- discovery: `hologram_ai_model_count`, `hologram_ai_model_entry`,
  capability/provenance inspection as JSON;
- sessions: `hologram_ai_session_open`, `hologram_ai_session_invoke_json`
  (request/response JSON envelopes), `hologram_ai_session_close`;
- errors: appended stable code range mapped from
  `hologram-ai-core::ErrorCategory` (codes 1–20);
- capability probing via the existing `FEATURES` mechanism
  (`ai-download`, `ai-compile`, `ai-session`, …).

JSON is used at the FFI edge for binding simplicity; the canonical binary
schemas in `hologram-ai-core` remain the normative wire forms inside
bundles.

## Error mapping

| Code | Category | Python | TypeScript |
|---|---|---|---|
| 1 | invalid-argument | `hologram.ai.InvalidArgumentError` | `InvalidArgumentError` |
| 3 | authentication | `AuthenticationError` | `AuthenticationError` |
| 4 | unsupported-model | `UnsupportedModelError` | `UnsupportedModelError` |
| 8 | compile | `CompileError` | `CompileError` |
| 14 | integrity-mismatch | `IntegrityMismatchError` | `IntegrityMismatchError` |
| 15 | model-selection | `ModelSelectionError` | `ModelSelectionError` |
| 17 | engine-init | `EngineInitError` | `EngineInitError` |
| 19 | cancelled | `CancelledError` | `CancelledError` |

(Full mapping: one error class per category; table abbreviated. Codes are
append-only ABI.)

## Examples

Python:

```python
import hologram

hologram.ai.compile_huggingface(
    repository="owner/model", revision="<full-sha>", output="model.holo",
)

app = hologram.App.load("model.holo")
model = app.ai.model("ai.default")
session = model.session()
result = session.generate(
    prompt="Explain content addressing.", max_output_tokens=128,
)
print(result.text, result.finish_reason, result.status)
```

TypeScript/Node:

```ts
import { HologramApp, ai } from "@tryhologram/sdk";
import * as native from "@tryhologram/native";

await ai.compileHuggingFace(
  { repository: "owner/model", revision, output: "model.holo" },
  native,
);

const app = await HologramApp.load("model.holo", native);
const model = app.ai.model("ai.default");
const result = await model.invoke("generate", {
  prompt: { kind: "text", value: "Explain content addressing." },
  maxOutputTokens: 128,
});
```

The TypeScript SDK never imports Node modules directly; native and future
WASM adapters implement the same `NativeBinding` protocol.
