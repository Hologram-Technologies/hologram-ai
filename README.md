# hologram-ai

Compile Hugging Face models into self-contained `.holo` AI applications and
run them with deterministic, CPU-only R4G1 inference.

hologram-ai is the production integration layer between three projects:

| Project | Owns |
|---|---|
| [`uor-r4`](https://github.com/UOR-Foundation/uor-r4) | transformerless graph compilation, the R4G1 format, scoring, inference, status/abstention, witnesses, and the multiplication-free, allocation-free runtime contract |
| [`hologram`](https://github.com/Hologram-Technologies/hologram) | the `.holo` container, application manifests, layer declarations, archives, the `hologram` CLI, FFI, and SDK infrastructure |
| **hologram-ai** (this repo) | Hugging Face / local source acquisition, immutable revision + cache policy, compile orchestration, deterministic R4 inference bundles, `.holo` `InferenceModel` packaging, model registry and sessions, and the multimodal inference ABI |

**R4G1 is the only inference path.** No transformer runtime, no `MatMul`,
no GPU backend, no ONNX/ORT, no external inference provider, no runtime
network access — and no tensor-plan fallback. See
[ADR-0002](docs/adrs/0002-r4g1-only-inference.md).

## How it works

```text
Hugging Face repo@<full-sha>  or  local model source
   → verified immutable source (content-addressed cache)
   → uor-r4 typed compiler API
   → R4G1 artifacts (validated)
   → deterministic R4 inference bundle (schema v1)
   → InferenceModel layer in a single self-contained .holo archive
   → hologram CLI / Rust / Python / TypeScript
   → uor-r4 R4G1 inference engine
```

The only public compile product is a `.holo` file. R4G1 files are private
work-directory implementation details.

## Rust

```rust
// Compile a pinned Hugging Face revision into one .holo archive.
let compiled = hologram_ai::Compiler::builder()
    .source(hologram_ai::HuggingFaceSource::pinned(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "<full-commit-sha>",
    ))
    .entry("ai.default")
    .cache_dir(".cache/hf")
    .work_dir(".cache/work")
    .compile_to_path("model.holo")?;

// Load, discover, select, and run.
let app = hologram_ai::Application::open_path("model.holo")?;
for model in app.models() {
    println!("{} (engine {})", model.entry(), model.engine());
}
let model = app.model("ai.default")?;      // explicit selection
let mut session = model.session()?;

let completion = session.invoke(
    "generate",
    hologram_ai::InferenceRequest::builder()
        .text("prompt", "Explain content addressing.")
        .max_output_tokens(128)
        .build(),
)?;
println!("{:?}: {:?}", completion.finish_reason, completion.status);
```

Multimodal requests travel through the same generic `invoke` path — inline
or by content-addressed κ reference:

```rust
let completion = session.invoke(
    "generate",
    hologram_ai::InferenceRequest::builder()
        .text("prompt", "Describe this image.")
        .content_ref("image", "image/png", &image_kappa)
        .build(),
)?;
```

Allocation-free low-level APIs (zero heap allocations per step after
session initialization):

```rust
session.predict_next_into(&window, &mut prediction)?;
session.generate_into(&seed, &mut token_buffer, &mut event_sink)?;
```

## CLI

The public CLI is `hologram` (there is no `hologram-ai` binary):

```bash
hologram ai download HuggingFaceTB/SmolLM2-135M-Instruct --revision <full-sha>
hologram ai compile  HuggingFaceTB/SmolLM2-135M-Instruct --revision <full-sha> \
    --output smollm2.holo
hologram ai compile  --source /path/to/model-source --output model.holo
hologram ai inspect  model.holo
hologram ai infer    model.holo --model ai.default --operation generate \
    --prompt "Explain content addressing." --max-output-tokens 128
```

## Python

```python
import hologram

hologram.ai.compile_huggingface(
    repository="owner/model", revision="<full-sha>", output="model.holo",
)
app = hologram.App.load("model.holo")
session = app.ai.model("ai.default").session()
result = session.generate(prompt="Explain content addressing.",
                          max_output_tokens=128)
```

## TypeScript / Node

```ts
import { HologramApp, ai } from "@tryhologram/sdk";
import * as native from "@tryhologram/native";

await ai.compileHuggingFace(
  { repository: "owner/model", revision, output: "model.holo" }, native);
const app = await HologramApp.load("model.holo", native);
const result = await app.ai.model("ai.default").invoke("generate", {
  prompt: { kind: "text", value: "Explain content addressing." },
});
```

## `InferenceModel` layers

A compiled model is a callable AI service layer in `.holo` v4:
`kind = InferenceModel`, `content = κ` of the deterministic model bundle,
`entry = ai.default` (unique callable name), `aux = uor-r4` (engine id).
An application may contain zero, one, or many model layers (multimodal
systems can be one multimodal model or several cooperating layers); a
model-only archive has no primary layer. Modality lives in operation and
value descriptors, never in layer kinds. See
[ADR-0003](docs/adrs/0003-inference-model-layer-semantics.md).

## Platform support

- **Supported now:** native macOS/Linux, Rust API. CLI/Python/TypeScript
  ship via the Hologram FFI patch (see `docs/upstream/`).
- **Deferred:** browser/WASM inference (portable core is already
  wasm32-gated; full browser R4 execution lands after upstream runtime
  work — [ADR-0008](docs/adrs/0008-native-first-browser-later.md)).
- **Not claimed:** real image/audio compilation — the multimodal ABI,
  packaging, and validation are implemented and tested with deterministic
  fixtures, but uor-r4 compiles text models only today.

## Security and determinism

- Bundle and archive bytes are parsed as untrusted input: checked
  arithmetic, size caps, bounded allocation, digest verification before
  engine initialization, capability/processor consistency validation.
- Identical pinned inputs (source revision, bytes, compile options, tool
  revisions) produce byte-identical bundles and `.holo` archives.
- Hugging Face downloads require full commit SHAs by default, resume into
  atomic staging, and never log or persist tokens.

## Documentation

- [docs/architecture.md](docs/architecture.md) — system map
- [docs/rewrite-plan.md](docs/rewrite-plan.md) — plan, pins, phases
- [docs/bundle-format.md](docs/bundle-format.md) — R4 bundle schema v1
- [docs/inference-abi.md](docs/inference-abi.md) — multimodal invocation ABI
- [docs/huggingface.md](docs/huggingface.md) — acquisition and cache policy
- [docs/sdk-integration.md](docs/sdk-integration.md) — FFI/SDK protocol
- [docs/testing.md](docs/testing.md) — gates and test layers
- [docs/adrs/](docs/adrs/) — ADR-0001 … 0009
- [docs/upstream/](docs/upstream/) — uor-r4 / hologram patch contracts

## License

MIT OR Apache-2.0.
