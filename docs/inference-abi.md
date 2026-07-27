# Inference ABI

The modality-neutral, versioned invocation contract (ADR-0004). Rust types
live in `hologram-ai-core`; this document is the language-neutral statement.

## Discovery

A model's manifest is authoritative. Callers enumerate `operations`:

```text
OperationDescriptor {
  name: string                 // e.g. "generate", "predict", "embed",
                               // "classify", "transcribe" (non-exhaustive)
  inputs:  [ValueDescriptor]   // named, typed, required/optional
  outputs: [ValueDescriptor]
  streaming: bool
}
ValueDescriptor { name, kind, media_type?, required }
ValueKind = Text | Tokens | Image | Audio | Bytes | ContentRef
          | Structured | Buffer
```

No model is required to implement any particular operation. New operations
MUST NOT require a `.holo` format change or a new layer kind.

## Invocation

```text
invoke(operation, InferenceRequest) -> InferenceCompletion

InferenceRequest {
  values: [(name, Value)],     // matching the operation's input descriptors
  max_output_tokens?: u32
}

Value = Text(string)
      | Tokens([u32])
      | Image { media_type, inline bytes | κ ref }
      | Audio { media_type, inline bytes | κ ref }
      | Bytes { media_type, data }
      | ContentRef { media_type, kappa }   // "blake3:<64 hex>"
      | Structured(canonical bytes)

InferenceCompletion {
  finish_reason: EndOfSequence | OutputLimit | Abstained | Cancelled
  status: Exact | Graph | Novel | none     // none when abstained w/o output
  widened: bool
  output: [(name, Value)]
  witness?: { kind, bytes }
}
```

Abstention is a successful typed completion with
`finish_reason = Abstained`; no token is fabricated.

## Low-level allocation-free entrypoints

```text
predict_next_into(&request, &mut prediction)
generate_into(&request, &mut token_buffer, &mut event_sink)
    -> GenerationStatus
```

Caller-owned buffers only; zero heap allocation per step after session
initialization (ADR-0009). Convenience APIs (text in/out, SDK marshaling)
allocate at the outer boundary but call this same engine path.

## Streaming

`EventSink::on_event(StreamEvent)` with
`Token(u32) | Text(string) | Status(..) | Widened | Finished(..)`.
Sinks are called from the steady-state path and must be cheap.

## κ content references

Large media SHOULD travel by content reference (`ContentRef`, or
`Image`/`Audio` with a `Ref` payload) so application layers and SDK callers
do not copy bulk bytes. Resolution of a κ reference to bytes is a Hologram
store concern; the inference ABI treats the label as opaque.

## Errors

All failures are `AiError { category, message }` with stable numeric
category codes 1–20 (append-only) — see `docs/sdk-integration.md` for the
per-language mapping.
