# ADR-0016: hologram-ai is a Compiler, Not a Runtime

**Status:** Accepted
**Date:** 2026-03-07
**Supersedes:** hologram-ai runtime model as described in architecture.md (pre-0016)

---

## Context

The original `hologram-ai` architecture combined a compiler pipeline (import →
optimize → lower → compile) with a full runtime (session management, KV-cache
buffer ownership, autoregressive generation loop, streaming decoder).

This created several problems:

1. **Duplicated responsibilities.** `hologram` already owns execution (`KvExecutor`,
   `BufferArena`, `ExecutionSchedule`). The `InferenceSession` type in hologram-ai
   was managing execution state that rightfully belongs to either the archive format
   or the caller.

2. **Ignored existing infrastructure.** The `.holo` archive format already supports
   named layer entrypoints via section `0x0002` (`LayerHeader`, `LayerDescriptor`,
   `LayerEntrypoint`). hologram-ai was not using this — it compiled models into a
   single anonymous graph and then managed all dispatch logic itself.

3. **Forced coupling.** Applications that only needed to compile models (batch
   processing, offline compilation pipelines) were forced to link session management
   and streaming code. Applications that wanted custom generation strategies had to
   subclass `InferenceSession`.

4. **Wrong ownership model.** KV-cache is a stateful buffer that lives across
   multiple calls. Placing it inside `InferenceSession` (a hologram-ai type) prevents
   callers from managing their own buffer lifecycle, pooling sessions, or integrating
   with custom allocators.

---

## Decision

**`hologram-ai` is a compiler. Its output is a `.holo` archive. It has no
runtime session type.**

Specifically:

1. **hologram-ai translates foreign model artifacts into `.holo` archives.**
   Input: ONNX protobuf, GGUF binary, GGML checkpoint.
   Output: `.holo` archive consumed by standard hologram loader and executor.

2. **The archive uses `LayerHeader` to declare named execution entrypoints.**
   For autoregressive LLMs, hologram-ai emits at minimum:
   - `"lm.prefill"` — variable-length prompt ingestion subgraph
   - `"lm.decode"` — single-token decode subgraph

   The `LayerDescriptor` declares all input/output tensor ports including the
   KV-cache buffer, which is treated as an explicit mutable in/out parameter.

3. **KV-cache is an explicit buffer parameter, not session state.**
   The KV-cache layout is encoded in a new archive section (`SECTION_LLM_META`,
   `0x0011`). The caller allocates the buffer, and both prefill and decode layers
   accept it as an input and return the updated version as an output.

4. **Token generation is the caller's responsibility.**
   The generation loop (sample next token, check EOS, manage present_len) is
   application-level code. It calls the appropriate layer on each step using
   `hologram`'s standard `HoloLoader` + `KvExecutor` API — no hologram-ai types
   involved.

5. **hologram-ai-session and hologram-ai-stream are removed.**
   They are replaced by the archive's layer entrypoints and by thin caller-side
   utility code (≤ 200 lines) that any application can maintain.

---

## Consequences

### Crate changes

**Removed from hologram-ai:**
- `hologram-ai-session` — `InferenceSession`, `CompiledModel`, `SessionState`,
  `ChatSession`, `ConversationHistory`
- `hologram-ai-stream` — `TokenStream`, `Sampler`, sampling strategies

**Retained (unchanged scope):**
- `hologram-ai-common` — `AiGraph` IR, optimization passes, quantization, KV-cache
  layout planning, lowering to `hologram::Graph`
- `hologram-ai-gguf` — GGUF importer
- `hologram-ai-ggml` — GGML importer
- `hologram-ai-onnx` − ONNX importer
- `hologram-ai-tokenizer` — BPE/SentencePiece/WordPiece (embedded in `.holo` via
  section 0x1001 per ADR-0012)
- `hologram-ai-validate` — validation harness (compile + execute + compare; uses
  `HoloLoader` + `KvExecutor` directly)

**Modified scope:**
- `hologram-ai-cli` — retains `compile`, `inspect`, `validate`; loses `generate`,
  `run` as library commands (may re-implement `generate` as CLI-only code with
  ~100 lines of generation loop using standard hologram API)

### Archive format additions

A new well-known section is added to the `.holo` format and its type is defined
in the **`hologram` crate** (alongside `LayerHeader` and other section types).
hologram-ai constructs and writes this section but does not define the struct.

- **`SECTION_LLM_META` (0x0011)** — LLM inference metadata:
  ```rust
  // Defined in hologram, not in hologram-ai
  pub struct LlmMetaSection {
      pub model_type: LlmModelType,      // LlamaFamily, Bert, Gpt2, etc.
      pub kv_layout: KvCacheLayout,      // n_layers, n_kv_heads, head_dim, max_seq_len
      pub prefill_layer: LayerId,        // which layer is the prefill entrypoint
      pub decode_layers: DecodeLayers,   // Single(LayerId) | Bucketed(Vec<(u64, LayerId)>)
  }
  ```
  `BucketSelector` (also in `hologram`) provides the caller-side utility for
  mapping an actual seq_len to the correct bucketed decode layer.

The `kv_cache` tensor port is declared in the `LayerDescriptor` input/output lists:
```
LayerDescriptor "lm.prefill":
  inputs:  [input_ids: [batch, seq_len] i64, kv_cache: [n_bytes] u8]
  outputs: [logits: [batch, vocab_size] f32, kv_cache: [n_bytes] u8]

LayerDescriptor "lm.decode":
  inputs:  [input_ids: [batch, 1] i64, present_len: [] u32, kv_cache: [n_bytes] u8]
  outputs: [logits: [batch, vocab_size] f32, kv_cache: [n_bytes] u8]
```

### Calling pattern (application-side, ~100 lines)

```rust
// Load the archive
let archive = HoloLoader::load("model.holo")?;
let llm = archive.section::<LlmMetaSection>(SECTION_LLM_META)?;

// Allocate KV-cache buffer
let mut kv_cache = vec![0u8; llm.kv_layout.total_bytes as usize];
let mut present_len: u32 = 0;

// Prefill
let (logits, kv_cache) = KvExecutor::execute_layer(
    &archive, llm.prefill_layer,
    inputs!{
        "input_ids" => &prompt_token_ids,
        "kv_cache"  => &kv_cache,
    },
)?;

// Decode loop
let mut tokens = vec![argmax(&logits)];
loop {
    present_len += prompt_len as u32;
    let (logits, new_kv) = KvExecutor::execute_layer(
        &archive, llm.decode_layer,
        inputs!{
            "input_ids"   => &[tokens.last()],
            "present_len" => &present_len,
            "kv_cache"    => &kv_cache,
        },
    )?;
    kv_cache = new_kv;
    present_len += 1;
    let tok = argmax(&logits);
    if tok == eos_id { break; }
    tokens.push(tok);
}
```

### Impact on existing sprints

The following sprint sections are **affected by this decision**:

| Sprint | Status | Impact |
|--------|--------|--------|
| sprint-003 (Week 3: KV-cache + session + streaming) | Revised | KV-cache lowering retained; `InferenceSession`/`TokenStream` removed; replaced by multi-layer lowering (prefill + decode subgraphs) |
| sprint-009 (Bucketed compilation) | Revised | Bucketed variants become multiple named layers in `LayerHeader` rather than session-internal `BucketedCompiledModel` |
| sprint-010 (Phase 2 polish) | Revised | Exit criteria updated; remove session/stream items |
| sprints-013–017 (hologram-network) | Unaffected | Network layer uses `.holo` archives natively — cleaner with this change |

### What does NOT change

- `AiGraph` IR and all optimization passes
- All format importers (GGUF, GGML, ONNX)
- Quantization and quant-aware lowering
- Tokenizer (embedded in archive via ADR-0012)
- Validation harness (compiles and then calls KvExecutor directly)
- hologram-sandbox integration (also uses `.holo` archives)
- hologram-network (distributes and executes `.holo` archives — fully compatible)

---

## Rationale for removing session management from hologram-ai

The generation loop, once the prefill and decode entrypoints are available, is
approximately:

```python
1. allocate kv_cache_buffer
2. call lm.prefill(token_ids, kv_cache_buffer) → logits, kv_cache_buffer
3. sample_token(logits)
4. while not eos:
      call lm.decode(token, present_len, kv_cache_buffer) → logits, kv_cache_buffer
      present_len += 1
      sample_token(logits)
```

This is ~20 lines of business logic. There is no framework value in wrapping it.
Any application-specific concerns (stop sequences, repetition penalty, conversation
history, prompt templating) are better handled by the caller without an opinionated
session type in between.

The session type was a convenience that became an architectural burden: it forced
hologram-ai to know about sampling, temperature, top-p, stop tokens, and conversation
templates — concerns that have nothing to do with compiling ONNX/GGUF/GGML models.

---

## Alternatives considered

**A. Keep session in hologram-ai as an optional feature.**
Rejected. Optional features create maintenance surface. The session type would
still need to evolve with every new model family and generation strategy.

**B. Move session to a new `hologram-ai-runtime` crate.**
Rejected. The generation loop is so thin that a dedicated crate adds dependency
overhead without providing meaningful abstraction. Applications write the loop
themselves using standard hologram API.

**C. Move session to `hologram` itself.**
Rejected. `hologram` is domain-agnostic. Adding LLM semantics (EOS tokens, KV-cache
management naming conventions, prompt templating) would violate its purpose as a
general execution substrate.
