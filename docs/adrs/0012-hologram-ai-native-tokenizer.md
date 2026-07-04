# ADR-0012: Hologram-Native Tokenizer via ConstantStore and .holo Archives

- Status: Accepted
- Date: 2026-03-07
- Owners: Architecture

---

## Context

`hologram-ai` defines a `Tokenizer` trait (`encode`, `decode`, `eos_token_id`) but
currently ships no bundled implementation. The planned integration path was to wrap
the HuggingFace `tokenizers` crate (a Rust wrapper around C++ tokenizer
implementations), with callers providing `Box<dyn Tokenizer>`.

This approach has several drawbacks:

1. **External C dependency.** The `tokenizers` crate pulls in a non-trivial C++
   dependency chain, complicating cross-compilation and WASM targets.
2. **Separate file management.** `tokenizer.json` must travel alongside `.holo`
   archives. A model archive without its tokenizer is not self-contained.
3. **Missed data lifecycle opportunity.** hologram already provides `ConstantStore`
   for constant data, `HoloWriter::add_section()` for custom archive sections,
   and `HoloLoader` for memory-mapped loading — all suitable for vocab data.
4. **GGUF already ships vocab.** GGUF metadata contains full vocabulary tables,
   merge rules, and scores. Requiring an external `tokenizer.json` when the data
   is already in the model file is redundant.

---

## Decision

### 1. New crate: `hologram-ai-tokenizer`

Tokenizer logic lives in a dedicated crate between `hologram-ai-common` and the
facade:

```
hologram-ai-quant      → (no internal deps)
hologram-ai-common     → hologram-ai-quant, hologram
hologram-ai-tokenizer  → hologram-ai-common, hologram       ← NEW
hologram-ai-onnx       → hologram-ai-common
hologram-ai-gguf       → hologram-ai-common, hologram-ai-quant
hologram-ai-ggml       → hologram-ai-common, hologram-ai-quant
hologram-ai            → all of the above + hologram
```

`hologram-ai-tokenizer` depends on `hologram-ai-common` (for `AiGraph` metadata
types) and `hologram` (for `ConstantStore`, `ConstantId`). It does not depend on
any importer crate.

### 2. Tokenization stays outside the graph

Tokenization is a text preprocessing/postprocessing step. It does not benefit
from LUT dispatch, CSE, parallel level scheduling, or any `hologram::Graph`
execution model feature. String processing through `CustomOpRegistry` would
require non-byte-domain data flowing through a byte-domain graph with no
meaningful fusion opportunities.

Tokenization operates at the session boundary: text → token IDs before the
graph, token IDs → text after the graph. The `Tokenizer` trait remains the
interface between text and the numerical pipeline.

### 3. Vocab and merge data stored in `ConstantStore`, serialized via `.holo` custom section

Tokenizer data is treated as constant data alongside model weights:

- **Vocab table** — stored as `ConstantData::Bytes` entries in `ConstantStore`,
  indexed by `ConstantId` with a well-known prefix (`tokenizer.vocab`).
- **Merge rules** (BPE) — stored as `ConstantData::Bytes` with prefix
  `tokenizer.merges`.
- **Scores** (SentencePiece) — stored as `ConstantData::Bytes` with prefix
  `tokenizer.scores`.
- **Tokenizer metadata** — algorithm type, special token config, normalization
  rules — stored in a custom `.holo` section: `SECTION_TOKENIZER = 0x1001`
  (within the `0x1000–0x1FFF` range reserved for hologram-ai).

This makes `.holo` archives self-contained: model weights + tokenizer in a
single file. No external `tokenizer.json` needed at runtime.

### 4. Import sources

Tokenizer data is extracted at compile time from:

- **GGUF metadata** — `tokenizer.ggml.model` (algorithm type), `tokenizer.ggml.tokens`
  (vocab), `tokenizer.ggml.scores` (unigram scores), `tokenizer.ggml.merges`
  (BPE merges), `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`, etc.
- **`tokenizer.json`** (HuggingFace format) — parsed at compile time, vocab and
  merges packed into `ConstantStore`.
- **`tokenizer.model`** (SentencePiece protobuf) — parsed at compile time.

After import, all sources produce the same internal representation — downstream
code does not know or care which format the tokenizer data came from.

### 5. Algorithm priority: BPE first, SentencePiece second, WordPiece third

- **BPE** (Phase 2) — covers LLaMA, Mistral, GPT-family, Phi, Qwen, Gemma.
  This is the vast majority of decoder-only LLMs.
- **SentencePiece/Unigram** (Phase 2, later) — covers some multilingual models.
- **WordPiece** (Phase 3) — covers BERT-class encoder models, aligns with ONNX
  Phase 2 scope.

### 6. Expanded `Tokenizer` trait with `NativeTokenizer` implementation

The trait gains additional methods:

```rust
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn eos_token_id(&self) -> u32;
    fn bos_token_id(&self) -> Option<u32>;
    fn vocab_size(&self) -> usize;
    fn id_to_token(&self, id: u32) -> Option<&str>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
}
```

`NativeTokenizer` implements this trait using data loaded from `ConstantStore`.
It is constructed automatically when loading a `.holo` archive that contains
`SECTION_TOKENIZER`.

Callers can still provide `Box<dyn Tokenizer>` for custom implementations.

---

## Consequences

**Positive:**

- `.holo` archives are self-contained — model + tokenizer in a single file
- No external C dependency for tokenization
- Tokenizer data benefits from hologram's lazy loading / mmap infrastructure
- GGUF vocab data is used directly — no redundant `tokenizer.json` needed
- Clean WASM and cross-compilation story (pure Rust)
- Consistent data lifecycle: all constant data goes through `ConstantStore`

**Negative:**

- Must implement BPE/SentencePiece/WordPiece correctly — these are non-trivial
  algorithms with Unicode edge cases, pre-tokenization regex patterns, and
  byte-fallback handling
- Must validate correctness against the HuggingFace `tokenizers` crate as a
  reference implementation
- Additional crate in the workspace (7 crates total)

**Neutral:**

- The `Tokenizer` trait remains the abstraction boundary — callers who prefer
  external tokenizer implementations can still use `Box<dyn Tokenizer>`
- GGUF models that lack vocab metadata (rare) fall back to requiring an
  external tokenizer

---

## Alternatives Considered

**Wrap HuggingFace `tokenizers` crate**
Rejected. Introduces a C++ dependency chain, complicates cross-compilation and
WASM targets, and leaves `.holo` archives non-self-contained.

**Tokenizer as graph ops via `CustomOpRegistry`**
Rejected. String processing does not benefit from LUT dispatch, CSE, or any
graph execution model feature. Would force non-byte-domain data through a
byte-domain graph.

**Keep tokenizer entirely external (current design)**
Rejected. Makes `.holo` archives not self-contained. Wastes GGUF vocab data
that is already present in the model file. Requires users to manage
`tokenizer.json` files alongside model archives.
