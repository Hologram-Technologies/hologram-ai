# hologram-ai: Tokenizer Architecture

---

## Overview

The tokenizer subsystem converts text to token IDs (encode) and token IDs back
to text (decode). It operates at the session boundary — before and after the
numerical inference pipeline — and is not part of the computation graph.

Tokenizer data (vocab tables, merge rules, special tokens) is stored in
`hologram::ConstantStore` and serialized into `.holo` archives via a custom
section (`SECTION_TOKENIZER = 0x1001`). This makes `.holo` archives
self-contained: model weights + tokenizer in a single file.

See [ADR-0012](../../adrs/0012-hologram-native-tokenizer.md) for the decision
rationale.

---

## Crate: `hologram-ai-tokenizer`

**Position in dependency graph:**

```
hologram-ai-common     → hologram-ai-quant, hologram
hologram-ai-tokenizer  → hologram-ai-common, hologram
hologram-ai            → hologram-ai-tokenizer + all other crates
```

**Responsibilities:**

- `Tokenizer` trait definition (expanded)
- `NativeTokenizer` implementation (BPE, SentencePiece, WordPiece)
- Tokenizer data model types (`VocabTable`, `MergeRules`, `SpecialTokens`)
- Construction from `ConstantStore` + section metadata
- Import helpers for parsing vocab from GGUF metadata, `tokenizer.json`, and
  `tokenizer.model`
- Serialization / deserialization for `SECTION_TOKENIZER`

**Does not depend on** any importer crate. Importers call tokenizer helpers
to extract and store vocab data, but the tokenizer crate does not know about
GGUF/ONNX/GGML formats.

---

## Expanded `Tokenizer` Trait

```rust
pub trait Tokenizer: Send + Sync {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode token IDs back to text.
    fn decode(&self, tokens: &[u32]) -> String;

    /// End-of-sequence token ID.
    fn eos_token_id(&self) -> u32;

    /// Beginning-of-sequence token ID, if the model uses one.
    fn bos_token_id(&self) -> Option<u32>;

    /// Total vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Look up the string representation of a token ID.
    fn id_to_token(&self, id: u32) -> Option<&str>;

    /// Look up the token ID for a string token.
    fn token_to_id(&self, token: &str) -> Option<u32>;
}
```

The trait is object-safe (`Box<dyn Tokenizer>` works). Callers can provide
custom implementations for use cases not covered by `NativeTokenizer`.

---

## Data Model

### `VocabTable`

Maps between token IDs and token byte sequences:

```rust
pub struct VocabTable {
    pub id_to_token: Vec<Vec<u8>>,       // indexed by token ID
    pub token_to_id: HashMap<Vec<u8>, u32>,
}
```

Tokens are stored as raw bytes, not `String`, because BPE byte-fallback tokens
are arbitrary byte sequences that may not be valid UTF-8.

### `MergeRules` (BPE)

Ordered list of merge pairs with priority:

```rust
pub struct MergeRules {
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,  // ordered by priority (index = rank)
}
```

Lower index = higher priority. During encoding, the pair with the lowest rank
is merged first.

### `UnigramModel` (SentencePiece)

Scored vocabulary for Viterbi segmentation:

```rust
pub struct UnigramModel {
    pub pieces: Vec<(Vec<u8>, f32)>,  // (token bytes, log probability)
}
```

### `SpecialTokens`

```rust
pub struct SpecialTokens {
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub pad_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub additional: HashMap<String, u32>,  // e.g. "<|im_start|>", "<|im_end|>"
}
```

### `NormalizationConfig`

Pre-tokenization text normalization:

```rust
pub enum NormalizationConfig {
    None,
    Nfc,
    Nfkc,
    /// Prepend a space to the input (SentencePiece convention).
    PrependSpace,
    /// Custom sequence of normalization steps.
    Sequence(Vec<NormStep>),
}

pub enum NormStep {
    Nfc,
    Nfkc,
    Lowercase,
    StripAccents,
    PrependSpace,
    Replace { pattern: String, replacement: String },
}
```

### `TokenizerConfig`

Top-level configuration tying everything together:

```rust
pub struct TokenizerConfig {
    pub algorithm: TokenizerAlgorithm,
    pub special_tokens: SpecialTokens,
    pub normalization: NormalizationConfig,
    pub byte_fallback: bool,
    pub add_bos: bool,
    pub add_eos: bool,
}

pub enum TokenizerAlgorithm {
    Bpe {
        vocab: VocabTable,
        merges: MergeRules,
    },
    Unigram {
        vocab: VocabTable,
        model: UnigramModel,
    },
    WordPiece {
        vocab: VocabTable,
        continuing_subword_prefix: String,  // typically "##"
        max_input_chars_per_word: usize,
    },
}
```

---

## Storage Layout

### ConstantStore entries

Tokenizer data is packed into `ConstantStore` using well-known `ConstantId`
prefixes:

| ConstantId | Content | Format |
|------------|---------|--------|
| `tokenizer.vocab` | Vocab table (id→token mapping) | Length-prefixed byte sequences, concatenated |
| `tokenizer.merges` | BPE merge pairs | Newline-delimited `"token1 token2"` pairs |
| `tokenizer.scores` | Unigram scores | `f32` array, little-endian, indexed by token ID |
| `tokenizer.token_types` | Token type flags | `u8` array (0=normal, 1=unknown, 2=control, 3=user_defined, 4=unused, 5=byte) |

### `.holo` section: `SECTION_TOKENIZER = 0x1001`

The custom section stores tokenizer metadata that doesn't fit naturally into
`ConstantStore`:

```rust
pub struct TokenizerSectionData {
    pub algorithm: u8,        // 0=BPE, 1=Unigram, 2=WordPiece
    pub byte_fallback: bool,
    pub add_bos: bool,
    pub add_eos: bool,
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub pad_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub normalization: u8,    // 0=None, 1=NFC, 2=NFKC, 3=PrependSpace
    pub vocab_size: u32,
    pub additional_special_tokens: Vec<(String, u32)>,
}
```

Serialized as a simple binary format with a version byte prefix. The section
kind `0x1001` is within the `0x1000–0x1FFF` range reserved for hologram-ai
in the `.holo` archive format.

---

## Import Sources

### GGUF Metadata

GGUF files store tokenizer data in metadata keys:

| GGUF key | Maps to |
|----------|---------|
| `tokenizer.ggml.model` | `TokenizerAlgorithm` variant selection (`"llama"` → BPE, `"gpt2"` → BPE, etc.) |
| `tokenizer.ggml.tokens` | `VocabTable::id_to_token` |
| `tokenizer.ggml.scores` | `UnigramModel::pieces` scores |
| `tokenizer.ggml.token_type` | `tokenizer.token_types` in ConstantStore |
| `tokenizer.ggml.merges` | `MergeRules::merges` |
| `tokenizer.ggml.bos_token_id` | `SpecialTokens::bos_id` |
| `tokenizer.ggml.eos_token_id` | `SpecialTokens::eos_id` |
| `tokenizer.ggml.padding_token_id` | `SpecialTokens::pad_id` |
| `tokenizer.ggml.unknown_token_id` | `SpecialTokens::unk_id` |
| `tokenizer.ggml.add_bos_token` | `TokenizerConfig::add_bos` |
| `tokenizer.ggml.add_eos_token` | `TokenizerConfig::add_eos` |

The GGUF importer calls `hologram_ai_tokenizer::import_gguf_vocab()` to
extract this data and store it in the `AiGraph::metadata` map under the
`tokenizer.*` namespace.

### `tokenizer.json` (HuggingFace format)

Parsed at compile time. The JSON structure maps to the internal types:

- `model.type` → `TokenizerAlgorithm` variant
- `model.vocab` → `VocabTable`
- `model.merges` → `MergeRules`
- `added_tokens` → `SpecialTokens::additional`
- `normalizer` → `NormalizationConfig`
- `pre_tokenizer` → pre-tokenization regex pattern

### `tokenizer.model` (SentencePiece protobuf)

SentencePiece model files are parsed at compile time. The protobuf
`ModelProto` is decoded to extract vocabulary pieces and scores.

---

## Algorithm Implementations

### BPE (Byte-Pair Encoding)

**Encode:**

1. **Normalize** input text per `NormalizationConfig`
2. **Pre-tokenize** — split input by regex pattern (model-specific; LLaMA uses
   `r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"`)
3. **Byte-encode** each pre-token to a sequence of byte tokens (if byte_fallback)
   or character tokens
4. **Merge** — iteratively find the highest-priority merge pair in each pre-token
   and merge it. Use a priority queue keyed by merge rank for O(n log n)
   performance.
5. **Map** merged sequences to token IDs via `VocabTable::token_to_id`
6. **Prepend** BOS token if `add_bos`

**Decode:**

1. Look up each token ID in `VocabTable::id_to_token`
2. Concatenate byte sequences
3. If byte_fallback, reassemble byte tokens into UTF-8
4. Handle special tokens (skip BOS/EOS in output)

### SentencePiece (Unigram)

**Encode:**

1. **Normalize** input text
2. **Viterbi segmentation** — find the segmentation that maximizes the sum of
   log probabilities from `UnigramModel::pieces`
3. Map segments to token IDs

**Decode:**

1. Look up each token ID, concatenate
2. Strip leading space if `PrependSpace` normalization was applied

### WordPiece

**Encode:**

1. **Normalize** and tokenize by whitespace
2. For each word, greedily find the longest prefix in the vocab
3. Continue with `continuing_subword_prefix` (e.g. `"##"`) for remaining subwords
4. Unknown tokens map to `unk_id`

---

## `NativeTokenizer` Struct

```rust
pub struct NativeTokenizer {
    config: TokenizerConfig,
    // Pre-computed lookup structures
    vocab_trie: VocabTrie,           // for fast token lookup
    merge_ranks: HashMap<(u32, u32), u32>,  // (left_id, right_id) → rank (BPE)
}

impl NativeTokenizer {
    /// Construct from ConstantStore + section metadata.
    /// Called automatically when loading a .holo archive with SECTION_TOKENIZER.
    pub fn from_constant_store(
        store: &hologram::ConstantStore,
        section: &TokenizerSectionData,
    ) -> Result<Self>

    /// Construct from a tokenizer.json file.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self>

    /// Construct from GGUF metadata.
    pub fn from_gguf_metadata(meta: &GgufVocabData) -> Result<Self>
}

impl Tokenizer for NativeTokenizer { ... }
```

---

## Integration with ModelCompiler

During compilation, tokenizer data flows alongside model data:

```
Model artifact
  │
  ├── import_gguf() ─── extracts vocab from GGUF metadata ──┐
  │                                                          │
  ├── tokenizer.json ─── parsed at compile time ─────────────┤
  │                                                          │
  └── tokenizer.model ── parsed at compile time ─────────────┤
                                                             │
                                          ┌──────────────────▼──────────────┐
                                          │  TokenizerConfig               │
                                          │  + VocabTable + MergeRules     │
                                          └──────────────────┬─────────────┘
                                                             │
                                                pack into ConstantStore
                                                write SECTION_TOKENIZER
                                                             │
                                                             ▼
                                                    .holo archive
                                          (model weights + tokenizer data)
```

`ModelCompiler::compile()` gains an optional `tokenizer_source` parameter:

```rust
pub enum TokenizerSource {
    /// Extract from model metadata (GGUF). Default when available.
    FromModel,
    /// Parse from external file.
    File(PathBuf),
    /// No tokenizer (single-pass / raw tensor workflows).
    None,
}
```

---

## Integration with CompiledModel

```rust
pub struct CompiledModel {
    // ... existing fields ...
    tokenizer: Option<Arc<dyn Tokenizer>>,
}

impl CompiledModel {
    /// Returns the embedded tokenizer, if available.
    pub fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.tokenizer.as_deref()
    }
}
```

When loading a `.holo` archive:
1. Check for `SECTION_TOKENIZER` (kind `0x1001`)
2. If present, deserialize `TokenizerSectionData`
3. Load vocab/merges/scores from `ConstantStore`
4. Construct `NativeTokenizer`
5. Store as `Arc<dyn Tokenizer>` in `CompiledModel`

---

## Integration with Streaming

`stream_tokens()` tokenizer parameter becomes optional:

```rust
pub fn stream_tokens(
    session: InferenceSession,
    tokenizer: Option<Box<dyn Tokenizer>>,
    prompt: &str,
    opts: GenerateOptions,
) -> TokenStream
```

Resolution order:
1. Explicit `tokenizer` parameter (if `Some`)
2. `session.model().tokenizer()` (embedded in `.holo`)
3. Error: "no tokenizer available"

---

## CLI Changes

### `generate` command — tokenizer resolution

```
Priority:
1. --tokenizer <PATH>   (explicit external tokenizer)
2. Embedded tokenizer    (from .holo SECTION_TOKENIZER)
3. tokenizer.json        (auto-discovered next to model file)
4. Error
```

### `compile` command — tokenizer embedding

```
hologram-ai compile model.gguf -o model.holo [--tokenizer tokenizer.json]
```

- If the model contains vocab data (GGUF): embedded automatically
- If `--tokenizer` flag provided: parsed and embedded
- If neither: `.holo` has no tokenizer section (raw tensor workflows)

### `inspect` command — tokenizer info

```
$ hologram-ai inspect model.holo
...
Tokenizer:      BPE (embedded)
Vocab size:     32000
BOS token:      1 (<s>)
EOS token:      2 (</s>)
```

---

## Testing Strategy

### Golden encode/decode tests

Compare `NativeTokenizer` output against the HuggingFace `tokenizers` crate
(as a test-only dev dependency, not a runtime dependency):

```rust
#[test]
fn bpe_encode_matches_hf_reference() {
    let native = NativeTokenizer::from_tokenizer_json("fixtures/llama-tokenizer.json").unwrap();
    let expected_ids = load_golden("fixtures/llama-encode-golden.json");
    for (text, expected) in &expected_ids {
        assert_eq!(native.encode(text), *expected);
    }
}
```

### Round-trip tests

```rust
#[test]
fn encode_decode_roundtrip() {
    let tok = load_test_tokenizer();
    let texts = ["Hello, world!", "こんにちは", "🎉 emoji test", ""];
    for text in texts {
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }
}
```

### ConstantStore serialization tests

Verify that tokenizer data survives a round-trip through `ConstantStore` +
`HoloWriter` → `HoloLoader` + `ConstantStore`:

```rust
#[test]
fn tokenizer_survives_holo_roundtrip() {
    let original = NativeTokenizer::from_tokenizer_json("fixtures/llama-tokenizer.json").unwrap();
    let holo_bytes = write_to_holo(&original);
    let loaded = load_from_holo(&holo_bytes);
    assert_eq!(original.encode("test"), loaded.encode("test"));
}
```

### Edge cases

- Empty string encoding
- Single character encoding
- Byte-fallback tokens (non-UTF-8 sequences)
- Unicode normalization edge cases (combining characters, zero-width joiners)
- Maximum vocab ID boundary
- Unknown token handling
