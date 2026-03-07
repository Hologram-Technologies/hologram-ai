# Prompt: Native Tokenizer Implementation

## Purpose

Implement the `hologram-ai-tokenizer` crate: a native BPE tokenizer that uses
hologram's `ConstantStore` for vocab/merge storage and integrates into `.holo`
archives via a custom section.

Run this prompt after the GGUF importer and `.holo` archive support are working.

---

## Context

At this point:
- `hologram-ai-gguf` imports LLaMA-family GGUF models into `AiGraph`
- `hologram-ai` has working `InferenceSession` with `run()` and `generate()`
- `.holo` archive write/load works via `HoloWriter` / `HoloLoader`
- `stream_tokens()` exists but requires an external `Box<dyn Tokenizer>`

Your task is to:
1. Create the `hologram-ai-tokenizer` crate with BPE tokenizer
2. Extract GGUF vocab into `ConstantStore`
3. Serialize/deserialize tokenizer data in `.holo` archives
4. Integrate with `CompiledModel` so models have embedded tokenizers
5. Update CLI to use embedded tokenizer by default

Architecture references:
- `../hologram-architecture/specs/projects/hologram-ai/tokenizer.md`
- `../hologram-architecture/specs/adrs/0012-hologram-native-tokenizer.md`

---

## Task 1: Create `hologram-ai-tokenizer` crate

Scaffold the crate at `crates/hologram-ai-tokenizer/`.

**Cargo.toml:**
```toml
[package]
name = "hologram-ai-tokenizer"
version = "0.1.0"
edition = "2021"

[dependencies]
hologram-ai-common = { path = "../hologram-ai-common" }
hologram = { path = "../../../hologram" }

[dev-dependencies]
serde_json = "1"
```

**Public API surface (`src/lib.rs`):**
```rust
mod bpe;
mod config;
mod native;
mod section;
mod vocab;

pub use config::{
    TokenizerConfig, TokenizerAlgorithm, SpecialTokens,
    NormalizationConfig, NormStep,
};
pub use native::NativeTokenizer;
pub use section::{TokenizerSectionData, SECTION_TOKENIZER};
pub use vocab::{VocabTable, MergeRules, UnigramModel};

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

Update the workspace `Cargo.toml` to include the new crate.

---

## Task 2: Implement data model types

In `src/vocab.rs`:

```rust
pub struct VocabTable {
    pub id_to_token: Vec<Vec<u8>>,
    pub token_to_id: HashMap<Vec<u8>, u32>,
}

impl VocabTable {
    pub fn new(tokens: Vec<Vec<u8>>) -> Self { ... }
    pub fn len(&self) -> usize { ... }
}
```

```rust
pub struct MergeRules {
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
}
```

In `src/config.rs`, implement `TokenizerConfig`, `TokenizerAlgorithm`,
`SpecialTokens`, `NormalizationConfig` as defined in `tokenizer.md`.

---

## Task 3: Implement BPE encode

In `src/bpe.rs`:

```rust
pub struct BpeEncoder {
    vocab: VocabTable,
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), u32>,
    byte_fallback: bool,
    pre_tokenize_pattern: Option<regex::Regex>,
}

impl BpeEncoder {
    pub fn new(vocab: VocabTable, merges: MergeRules, byte_fallback: bool) -> Self

    pub fn encode(&self, text: &str) -> Vec<u32>
}
```

**Encoding steps:**
1. Pre-tokenize input using regex pattern (LLaMA pattern:
   `r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"`)
2. For each pre-token, convert to initial byte-level token sequence
3. Iteratively merge: find the pair with the lowest merge rank, merge it,
   repeat until no more merges apply
4. Map final token sequences to IDs via `VocabTable::token_to_id`
5. Handle byte-fallback: if a byte isn't in vocab, use byte tokens `<0xNN>`

Use a priority-based approach: scan all adjacent pairs, pick the one with the
lowest rank, merge, rescan. This is O(n^2) in the worst case but correct.
Optimize later if profiling shows it's a bottleneck.

**Tests:**
- Empty string → empty vec
- Single character → correct single token
- Known LLaMA tokenizations (pre-computed golden values)

---

## Task 4: Implement BPE decode

In `src/bpe.rs`:

```rust
impl BpeEncoder {
    pub fn decode(&self, token_ids: &[u32]) -> String
}
```

**Decoding steps:**
1. For each token ID, look up bytes in `VocabTable::id_to_token`
2. Concatenate all byte sequences
3. If byte_fallback is enabled, reassemble `<0xNN>` byte tokens
4. Validate UTF-8, replace invalid sequences with replacement character

**Tests:**
- Round-trip: `decode(encode(text)) == text` for ASCII, Unicode, emoji
- Byte fallback tokens decode correctly
- Special tokens (BOS/EOS) are omitted from decode output

---

## Task 5: Implement `NativeTokenizer`

In `src/native.rs`:

```rust
pub struct NativeTokenizer {
    config: TokenizerConfig,
    encoder: BpeEncoder,  // or UnigramEncoder / WordPieceEncoder in future
}

impl NativeTokenizer {
    pub fn from_constant_store(
        store: &hologram::ConstantStore,
        section: &TokenizerSectionData,
    ) -> Result<Self>

    pub fn from_tokenizer_json(path: &Path) -> Result<Self>

    pub fn from_gguf_vocab(
        tokens: &[String],
        scores: Option<&[f32]>,
        merges: Option<&[String]>,
        token_types: Option<&[u32]>,
        special: SpecialTokens,
        model_type: &str,
    ) -> Result<Self>
}

impl Tokenizer for NativeTokenizer { ... }
```

**`from_constant_store` flow:**
1. Read `tokenizer.vocab` from `ConstantStore` → deserialize `VocabTable`
2. Read `tokenizer.merges` from `ConstantStore` → deserialize `MergeRules`
3. Read `tokenizer.scores` if present → deserialize scores
4. Read `tokenizer.token_types` if present → deserialize types
5. Construct `TokenizerConfig` from `TokenizerSectionData` + loaded data
6. Build `BpeEncoder` (or other algorithm encoder)

**`from_tokenizer_json` flow:**
1. Parse JSON file
2. Extract `model.vocab`, `model.merges`, `added_tokens`, `normalizer`
3. Build `VocabTable`, `MergeRules`, `SpecialTokens`
4. Construct `NativeTokenizer`

---

## Task 6: Implement `.holo` section serialization

In `src/section.rs`:

```rust
pub const SECTION_TOKENIZER: u16 = 0x1001;

pub struct TokenizerSectionData {
    pub algorithm: u8,
    pub byte_fallback: bool,
    pub add_bos: bool,
    pub add_eos: bool,
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub pad_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub normalization: u8,
    pub vocab_size: u32,
    pub additional_special_tokens: Vec<(String, u32)>,
}

impl TokenizerSectionData {
    pub fn serialize(&self) -> Vec<u8>
    pub fn deserialize(data: &[u8]) -> Result<Self>
}
```

Also implement helpers to pack/unpack tokenizer data into `ConstantStore`:

```rust
pub fn pack_tokenizer_to_store(
    config: &TokenizerConfig,
    store: &mut hologram::ConstantStore,
) -> Result<()>

pub fn write_tokenizer_section(
    config: &TokenizerConfig,
    writer: &mut hologram::HoloWriter,
) -> Result<()>
```

**Tests:**
- Round-trip serialization: `deserialize(serialize(data)) == data`
- ConstantStore pack/unpack round-trip

---

## Task 7: GGUF vocab extraction

Update `hologram-ai-gguf` to extract tokenizer data during import.

In the GGUF importer's `GgufMetadata`, add fields for tokenizer data:

```rust
pub struct GgufMetadata {
    // ... existing fields ...
    pub tokenizer_model: Option<String>,
    pub tokens: Option<Vec<String>>,
    pub scores: Option<Vec<f32>>,
    pub token_type: Option<Vec<u32>>,
    pub merges: Option<Vec<String>>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub unknown_token_id: Option<u32>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
}
```

After `import_gguf()` returns an `AiGraph`, store tokenizer data in
`AiGraph::metadata` under the `tokenizer.*` namespace:

```rust
graph.metadata.insert("tokenizer.model".into(), meta.tokenizer_model.clone());
graph.metadata.insert("tokenizer.tokens".into(), /* serialized tokens */);
// etc.
```

---

## Task 8: Integrate with `CompiledModel`

Update the `ModelCompiler` pipeline in `hologram-ai`:

```rust
impl ModelCompiler {
    pub fn compile(&self, source: ModelSource, tok_source: TokenizerSource) -> Result<CompiledModel>
}

pub enum TokenizerSource {
    FromModel,      // extract from model metadata (default for GGUF)
    File(PathBuf),  // parse from tokenizer.json or tokenizer.model
    None,           // no tokenizer
}
```

After compilation:
1. Extract tokenizer data from `AiGraph::metadata` (if `FromModel`)
   or parse from file (if `File`)
2. Pack into `ConstantStore`
3. Write `SECTION_TOKENIZER` to `.holo` archive
4. Construct `NativeTokenizer` and store in `CompiledModel`

Update `CompiledModel`:
```rust
pub struct CompiledModel {
    // ... existing fields ...
    tokenizer: Option<Arc<dyn Tokenizer>>,
}

impl CompiledModel {
    pub fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.tokenizer.as_deref()
    }
}
```

When loading a `.holo` archive:
1. Check for `SECTION_TOKENIZER`
2. If present, deserialize and construct `NativeTokenizer` from `ConstantStore`
3. Store in `CompiledModel`

---

## Task 9: Update CLI

Update `generate` command tokenizer resolution:

```rust
// Priority:
// 1. --tokenizer <PATH> (explicit)
// 2. compiled_model.tokenizer() (embedded in .holo)
// 3. tokenizer.json next to model file (auto-discover)
// 4. Error
```

Update `compile` command:
```
hologram-ai compile model.gguf -o model.holo [--tokenizer tokenizer.json]
```

Update `inspect` command to show tokenizer info:
```
Tokenizer:      BPE (embedded)
Vocab size:     32000
BOS token:      1 (<s>)
EOS token:      2 (</s>)
```

---

## Task 10: Golden tests

Create golden encode/decode test fixtures:

1. Use the HuggingFace `tokenizers` crate (as a **test-only** dev dependency)
   to generate reference tokenizations for a set of test strings
2. Save as `tests/fixtures/tokenizer/llama-encode-golden.json`
3. Write tests that load the golden file and compare `NativeTokenizer` output

```rust
#[test]
fn bpe_encode_matches_hf_llama_tokenizer() {
    let tok = NativeTokenizer::from_tokenizer_json(
        "tests/fixtures/tokenizer/llama-tokenizer.json"
    ).unwrap();
    let golden: Vec<(String, Vec<u32>)> = load_golden("llama-encode-golden.json");
    for (text, expected_ids) in &golden {
        assert_eq!(tok.encode(text), *expected_ids, "mismatch for: {text:?}");
    }
}
```

Test strings should include:
- Simple English text
- Punctuation and special characters
- Unicode (CJK, emoji, combining marks, RTL)
- Whitespace edge cases (leading, trailing, multiple spaces, tabs, newlines)
- Empty string
- Very long string (1000+ characters)

---

## Acceptance Criteria

- [ ] `NativeTokenizer::from_tokenizer_json()` loads a LLaMA tokenizer.json
- [ ] `encode()` / `decode()` round-trips correctly for all golden test strings
- [ ] `NativeTokenizer` output matches HuggingFace `tokenizers` reference for LLaMA
- [ ] GGUF importer extracts vocab and stores in `AiGraph::metadata`
- [ ] `ModelCompiler::compile()` embeds tokenizer in `.holo` archive
- [ ] Loading a `.holo` archive auto-constructs `NativeTokenizer`
- [ ] `hologram-ai generate model.holo "Hello"` works without `--tokenizer` flag
- [ ] `hologram-ai inspect model.holo` shows tokenizer info
- [ ] All existing tests continue to pass
