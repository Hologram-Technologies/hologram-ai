# Plan 26: Trait-Based Embeddable Sections for Single-File Model Distribution

**Status:** Implemented
**Created:** 2026-01-11
**Implemented:** 2026-01-11

## Goal

Enable single-file model distribution by embedding auxiliary data (vocabulary, configs, preprocessor settings) in the `.holo` bundle using a **trait-based extensible system**.

## Motivation

Currently, deploying an AI model requires distributing multiple files:
- The compiled model (`.holo`)
- Vocabulary files (`vocab.txt`, `vocab.json`)
- Tokenizer configuration (`tokenizer_config.json`)
- Model configuration (`config.json`)
- Special tokens map
- SentencePiece models (`.model`)
- Preprocessor configs (for vision models)

This plan enables embedding all these auxiliary files directly into the `.holo` bundle for true single-file distribution.

---

## Design: EmbeddableSection Trait

### Core Trait Definition

```rust
/// Trait for data that can be embedded in a .holo bundle.
/// Implement this trait to add new embeddable section types.
pub trait EmbeddableSection: Send + Sync {
    /// Unique section identifier (e.g., "vocabulary", "tokenizer_config")
    fn section_id(&self) -> &'static str;

    /// Serialize section data to bytes
    fn to_bytes(&self) -> Vec<u8>;

    /// Content type for this section (for tooling/debugging)
    fn content_type(&self) -> &'static str {
        "application/octet-stream"
    }

    /// Optional version for this section format
    fn version(&self) -> u32 {
        1
    }
}

/// Trait for deserializing embedded sections
pub trait FromEmbeddedSection: Sized {
    /// Section identifier this type handles
    const SECTION_ID: &'static str;

    /// Deserialize from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self, EmbedError>;
}
```

### Built-in Section Types

| Type | Section ID | Content Type | Use Case |
|------|------------|--------------|----------|
| `VocabularySection` | `vocabulary` | text/plain | WordPiece vocab (BERT) |
| `TokenizerConfigSection` | `tokenizer_config` | application/json | Tokenizer parameters |
| `ModelConfigSection` | `model_config` | application/json | Model architecture |
| `SpecialTokensSection` | `special_tokens_map` | application/json | Special token mappings |
| `PreprocessorConfigSection` | `preprocessor_config` | application/json | Vision preprocessing |
| `SentencePieceSection` | `sentencepiece_model` | application/x-sentencepiece | SentencePiece binary |
| `RawFileSection` | (custom) | (custom) | Unknown file types |

---

## Bundle Format with Sections Table

### New Header Structure (V2)

```
+================================+
| Magic: "HOLB" (4 bytes)        |
| Version: u32 (4 bytes)         |  Version 2 for sections support
| Flags: u32 (4 bytes)           |
+================================+
| Graph offset: u64              |
| Graph size: u64                |
| Weights offset: u64            |
| Weights size: u64              |
+--------------------------------+
| Sections table offset: u64     |  NEW
| Sections count: u32            |  NEW
| Reserved: [u8; 20]             |
+================================+  Total: 80 bytes
```

### Sections Table Format

```
+================================+
| Section 0:                     |
|   ID length: u16               |
|   ID: [u8; ID_length]          |  e.g., "vocabulary"
|   Content-Type length: u16     |
|   Content-Type: [u8; len]      |  e.g., "text/plain"
|   Version: u32                 |
|   Offset: u64                  |
|   Size: u64                    |
|   Checksum: u32 (CRC32)        |
+--------------------------------+
| Section 1: ...                 |
| Section N: ...                 |
+================================+
```

### Full Bundle Layout

```
+================================+
| Header (80 bytes)              |
+================================+
| Graph (HOLP data)              |
+---- Padding to 4KB -----+
| Sections Table                 |  NEW: Index of all sections
+---- Padding to 4KB -----+
| Section 0 data                 |  e.g., vocabulary
+---- Padding to 4KB -----+
| Section 1 data                 |  e.g., tokenizer_config.json
+---- Padding to 4KB -----+
| Section N data                 |
+---- Padding to 4KB -----+
| Weights (page-aligned)         |
+================================+
```

---

## Implementation Plan

### Phase 1: Define Traits and Section Types

**New file:** `src/core/sections.rs`

```rust
pub mod sections {
    mod traits;        // EmbeddableSection, FromEmbeddedSection
    mod vocabulary;    // VocabularySection
    mod config;        // ModelConfigSection, TokenizerConfigSection
    mod preprocessor;  // PreprocessorConfigSection
    mod sentencepiece; // SentencePieceSection
    mod raw;           // RawFileSection

    pub use traits::*;
    pub use vocabulary::*;
    pub use config::*;
    // ...
}
```

### Phase 2: Update Bundle Header

**File:** `src/core/serialization.rs`

- Add `HOLB_VERSION_V2 = 2` constant
- Create `HoloBundleHeaderV2` struct with sections table fields
- Maintain backward compatibility with V1 bundles

### Phase 3: Update Bundle Writer

**File:** `src/core/bundle.rs`

- Add `sections: Vec<Box<dyn EmbeddableSection>>` to `UnifiedBundleWriter`
- Implement `add_section<S>()` method
- Update `finish()` to serialize sections table and data

### Phase 4: Update Bundle Reader

**File:** `src/core/bundle.rs`

- Add `SectionEntry` struct for section metadata
- Add `sections()`, `has_section()`, `get_section_bytes()` methods
- Add generic `get_section<T: FromEmbeddedSection>()` method
- Add convenience methods like `vocabulary()`, `tokenizer_config()`

### Phase 5: Update OnnxConfig

**File:** `src/core/config.rs`

```rust
pub struct EmbeddedFileConfig {
    pub path: PathBuf,
    pub section_type: SectionType,
    pub custom_id: Option<String>,
}

pub enum SectionType {
    Vocabulary,
    VocabularyJson,
    TokenizerConfig,
    ModelConfig,
    SpecialTokensMap,
    PreprocessorConfig,
    SentencePiece,
    GenerationConfig,
    Raw { content_type: String },
}
```

### Phase 6: Update compile_to_bundle()

**File:** `src/lib.rs`

- Load sections from configured `embedded_files`
- Add sections to `UnifiedBundleWriter` before finishing

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `src/core/sections.rs` | **NEW**: Trait definitions and built-in sections |
| `src/core/serialization.rs` | Update header to V2 with sections table |
| `src/core/bundle.rs` | Add section support to writer/reader |
| `src/core/config.rs` | Add `embedded_files` configuration |
| `src/lib.rs` | Update `compile_to_bundle()` to embed sections |
| `src/core/mod.rs` | Export sections module |

---

## Usage Examples

### Compiling with Sections

```rust
let config = OnnxConfig {
    embedded_files: vec![
        EmbeddedFileConfig {
            path: "models/bert/vocab.txt".into(),
            section_type: SectionType::Vocabulary,
            custom_id: None,
        },
        EmbeddedFileConfig {
            path: "models/bert/tokenizer_config.json".into(),
            section_type: SectionType::TokenizerConfig,
            custom_id: None,
        },
    ],
    ..Default::default()
};

let compiler = OnnxCompiler::with_config(config);
let bundle = compiler.compile_to_bundle(&onnx_bytes)?;
fs::write("model.holo", bundle)?;
```

### Reading Sections

```rust
let reader = UnifiedBundleReader::from_file("model.holo")?;

// List all sections
for section in reader.sections() {
    println!("{}: {} bytes", section.id, section.size);
}

// Get typed sections
if let Some(vocab) = reader.vocabulary() {
    println!("Vocabulary size: {}", vocab.tokens.len());
}

// Get raw bytes for custom sections
if let Some(bytes) = reader.get_section_bytes("custom_data") {
    println!("Custom data: {} bytes", bytes.len());
}
```

### Adding Custom Section Types

```rust
pub struct MyCustomSection {
    pub data: Vec<u8>,
}

impl EmbeddableSection for MyCustomSection {
    fn section_id(&self) -> &'static str { "my_custom_section" }
    fn content_type(&self) -> &'static str { "application/x-custom" }
    fn to_bytes(&self) -> Vec<u8> { self.data.clone() }
}

impl FromEmbeddedSection for MyCustomSection {
    const SECTION_ID: &'static str = "my_custom_section";

    fn from_bytes(bytes: &[u8]) -> Result<Self, EmbedError> {
        Ok(MyCustomSection { data: bytes.to_vec() })
    }
}
```

---

## Common Sections for AI Models

| Section ID | Content Type | Use Case |
|------------|--------------|----------|
| `vocabulary` | text/plain | WordPiece vocab (BERT) |
| `vocabulary_json` | application/json | BPE vocab (GPT-2) |
| `merges` | text/plain | BPE merges.txt |
| `tokenizer_config` | application/json | Tokenizer parameters |
| `model_config` | application/json | Model architecture |
| `special_tokens_map` | application/json | Special token mappings |
| `sentencepiece_model` | application/x-sentencepiece | SentencePiece binary |
| `preprocessor_config` | application/json | Vision preprocessing |
| `feature_extractor` | application/json | Audio preprocessing |
| `generation_config` | application/json | LLM generation params |

---

## Verification

1. **Unit tests:**
   ```bash
   cargo test -p hologram-ai-onnx
   ```

2. **Section roundtrip test:**
   - Create section, serialize to bytes, deserialize back
   - Verify data matches

3. **Bundle with sections test:**
   - Create bundle with multiple sections
   - Read bundle and verify all sections present
   - Verify section data integrity

4. **Backward compatibility:**
   - V1 bundles (no sections) should still load
   - `reader.sections()` returns empty for V1 bundles

---

## Benefits of Trait-Based Design

1. **Extensibility**: New section types just implement the trait
2. **Type safety**: Compile-time checking for section types
3. **Separation of concerns**: Each section type handles its own serialization
4. **Discoverability**: `sections()` lists all embedded data
5. **Forward compatibility**: Unknown sections can be read as raw bytes
6. **Testability**: Each section type can be tested independently
