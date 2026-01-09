# T5 Tokenization Issue Analysis

## Summary

The T5 pipeline runs successfully end-to-end, but generates meaningless output (`<extra_id_0>` repeated). Root cause: the greedy tokenization algorithm produces broken input tokens.

## Evidence

### Input Tokenization (Broken)
```
Text: "Tell me a joke about programming"
Tokens: [8779, 2, 140, 2, 9, 2, 10802, 2, 81, 2, ...]
```

**Pattern**: Every other token is `2` (`<unk>` token), indicating tokenization failure.

### Output Generation (Broken as a result)
```
Generated tokens: [0, 32099, 32099, 32099, ...]
Decoded: "<extra_id_0><extra_id_0><extra_id_0>..."
```

The model generates `<extra_id_0>` (token 32099) repeatedly because the input is garbled.

## Root Cause

The greedy longest-match tokenization in [src/tokenizers/sentencepiece.rs:70-110](../../src/tokenizers/sentencepiece.rs) is too simplistic:

```rust
fn tokenize_greedy(&self, text: &str) -> Vec<u32> {
    // Try longest match first
    for len in (1..=(chars.len() - i).min(10)).rev() {
        let substr: String = chars[i..i + len].iter().collect();

        // Try with underscore prefix (SentencePiece uses ▁ for spaces)
        let prefixed = if i == 0 || chars[i - 1] == ' ' {
            format!("▁{}", substr.trim_start())
        } else {
            substr.clone()
        };

        if let Some(&token_id) = self.vocab.get(&prefixed) {
            tokens.push(token_id);
            // ...
        }
    }
}
```

**Problems**:
1. Doesn't implement SentencePiece's unigram language model
2. Doesn't handle subword boundaries correctly
3. Doesn't use token scores/probabilities
4. Produces alternating unknown tokens

## What Works ✅

Despite broken tokenization, the infrastructure is solid:

1. **Vocab Loading**: 32,100 tokens loaded correctly
2. **Vocab Size Fix**: Dynamic vocab_size from tokenizer (was hardcoded to 32,128)
3. **Tokenizer Compilation**: Compiles to .holo format successfully
4. **Pipeline Execution**: End-to-end execution completes
5. **Decoding**: Detokenization works (decodes `<extra_id_0>` correctly)

## Solutions

### Option A: Use tokenizers crate temporarily ✅ Recommended

Use HuggingFace `tokenizers` crate as a bridge until hologram_ir gains necessary operations:

```rust
// In src/tokenizers/sentencepiece.rs
use tokenizers::Tokenizer;

impl SentencePieceTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let hf_tokenizer = Tokenizer::from_file(path)?;

        // Extract vocab for compilation
        let vocab = extract_vocab(&hf_tokenizer)?;

        Ok(Self {
            hf_tokenizer: Some(hf_tokenizer),  // Use for runtime
            vocab,                              // Use for compilation
            // ...
        })
    }

    fn encode(&self, text: &str, max_length: usize) -> Result<Vec<u32>> {
        if let Some(ref hf) = self.hf_tokenizer {
            // Use proper SentencePiece tokenization
            let encoding = hf.encode(text, false)?;
            let mut tokens = encoding.get_ids().to_vec();
            tokens.truncate(max_length);
            // ...
        } else {
            // Fallback to greedy (for testing)
        }
    }
}
```

**Pros**:
- Production-quality tokenization immediately
- Maintains compilation to .holo (vocab still used for that)
- User can see T5 working end-to-end
- Bridge until hologram_ir implementation ready

**Cons**:
- Adds dependency on `tokenizers` crate
- Not "pure hologram" for runtime execution (yet)

### Option B: Implement proper SentencePiece algorithm

Implement the full unigram language model tokenization:

1. Load token scores from tokenizer.json
2. Build trie for efficient longest-match
3. Implement Viterbi algorithm for optimal segmentation
4. Handle special tokens properly

**Pros**:
- Pure Rust implementation
- Better understanding of algorithm
- Foundation for hologram_ir implementation

**Cons**:
- Significant implementation work
- Complex algorithm (100+ lines)
- Still not "pure hologram" (needs hologram_ir ops)

### Option C: Wait for hologram_ir operations

Block on hologram_ir gaining necessary operations for full IR-based tokenization.

**Pros**:
- Aligns with "pure hologram" vision
- Everything compiles to .holo and runs on hologram backend

**Cons**:
- Can't demonstrate working T5 generation now
- Upstream work required first

## Recommendation

**Use Option A** as a bridge solution:

1. Add `tokenizers` crate dependency (feature-gated if desired)
2. Use HF tokenizer for runtime encode/decode
3. Keep vocab extraction for .holo compilation
4. Document that runtime execution uses `tokenizers` temporarily
5. When hologram_ir gains ops, migrate to full IR implementation

This gives the best user experience (working T5 generation) while maintaining the compilation infrastructure. The user can see their joke generated in English!

## Next Steps

1. Add `tokenizers = "0.15"` to Cargo.toml
2. Update SentencePieceTokenizer to use HF tokenizer
3. Test T5 generation with proper tokenization
4. Generate English joke output
5. Document this as a temporary bridge solution

## Related Files

- [src/tokenizers/sentencepiece.rs](../../src/tokenizers/sentencepiece.rs) - Greedy tokenization (broken)
- [src/tokenizers/compiler.rs](../../src/tokenizers/compiler.rs) - Vocab parsing (working)
- [src/cli/run.rs](../../src/cli/run.rs) - Pipeline execution (working)
- [configs/t5-generate.toml](../../configs/t5-generate.toml) - T5 config (working)

## Status

- **Tokenizer Compilation**: ✅ Working (compiles to .holo)
- **Vocab Loading**: ✅ Working (32,100 tokens)
- **Runtime Tokenization**: ❌ Broken (greedy algorithm insufficient)
- **Pipeline Execution**: ✅ Working (runs end-to-end)
- **Model Generation**: ⚠️  Works but gets garbage input
- **Detokenization**: ✅ Working

**Blocker**: Runtime tokenization algorithm needs improvement before T5 can generate meaningful text.
