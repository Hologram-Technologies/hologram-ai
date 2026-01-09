# Tokenizer Integration - Implementation Plan

## Completed ✅

1. **Tokenizer Module Created** (`src/tokenizers/`)
   - SentencePiece tokenizer implementation
   - BPE tokenizer placeholder
   - Tokenizer compiler placeholder
   - Trait-based architecture for pluggable tokenizers

2. **Config Integration**
   - Added `TokenizerConfig` struct with serde support
   - Added `tokenizer` field to `UnifiedConfig`
   - Supports config-driven tokenizer loading

3. **PipelineContext Updated**
   - Added `tokenizer: Option<Box<dyn Tokenizer>>` field
   - Added `executor_cache: HashMap<PathBuf, ModelExecutor>` field
   - Tokenizer loaded at pipeline initialization

4. **Tokenize Builtin Updated**
   - Uses real tokenizer if available
   - Falls back to character-based tokenization
   - Supports both T5 (with attention_mask) and CLIP modes

## Remaining Tasks

### 1. Add Detokenization Builtin

Add `decode` or `detokenize` builtin in `/workspace/src/cli/run.rs` around line 1100:

```rust
"decode" | "detokenize" => {
    // Detokenize token IDs back to text
    let tokens_ref = get_arg_str("tokens")
        .ok_or_else(|| anyhow::anyhow!("decode requires 'tokens' argument"))?;

    if let Some(ref tokenizer) = ctx.tokenizer {
        let tokens_f32 = ctx.tensor_cache.get(tokens_ref)
            .ok_or_else(|| anyhow::anyhow!("Tokens '{}' not found", tokens_ref))?;

        // Convert f32 back to u32
        let tokens_u32: Vec<u32> = tokens_f32.iter().map(|&t| t as u32).collect();

        // Decode to text
        let text = tokenizer.decode(&tokens_u32)?;

        info!("  Decoded: \"{}\"", text);

        // Store as generated output for text output handler
        ctx.generated_outputs.insert(default_output.to_string(), text.clone());

        // Also store as tensor (ASCII bytes) for compatibility
        let bytes: Vec<f32> = text.bytes().map(|b| b as f32).collect();
        outputs.insert(default_output.to_string(), bytes);
    } else {
        return Err(anyhow::anyhow!("No tokenizer loaded for decoding"));
    }
}
```

### 2. Implement Executor Caching

Update `execute_model_stage` in `/workspace/src/cli/run.rs` around line 1220:

```rust
fn execute_model_stage(
    holo_path: &Path,
    input_mapping: &HashMap<String, crate::config::Expr>,
    output_names: &[String],
    ctx: &mut PipelineContext,
) -> Result<HashMap<String, Vec<f32>>> {
    use crate::runtime::{ModelExecutor, Tensor, infer_tensor_shape};

    // Check cache first
    let mut executor = if let Some(cached) = ctx.executor_cache.remove(holo_path) {
        debug!("  Using cached executor for: {}", holo_path.display());
        cached
    } else {
        info!("  Loading model: {}", holo_path.display());
        ModelExecutor::from_holo_file(holo_path)
            .with_context(|| format!("Failed to load model from {}", holo_path.display()))?
    };

    // ... existing input preparation and execution code ...

    // Store executor back in cache after use
    ctx.executor_cache.insert(holo_path.to_path_buf(), executor);

    Ok(outputs)
}
```

### 3. Update Config File

Add tokenizer section to `/workspace/configs/t5-generate.toml`:

```toml
# Tokenizer configuration
[tokenizer]
type = "sentencepiece"
vocab_path = "../models/t5-small/tokenizer.json"
max_length = 512
pad_token_id = 0
eos_token_id = 1
unk_token_id = 2
```

### 4. Add Detokenization Stage

Update pipeline in config to decode generated tokens:

```toml
# After generation
[[stages]]
type = "builtin"
builtin = "decode"
outputs = ["generated_text"]

[stages.args]
tokens = "generated_ids"

# Output
[outputs.result]
tensor = "generated_text"
handler = "text"
description = "Generated text"
```

## Performance Improvements

### Expected Speedup from Executor Caching

**Current (without cache)**:
- 50 decoder calls ×  73ms loading = **3.65 seconds** wasted on loading
- Total time: ~4 seconds

**After caching**:
- 1× decoder load (73ms) + 50× execution (~8ms each) = **473ms total**
- **Speedup: 8.5x faster** (4s → 0.47s)

### Memory Usage

- Tokenizer: ~100MB (vocabulary + model)
- Cached executor: ~222MB (decoder .holo file)
- Total additional memory: ~322MB

## Future Enhancements

1. **Compile Tokenizers to .holo**
   - Implement `tokenizers/compiler.rs`
   - Create IR operations for vocabulary lookups
   - Compile to optimized SIMD kernels
   - Expected speedup: 10-100x for tokenization

2. **Support More Tokenizer Types**
   - BPE (GPT models)
   - WordPiece (BERT models)
   - Unigram (XLM models)

3. **Batch Tokenization**
   - Process multiple texts in parallel
   - Use hologram's SIMD operations

4. **Cache Key-Value States**
   - Store decoder KV caches between generation steps
   - Avoid recomputing past tokens
   - Expected speedup: 2-3x for generation

## Testing

After implementation:

```bash
# 1. Build with tokenizer
cargo build --release

# 2. Test T5 generation with real tokenizer
RUST_LOG=info cargo run --release -- run --config configs/t5-generate.toml

# 3. Verify output
cat result.txt  # Should contain readable English text

# 4. Check performance
# Should complete in <1 second instead of 4 seconds
```

## Success Criteria

- ✅ Tokenizer module compiles
- ✅ Config supports tokenizer section
- ✅ Tokenize builtin uses real tokenizer
- ⏳ Decode builtin converts tokens to text
- ⏳ Executor caching reduces load time
- ⏳ Generation completes in <1 second
- ⏳ Output is readable English text
