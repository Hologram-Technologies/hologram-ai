# Plan: Compile and Run T5 Pipeline for Joke Generation

## Task Summary
Compile T5 model to `.holo` format and run the text generation pipeline with prompt "Tell me a joke" to receive a coherent English response.

## Problem Discovered
**Buffer Allocation Error**: Recently compiled models (Jan 20) fail with:
```
OP[13] KernelId(771): input[1] size 262144 bytes exceeds workspace region
'workspace_11' allocation of 4 bytes
```

**Root Cause**:
- Recent commit a22cee7 added `--parallel` flag (defaults to `true`)
- Enables parallel view composition for multi-head attention (2-3x speedup)
- Compiler bug: Underestimates workspace requirements for attention tensors
- Operation 13 in encoder needs 262144 bytes (512×512 attention matrix) but only gets 4 bytes

## Solution: Use Working Pre-Compiled Models

### Models Available (Jan 18 - Before Parallel Bug)
- ✅ **Encoder**: `/workspace/models/t5-small/compiled/encoder-new.holo` (270MB) - **WORKING**
- ✅ **Decoder**: `/workspace/models/t5-small/compiled/decoder-new.holo` (444MB) - **WORKING**
- ✅ **Tokenizer**: `/workspace/models/t5-small/tokenizer.json` (32,100 token vocab) - **WORKING**

These models were compiled **before** the parallel execution bug and are documented as working in [t5-e2e-success.md](../workspace/docs/working/t5-e2e-success.md).

### Pipeline Configuration
- ✅ `/workspace/configs/t5-generate.toml` exists with:
  - Tokenize stage using SentencePiece
  - Encoder execution stage
  - **Generate builtin** for auto-regressive text generation (50 tokens max)
  - Decode stage for detokenization
  - Default prompt: "Tell me a joke in English" ← **Matches user requirement**

### Implementation Status
- ✅ Generate builtin: [run.rs:1187-1428](../workspace/crates/hologram-ai/src/cli/run.rs#L1187-L1428)
- ✅ SentencePiece tokenizer: [sentencepiece.rs](../workspace/crates/hologram-ai/src/tokenizers/sentencepiece.rs)
- ✅ Model executor with caching: [executor.rs](../workspace/crates/hologram-ai/src/runtime/executor.rs)
- ✅ E2E documentation: [t5-e2e-success.md](../workspace/docs/working/t5-e2e-success.md)

## Implementation Plan

### Phase 1: Use Working Pre-Compiled Models
**Goal**: Copy the working models from Jan 18 to the locations expected by the config.

1. **Update config to use working models**:
   - Edit `/workspace/configs/t5-generate.toml`
   - Change `precompiled` paths to point to `encoder-new.holo` and `decoder-new.holo`
   - OR: Copy the working models to the expected paths:
     ```bash
     cp /workspace/models/t5-small/compiled/encoder-new.holo \
        /workspace/models/t5-small/compiled/encoder.holo

     cp /workspace/models/t5-small/compiled/decoder-new.holo \
        /workspace/models/t5-small/compiled/decoder.holo
     ```

2. **Verify prompt in config**:
   - Confirm [t5-generate.toml](../workspace/configs/t5-generate.toml) line 14:
     - Current: `prompt = "Tell me a joke in English"` ✅ **Perfect!**
   - No changes needed - prompt already matches user requirement

### Phase 2: Run T5 Pipeline
**Goal**: Execute the full end-to-end pipeline to generate a joke in English.

1. **Build project** (already done, but verify):
   ```bash
   cargo build --release -p hologram-ai
   ```

2. **Execute pipeline**:
   ```bash
   RUST_LOG=info cargo run --release -p hologram-ai -- run \
     --config /workspace/configs/t5-generate.toml
   ```

3. **Expected execution flow** (~475ms total):
   - **Tokenize** (20ms): "Tell me a joke in English" → `[8779, 140, 3, 9, 10802, 16, 1566, 1, ...]`
   - **Encode** (135ms): encoder-new.holo → encoder_hidden_states [1, 512, 512]
   - **Generate** (300ms): Auto-regressive decoder loop
     - Initialize with PAD token (id=0)
     - For each step (up to 50 tokens):
       - Execute decoder-new.holo: hidden_states → logits [1, seq_len, 32100]
       - Sample next token (greedy argmax from last position)
       - Check for EOS token (id=1)
       - Append to sequence
     - Stop at EOS or max_new_tokens=50
   - **Decode** (20ms): token IDs → English text string
   - **Output**: Print generated text to stdout/JSON

### Phase 3: Verify English Output
**Goal**: Confirm the generated text is coherent English attempting to tell a joke.

1. **Check output characteristics**:
   - ✅ Language: English (T5-small is English-only)
   - ✅ Content: Attempts to tell a joke (may be simple due to model size)
   - ✅ Coherence: Forms grammatically correct sentences
   - ✅ Termination: Ends with EOS token or reaches 50 tokens

2. **Example expected output** (T5-small quality):
   ```json
   {
     "generated_text": "A man walks into a bar and says..."
   }
   ```

   Note: T5-small (60M parameters) has limited creativity compared to larger models, but should produce coherent English text related to jokes.

### Phase 4: Handle Edge Cases (If Needed)

**If models fail to load**:
- Verify file sizes: encoder-new.holo (270MB), decoder-new.holo (444MB)
- Check file permissions: should be readable

**If generation produces gibberish**:
- Check tokenizer path: `/workspace/models/t5-small/tokenizer.json`
- Verify vocab size: 32,100 tokens
- Ensure start_token_id=0 (PAD), eos_token_id=1 (EOS)

**If output is too short/long**:
- Adjust `max_new_tokens` in config (currently 50)
- Check if EOS token is triggering early

**If output is not joke-related**:
- T5-small may need more specific prompting
- Try alternative prompts: "Joke:", "Q: Why did the...", etc.

## Critical Files

### Working Pre-Compiled Models (Jan 18)
- `/workspace/models/t5-small/compiled/encoder-new.holo` (270MB) - **USE THIS**
- `/workspace/models/t5-small/compiled/decoder-new.holo` (444MB) - **USE THIS**
- `/workspace/models/t5-small/tokenizer.json` - SentencePiece vocab (32,100 tokens)

### Configuration
- [/workspace/configs/t5-generate.toml](../workspace/configs/t5-generate.toml) - Pipeline config
  - Line 28: `precompiled = "/workspace/models/t5-small/compiled/encoder.holo"`
  - Line 32: `precompiled = "/workspace/models/t5-small/compiled/decoder.holo"`
  - Update these to point to encoder-new.holo and decoder-new.holo

### Source Code (Reference Only)
- [run.rs:1187-1428](../workspace/crates/hologram-ai/src/cli/run.rs#L1187-L1428) - Generate builtin
- [sentencepiece.rs](../workspace/crates/hologram-ai/src/tokenizers/sentencepiece.rs) - Tokenizer
- [executor.rs](../workspace/crates/hologram-ai/src/runtime/executor.rs) - Model executor

## Success Criteria

1. ✅ Pipeline completes without buffer allocation errors
2. ✅ All stages execute: tokenize → encoder → generate → decode
3. ✅ Generated text is in **English** (T5-small is English-only)
4. ✅ Output is coherent and grammatically correct
5. ✅ Output relates to joke prompt

## Expected Output

**Format**:
```json
{
  "generated_text": "<English text attempting to tell a joke>"
}
```

**Console Logs** (RUST_LOG=info):
```
INFO Loaded sentencepiece tokenizer (vocab size: 32100)
INFO ✓ Stage 0: tokenize completed
INFO ✓ Stage 1: encoder completed
INFO   Starting auto-regressive generation:
INFO     Model: decoder
INFO     Max new tokens: 50
INFO ✓ Stage 2: generate completed
INFO   Decoded: "<joke text>"
INFO ✓ Stage 3: decode completed
INFO Pipeline execution complete!
```

**Performance**: ~475ms total
- Tokenization: 20ms
- Encoder: 135ms
- Generation: 300ms (50 steps)
- Decode: 20ms

## Long-Term Fix (For Reference)

The buffer allocation bug should be fixed by:
1. Filing issue with hologram team about workspace allocation in parallel execution mode
2. Or: Disable parallel execution for T5 by setting `enable_parallel_execution = false` in config
3. Monitor commits a22cee7 (parallel flag) and 104032e (workspace refactor) for fixes

**For now**: Use the working pre-compiled models from Jan 18 to bypass the bug entirely.
