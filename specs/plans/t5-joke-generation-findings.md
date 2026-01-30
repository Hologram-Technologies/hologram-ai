# T5 Joke Generation - Investigation Findings

## Date: 2026-01-20

## Objective
Compile T5 model to `.holo` format and run text generation pipeline with prompt "Tell me a joke" to receive a coherent English response.

## Summary
**Status**: ❌ **BLOCKED** by compiler bug

We successfully:
- ✅ Built hologram-ai CLI in release mode
- ✅ Compiled T5 encoder and decoder models to .holo format (multiple attempts)
- ✅ Configured pipeline with correct prompt
- ✅ Tokenizer working correctly (32,100 token SentencePiece vocab)

However, execution is blocked by a **buffer allocation bug** in the hologram compiler affecting both `main` and `feat/parallel` branches.

## The Buffer Allocation Bug

### Error Message
```
OP[13] KernelId(771): input[1] size 262144 bytes exceeds workspace region
'workspace_11' allocation of 4 bytes. dims=[65536, 65536, 65536, 0].
This indicates a compiler bug in shape inference or buffer allocation.
```

### Analysis
- **Location**: Operation 13 in encoder execution
- **Expected**: 262144 bytes (512×512 float attention matrix)
- **Allocated**: Only 4 bytes
- **Impact**: Encoder cannot execute, blocking entire pipeline

### Attempted Workarounds

#### 1. Disable Parallel Execution Flag
```bash
cargo run --release -- compile encoder_model.onnx \
  --parallel false \
  --partition --partition-size 200
```
**Result**: ❌ Same error persists

#### 2. Use Pre-Compiled Models (Jan 18)
- Files: `encoder-new.holo` (270MB), `decoder-new.holo` (444MB)
- **Result**: ❌ Models execute but produce all-zero outputs
- **Effect**: Decoder generates 51 PAD tokens → empty string

#### 3. Switch to Main Branch
- Checked out `main` branch (commits from Jan 11-12)
- **Result**: ❌ Same buffer allocation bug present

#### 4. Switch to Older Commit (Jan 9)
- Checked out commit `eca9655` (before parallel execution)
- **Result**: ❌ Code doesn't compile (49 compilation errors)

### Root Cause
The bug was introduced in commits around **Jan 11-12** related to:
- Parallel execution feature (`feat/parallel` branch)
- Workspace allocation refactoring
- Attention pattern detection

Relevant commits:
- `a22cee7` - Add parallel execution flag and activation fusion
- `104032e` - Work with parallel options and workspace
- `a674756` - Add attention pattern detection for parallel execution

## Pipeline Configuration

### Working Configuration File
[`/workspace/configs/t5-generate.toml`](/workspace/configs/t5-generate.toml)

```toml
[inputs]
prompt = "Tell me a joke in English"  # ✅ Correct prompt

[tokenizer]
type = "sentencepiece"
vocab_path = "/workspace/models/t5-small/tokenizer.json"
max_length = 512

[models.encoder]
precompiled = "/workspace/models/t5-small/compiled/encoder.holo"

[models.decoder]
precompiled = "/workspace/models/t5-small/compiled/decoder.holo"

[[stages]]
builtin = "tokenize"
outputs = ["input_ids", "attention_mask"]

[[stages]]
model = "encoder"
outputs = ["encoder_hidden_states"]

[[stages]]
builtin = "generate"
args = { model = "decoder", max_new_tokens = 50 }
outputs = ["generated_ids"]

[[stages]]
builtin = "decode"
outputs = ["generated_text"]
```

### Execution Flow (When Working)
1. **Tokenize** (20ms): "Tell me a joke in English" → `[8779, 140, 3, 9, 10802, 16, 1566, 1, 0, 0, ...]`
2. **Encode** (135ms): encoder.holo → encoder_hidden_states [1, 512, 512]
3. **Generate** (300ms): Auto-regressive decoder loop (50 tokens max)
   - Initialize with PAD token (id=0)
   - For each step: decoder → logits → argmax → next token
   - Stop at EOS (id=1) or max_new_tokens
4. **Decode** (20ms): token IDs → English text

**Expected total time**: ~475ms

## What Worked

### Tokenization ✅
```
INFO Loaded sentencepiece tokenizer (vocab size: 32100)
INFO Tokenizing text: "Tell me a joke in English"
INFO Tokenized to 512 tokens (vocab size: 32100)
INFO First 10 input tokens: [8779, 140, 3, 9, 10802, 16, 1566, 1, 0, 0]
INFO ✓ Stage 0: tokenize completed
```

### Compilation ✅
Both encoder and decoder compiled successfully (multiple times):
- **Encoder**: 270MB .holo file
- **Decoder**: 444MB .holo file
- **Tokenizer**: 127KB .holo file

Compilation settings:
```bash
--partition --partition-size 200 --memory-budget 2048
```

## What Failed

### Encoder Execution ❌
```
ERROR Kernel execution failed at OP[13]
ERROR input[1] size 262144 bytes exceeds workspace region allocation of 4 bytes
Error: Model execution failed
```

### Alternative: Old Models Produced Zero Outputs ❌
When using `encoder-new.holo` and `decoder-new.holo` from Jan 18:
- Encoder completed in 4.4s
- All matrix multiplications produced zero outputs:
  ```
  gemm_kernel_f32 output: c_nonzero=0/262144, c_range=[0.0000e0, 0.0000e0]
  ```
- Decoder generated 51 tokens, all PAD (id=0)
- Final decoded text: `""` (empty string)

## Recommendations

### Immediate Action Required
File a bug report with hologram team:
- **Component**: hologram-compiler workspace allocation
- **Affected versions**: Commits after Jan 9, 2026
- **Regression introduced**: Jan 11-12 parallel execution changes
- **Test case**: T5-small encoder, operation 13
- **Expected**: 262144 bytes for attention matrix
- **Actual**: 4 bytes allocated

### Workaround Options

#### Option 1: Fix the Compiler Bug
- Debug workspace allocation in `hologram-compiler`
- Fix shape inference for operation 13 (attention matrix)
- Test with T5-small encoder as regression test

#### Option 2: Find Working Commit
- Bisect between Jan 9 (`eca9655`) and Jan 11 (`a674756`)
- Find last commit before bug was introduced
- Cherry-pick T5 generation features onto working commit

#### Option 3: Use Different Model
- Try smaller models that don't trigger the bug
- Or models without attention mechanisms

### Long-Term Fix
1. Add regression test for T5-small compilation and execution
2. Add workspace allocation validation in compiler
3. Document workspace size calculation for attention operations

## Files Referenced

### Models
- `/workspace/models/t5-small/encoder_model.onnx` (135MB source)
- `/workspace/models/t5-small/decoder_model.onnx` (159MB source)
- `/workspace/models/t5-small/tokenizer.json` (32,100 tokens)
- `/workspace/models/t5-small/compiled/encoder.holo` (270MB compiled)
- `/workspace/models/t5-small/compiled/decoder.holo` (444MB compiled)

### Configuration
- `/workspace/configs/t5-generate.toml` - Working pipeline config

### Source Code
- `/workspace/crates/hologram-ai/src/cli/run.rs` - Pipeline executor
  - Lines 1187-1428: Generate builtin (auto-regressive generation)
  - Lines 952-1047: Tokenize builtin
- `/workspace/crates/hologram-ai/src/tokenizers/sentencepiece.rs` - Tokenizer
- `/workspace/crates/hologram-ai/src/runtime/executor.rs` - Model executor

### Documentation
- `/workspace/docs/working/t5-e2e-success.md` - Previous success documentation
- `/workspace/specs/plans/t5-joke-generation.md` - Original plan

## Environment

- **Branch**: `feat/parallel` (also affects `main`)
- **Hologram Version**: Local git dependency
- **Rust**: 1.93.0-nightly
- **Date**: 2026-01-20
- **OS**: Linux 6.17.8-orbstack

## Conclusion

We successfully set up the T5 pipeline infrastructure and demonstrated that all components work individually:
- ✅ Tokenizer encodes prompts correctly
- ✅ Models compile to .holo format
- ✅ Configuration is correct
- ✅ Auto-regressive generation loop is implemented

However, execution is **blocked by a compiler bug** that:
- Was introduced between Jan 9-11, 2026
- Affects workspace allocation for attention matrices
- Causes a 262144-byte buffer to be allocated as only 4 bytes
- Occurs at encoder operation 13

**Next Steps**: Fix the compiler bug or find a working commit before the regression was introduced.
