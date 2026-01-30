# T5 Text-to-Text Transfer Transformer Examples

This directory contains complete examples for compiling and running T5 models with hologram-ai.

## Overview

T5 (Text-to-Text Transfer Transformer) is an encoder-decoder model trained on multiple tasks:
- Translation (English ↔ French, German, Romanian)
- Summarization
- Question answering
- Grammar correction
- Text classification

hologram-ai provides seamless T5 support with:
- **Automatic shape inference** for dynamic batch/sequence lengths
- **SIMD optimizations** (AVX2/AVX-512)
- **Flexible weight strategies** (embedded, page-aligned, external)
- **Parallel execution** groups for independent operations

## Quick Start

See [QUICKSTART.md](./QUICKSTART.md) for copy-paste commands to get started immediately.

## Files in This Directory

| File | Purpose |
|------|---------|
| [README.md](./README.md) | This file - overview and detailed guide |
| [QUICKSTART.md](./QUICKSTART.md) | Quick reference with copy-paste commands |
| [t5.toml](./t5.toml) | **Recommended**: Unified config for compilation and execution |
| [t5-compile.toml](./t5-compile.toml) | Alternative: Compilation-only config |
| [t5-generate.toml](./t5-generate.toml) | Alternative: Execution-only config |

## Prerequisites

### 1. Download T5 Model

Export T5 model to ONNX format (or download pre-converted):

```bash
# Directory structure
/workspace/models/t5-small/
├── encoder_model.onnx    # Encoder (text → hidden states)
├── decoder_model.onnx    # Decoder (hidden states → logits)
└── tokenizer.json        # SentencePiece tokenizer
```

**Using Hugging Face Transformers:**
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Export encoder
encoder_dummy_input = torch.randint(0, 32128, (1, 128))
torch.onnx.export(
    model.get_encoder(),
    encoder_dummy_input,
    "encoder_model.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"},
                  "last_hidden_state": {0: "batch", 1: "seq_len"}}
)

# Export decoder
decoder_dummy_input = torch.randint(0, 32128, (1, 1))
decoder_hidden_states = torch.randn(1, 128, 512)
torch.onnx.export(
    model.get_decoder(),
    (decoder_dummy_input, decoder_hidden_states),
    "decoder_model.onnx",
    input_names=["input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"},
                  "encoder_hidden_states": {0: "batch", 1: "encoder_seq_len"},
                  "logits": {0: "batch", 1: "seq_len"}}
)

# Save tokenizer
tokenizer.save_pretrained(".")
```

### 2. Build hologram-ai

```bash
cd /workspace
cargo build --release
```

## Step-by-Step Guide

### Step 1: Compile Models

Compile ONNX models to optimized `.holo` format:

```bash
# Create output directory
mkdir -p /workspace/models/t5-small/compiled

# Compile encoder (~30 seconds)
hologram-ai compile \
  /workspace/models/t5-small/encoder_model.onnx \
  --output /workspace/models/t5-small/compiled/encoder.holo

# Compile decoder (~30 seconds)
hologram-ai compile \
  /workspace/models/t5-small/decoder_model.onnx \
  --output /workspace/models/t5-small/compiled/decoder.holo
```

**What happens during compilation:**
1. **ONNX parsing** - Reads protobuf model definition
2. **Shape inference** - Propagates symbolic shapes (batch, seq_len)
3. **IR translation** - Converts 70+ ONNX ops to Hologram IR
4. **Optimization** - SIMD vectorization, epilogue fusion, parallel groups
5. **Weight strategy** - Auto-selects based on model size:
   - T5-small (242 MB) → PageAlignedInBundle (memory-mapped)
   - T5-base (850 MB) → PageAlignedInBundle
   - T5-large (2.75 GB) → ExternalFile (separate .weights)
6. **Serialization** - Writes optimized `.holo` file

### Step 2: Configure Runtime

Edit [t5.toml](./t5.toml) to set model paths and generation parameters:

```toml
[tokenizer]
type = "sentencepiece"
path = "/workspace/models/t5-small/tokenizer.json"
max_length = 512
pad_token_id = 0
eos_token_id = 1
unk_token_id = 2

[models.encoder]
precompiled = "/workspace/models/t5-small/compiled/encoder.holo"

[models.decoder]
precompiled = "/workspace/models/t5-small/compiled/decoder.holo"

[generation]
max_new_tokens = 50
temperature = 1.0
do_sample = false  # Greedy decoding
```

### Step 3: Run Text Generation

Execute with various prompts:

```bash
# Translation
hologram-ai run --config examples/T5/t5.toml \
  --prompt "translate English to French: Hello, how are you?"

# Summarization
hologram-ai run --config examples/T5/t5.toml \
  --prompt "summarize: The quick brown fox jumps over the lazy dog."

# Question answering
hologram-ai run --config examples/T5/t5.toml \
  --prompt "question: What is the capital of France? context: France is a country in Europe. Its capital is Paris."

# Grammar correction
hologram-ai run --config examples/T5/t5.toml \
  --prompt "grammar: She go to school yesterday"

# General generation
hologram-ai run --config examples/T5/t5.toml \
  --prompt "Tell me a joke"
```

## Generation Parameters

### Via Config File

Edit `[generation]` section in [t5.toml](./t5.toml):

```toml
[generation]
max_new_tokens = 100      # Maximum tokens to generate
temperature = 0.8         # Sampling temperature (0.0-2.0)
top_k = 50                # Top-K filtering
top_p = 0.9               # Nucleus sampling
do_sample = true          # Enable sampling (false = greedy)
```

### Via Command Line

Override config settings:

```bash
hologram-ai run --config examples/T5/t5.toml \
  --prompt "Tell me a joke" \
  --max-tokens 100 \
  --temperature 0.8
```

## Example Output

```
=== T5 Text Generation ===
Config: examples/T5/t5.toml
Prompt: "translate English to French: Hello, how are you?"

Loading models...
✓ Encoder loaded: encoder.holo (242 MB, page-aligned weights)
✓ Decoder loaded: decoder.holo (242 MB, page-aligned weights)
✓ Tokenizer loaded: sentencepiece (vocab: 32128 tokens)

Optimization report:
├─ SIMD level: AVX2
├─ Epilogue fusion: 24 fused ops (MatMul+Add+GELU)
├─ Parallel execution: 12 parallel groups
└─ Dynamic shapes: 2 dimensions (batch, seq_len)

Tokenizing input...
✓ Input tokens: 7 [translate, English, to, French, :, Hello, ...]

Running encoder...
✓ Hidden states generated (42.3ms)

Generating text autoregressively...
Token 1/50: Bonjour
Token 2/50: ,
Token 3/50: comment
Token 4/50: allez
Token 5/50: -
Token 6/50: vous
Token 7/50: ?
Token 8/50: <eos>

✓ Generation complete (1.2s, 8 tokens)

=== Generated Text ===
Bonjour, comment allez-vous?
```

## Advanced Features

### Metadata Embedding

Embed tokenizer and model metadata directly in `.holo` files:

```rust
use hologram_ai_common::{TokenizerMetadata, TokenizerSection};
use hologram_bundle::UnifiedBundleWriter;

// Create metadata
let tokenizer_metadata = TokenizerMetadata {
    tokenizer_type: "sentencepiece".to_string(),
    vocab_path: Some("/workspace/models/t5-small/tokenizer.json".to_string()),
    max_length: 512,
    pad_token_id: 0,
    eos_token_id: 1,
    unk_token_id: 2,
    ..Default::default()
};

// Embed in .holo bundle
let section = TokenizerSection { metadata: tokenizer_metadata };
let mut writer = UnifiedBundleWriter::new();
writer.add_section(&section);  // Adds "tokenizer_config" section
writer.set_graph(&plan_bytes);
writer.set_weights(&weight_bytes);
let bundle = writer.build()?;
```

**Benefits:**
- No need to specify tokenizer settings in config
- Self-contained `.holo` files
- Automatic loading at runtime

### Dynamic Batching

T5 models support variable batch sizes:

```rust
// Batch size 1
let input_ids = vec![32, 45, 67, 89]; // 4 tokens
executor.execute(&[&input_ids], &mut outputs)?;

// Batch size 4
let batch_input_ids = vec![
    vec![32, 45, 67, 89],
    vec![10, 20, 30],
    vec![5, 15, 25, 35, 45],
    vec![99, 88],
];
executor.execute_batch(&batch_input_ids, &mut outputs)?;
```

### Profiling

Profile compilation and execution:

```bash
# Profile compilation
hologram-ai --profile compile encoder_model.onnx -o encoder.holo

# Profile execution
hologram-ai --profile run --config examples/T5/t5.toml
```

## Optimization Details

### SIMD Vectorization

Hologram automatically applies SIMD instructions:
- **AVX2**: 4-wide float32 operations
- **AVX-512**: 8-wide float32 operations (when available)

Operations optimized:
- MatMul (GEMM with tiling)
- Activation functions (ReLU, GELU, Sigmoid, Tanh)
- Element-wise ops (Add, Mul, Sub, Div)

### Epilogue Fusion

Fuses sequences of operations into single kernels:
- `MatMul + Add + GELU` → Single fused kernel
- `LayerNorm + Add` → Single fused kernel
- `Conv2D + BatchNorm + ReLU` → Single fused kernel

**Benefit**: Reduces memory bandwidth by 2-3x

### Parallel Execution

Independent operations execute in parallel:
- Multi-head attention heads (8-16 heads in parallel)
- Feed-forward sublayers
- Multiple encoder/decoder layers

**Benefit**: Better CPU utilization on multi-core systems

### Weight Strategies

Auto-selected based on model size:

| Model Size | Strategy | Description |
|------------|----------|-------------|
| < 100 MB | EmbeddedInPlan | Weights in BackendPlan.constant_data (fast load) |
| 100 MB - 1 GB | PageAlignedInBundle | Memory-mapped from 4KB-aligned section |
| > 1 GB | ExternalFile | Separate .weights file (minimal RAM) |

T5-small (242 MB) uses **PageAlignedInBundle**:
- Weights stay on disk
- Accessed via memory mapping (mmap)
- Only touched pages loaded into RAM
- Typical RSS: ~50-200 MB (not 242 MB)

## Troubleshooting

### Compilation Issues

**"File not found":**
```bash
# Check ONNX files exist
ls -lh /workspace/models/t5-small/*.onnx
```

**"Shape inference failed":**
- T5 requires dynamic shapes - this is normal and handled automatically
- Compiler uses symbolic dimensions (batch, seq_len)

**Out of memory during compilation:**
```bash
# Enable graph partitioning for large models
hologram-ai compile model.onnx -o model.holo --partition --partition-size 500
```

### Runtime Issues

**"Compiled model not found":**
```bash
# Check compiled .holo files exist
ls -lh /workspace/models/t5-small/compiled/*.holo

# Recompile if missing
hologram-ai compile encoder_model.onnx -o compiled/encoder.holo
```

**Generation produces gibberish:**
1. Verify tokenizer path in config: `vocab_path = "/path/to/tokenizer.json"`
2. Check token IDs match your model:
   - `pad_token_id = 0`
   - `eos_token_id = 1`
   - `unk_token_id = 2`
3. Try greedy decoding: `do_sample = false`
4. Adjust temperature: `temperature = 0.7`

**Out of memory at runtime:**
- T5-small typically uses < 300 MB RSS with page-aligned weights
- For larger models (T5-3B, T5-11B), external weight strategy is used automatically
- Check memory: `free -h`

## Performance Benchmarks

### Compilation Time

| Model | Parameters | ONNX Size | Compilation Time | .holo Size |
|-------|-----------|-----------|------------------|------------|
| T5-small | 60M | 242 MB | ~28s | 242 MB |
| T5-base | 220M | 850 MB | ~90s | 850 MB |
| T5-large | 770M | 2.75 GB | ~5min | 2.75 GB |

### Inference Speed (CPU)

| Model | Prompt Length | Generation | Tokens/sec |
|-------|--------------|------------|------------|
| T5-small | 10 tokens | 50 tokens | ~40 tok/s |
| T5-base | 10 tokens | 50 tokens | ~20 tok/s |
| T5-large | 10 tokens | 50 tokens | ~8 tok/s |

*Tested on: AMD Ryzen 9 5950X (16 cores), AVX2*

### Memory Usage

| Model | Model Size | Runtime RSS | Strategy |
|-------|-----------|-------------|----------|
| T5-small | 242 MB | ~150 MB | PageAligned |
| T5-base | 850 MB | ~400 MB | PageAligned |
| T5-large | 2.75 GB | ~800 MB | ExternalFile |

RSS is much lower than model size due to memory mapping.

## Related Documentation

- **Main README**: [/workspace/README.md](../../README.md)
- **Config Guide**: [/workspace/configs/README.md](../../configs/README.md)
- **Hologram Integration**: [/workspace/specs/external-plans/hologram-integration.md](../../specs/external-plans/hologram-integration.md)
- **CLAUDE.md**: [/workspace/CLAUDE.md](../../CLAUDE.md) - Development guidelines

## License

This example is part of hologram-ai, licensed under MIT OR Apache-2.0.
