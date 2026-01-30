# Configuration Files

This directory contains configuration files for hologram-ai models.

## Quick Start (T5)

**Single unified config file:** [t5.toml](./t5.toml) - contains all settings for both compilation and execution.

```bash
# 1. Compile models (once)
mkdir -p /workspace/models/t5-small/compiled
hologram-ai compile /workspace/models/t5-small/encoder_model.onnx \
  --output /workspace/models/t5-small/compiled/encoder.holo
hologram-ai compile /workspace/models/t5-small/decoder_model.onnx \
  --output /workspace/models/t5-small/compiled/decoder.holo

# 2. Run generation (many times)
hologram-ai run --config configs/t5.toml
```

## T5 Model Workflow

### 1. Compilation

Compile T5 ONNX models to optimized .holo format:

**Option A: Compile as pipeline bundle (recommended)**
```bash
hologram-ai compile-pipeline \
  --output /workspace/models/t5-small/compiled/t5.holom \
  --config configs/t5-compile.toml
```

**Option B: Compile models individually**
```bash
# Create output directory
mkdir -p /workspace/models/t5-small/compiled

# Compile encoder
hologram-ai compile \
  /workspace/models/t5-small/encoder_model.onnx \
  --output /workspace/models/t5-small/compiled/encoder.holo

# Compile decoder
hologram-ai compile \
  /workspace/models/t5-small/decoder_model.onnx \
  --output /workspace/models/t5-small/compiled/decoder.holo
```

This will:
- Auto-select weight storage strategy based on model size
- Enable ONNX shape inference (critical for T5)
- Apply Hologram optimizations (SIMD, fusion, etc.)

**Outputs:**
- **Pipeline**: `/workspace/models/t5-small/compiled/t5.holom` (single bundle)
- **Individual**: `encoder.holo` and `decoder.holo` (separate files)

### 2. Text Generation

Run text generation with precompiled models:

```bash
hologram-ai run --config configs/t5-generate.toml
```

Or with a custom prompt:

```bash
hologram-ai run --config configs/t5-generate.toml --prompt "translate English to German: The weather is nice today"
```

**Example prompts for T5:**
- Translation: `"translate English to French: <text>"`
- Summarization: `"summarize: <long text>"`
- Question answering: `"question: <question> context: <context>"`
- Text generation: `"Tell me a joke"`

## Configuration Files

### [t5.toml](./t5.toml) - **Recommended: Unified Configuration**

Single configuration file containing all settings for both compilation and execution.

**Key sections:**
- `[tokenizer]` - Tokenizer settings (shared by both compilation and execution)
- `[models.encoder]` / `[models.decoder]` - Model paths (both ONNX source and compiled output)
- `[compilation]` - Compilation settings (weight strategy, shape inference, optimizations)
- `[generation]` - Generation parameters (max_new_tokens, temperature, sampling)
- `[[stages]]` - Execution pipeline stages

**Benefits:**
- Single source of truth for all settings
- No duplicate tokenizer/model configuration
- Easier to maintain and update
- Clear separation between compilation and execution sections

### Alternative: Separate Configuration Files

If you prefer separate files for compilation vs execution:

- [t5-compile.toml](./t5-compile.toml) - Compilation-only settings
- [t5-generate.toml](./t5-generate.toml) - Execution-only settings

**Note:** These maintain duplicate sections (`[tokenizer]`, `[models]`) which requires updating both files when settings change. The unified config above is recommended to avoid this duplication.

## Directory Structure

```
configs/
├── README.md              # This file
├── t5-compile.toml        # T5 compilation config
├── t5-generate.toml       # T5 execution config
├── examples/              # Example configs for various models
│   ├── mnist-minimal.toml
│   ├── resnet-unified.toml
│   ├── whisper-unified.toml
│   └── ...
└── old-tests/             # Legacy test configs (archived)
    └── ...
```

## Weight Strategies

The compilation config supports three weight storage strategies:

- **auto** (recommended) - Auto-select based on model size:
  - < 100MB: Embedded in .holo file
  - 100MB-1GB: Page-aligned section in .holo file (memory-mapped)
  - > 1GB: Separate .weights file

- **embedded** - All weights in .holo file (fast loading, more RAM)
- **page_aligned** - Weights in page-aligned section (memory-mapped)
- **external** - Weights in separate .weights file (minimal RAM)

## Customization

### Changing the Model

Edit `t5-compile.toml` and `t5-generate.toml` to point to your model:

```toml
[models.encoder]
path = "/path/to/your/encoder_model.onnx"
precompiled = "/path/to/your/compiled/encoder.holo"
```

### Adjusting Generation Parameters

Edit `t5-generate.toml`:

```toml
[generation]
max_new_tokens = 100      # Generate more tokens
temperature = 0.8         # More random (1.0 = neutral)
do_sample = true          # Enable sampling
top_k = 50                # Top-K filtering
top_p = 0.9               # Nucleus sampling
```

### Custom Prompts

**Option 1:** Edit the config file:
```toml
[inputs]
prompt = "Your custom prompt here"
```

**Option 2:** Override via CLI:
```bash
hologram-ai run --config configs/t5-generate.toml --prompt "Your prompt"
```

## Examples Directory

The `examples/` directory contains reference configurations for various models:

- **mnist-minimal.toml** - Simple MNIST classifier
- **resnet-unified.toml** - ResNet image classification
- **whisper-unified.toml** - Whisper speech recognition
- **sd-unified.toml** - Stable Diffusion image generation
- **t5.toml** - Advanced T5 configuration with all options

See `examples/T5_USAGE.md` for detailed T5 usage documentation.

## Troubleshooting

### Compilation fails

1. Check that ONNX files exist at the specified paths
2. Enable shape inference: `shape_inference = true`
3. Check ONNX model validity: `hologram-ai validate <model.onnx>`

### Generation produces gibberish

1. Verify tokenizer path is correct
2. Check that pad_token_id, eos_token_id match your model
3. Try adjusting temperature (0.7-1.2)
4. Use greedy decoding first: `do_sample = false`

### Out of memory

1. Use external weight strategy: `weight_strategy = "external"`
2. Reduce max_new_tokens
3. Reduce max_length for tokenization

## Related Documentation

- [Hologram Integration Guide](../specs/external-plans/hologram-integration.md)
- [T5 Usage Guide](./examples/T5_USAGE.md)
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
