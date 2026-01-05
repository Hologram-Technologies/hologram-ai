# hologram-onnx CLI Usage Guide

## Overview

The `hologram-onnx` CLI provides commands for compiling ONNX models to `.holo` format and running compiled models with unified configuration files.

## Commands

### 1. Compile Command

Compile ONNX models to `.holo` format for execution with the hologram runtime.

#### Basic Usage

```bash
# Compile a single ONNX file
hologram-onnx compile model.onnx -o output_name

# With partitioning for large models
hologram-onnx compile large_model.onnx -o output \
  --partition --partition-size 200

# With memory budget
hologram-onnx compile model.onnx -o output \
  --memory-budget 2048  # 2GB limit
```

#### Using Configuration Files

```bash
# Compile all models specified in a config file
hologram-onnx compile --config configs/my_pipeline.toml -o output_dir/
```

**Note:** When using `--config`, the output path (`-o`) should be a directory where each model will be saved as `<model_name>.holo`.

#### Example Config (TOML)

```toml
name = "my-pipeline"
version = "1.0"
description = "My ONNX pipeline"

[models]
encoder = "models/encoder.onnx"
decoder = "models/decoder.onnx"

[compiler]
weight_threshold = 4096
enable_partitioning = true
partition_size = 200
decompose_conv2d = true
decompose_pooling = true
pack_weights = true
memory_budget = 2048  # MB
enable_resize_upscaling = true
```

Running `hologram-onnx compile --config my_pipeline.toml -o compiled/` will create:
- `compiled/encoder.holo`
- `compiled/decoder.holo`

#### Compilation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--partition` | Enable graph partitioning for large models | `false` |
| `--partition-size` | Nodes per partition | `500` |
| `--memory-budget` | Memory limit in MB | unlimited |
| `--weight-threshold` | Threshold for external weight storage (bytes) | `4096` |
| `--input-shape` | Concrete shapes for dynamic inputs (e.g., `input=1,3,224,224`) | - |

### 2. Run Command

Execute a compiled pipeline using a unified configuration file.

```bash
hologram-onnx run --config my_pipeline.toml \
  --input text="Hello world" \
  --input seed=42 \
  --output results/
```

**Prerequisites:** Models must be compiled to `.holo` format first using the `compile` command.

### 3. Other Commands

#### Info

Display information about an ONNX model:

```bash
hologram-onnx info model.onnx --detailed
```

#### Validate

Validate an ONNX model:

```bash
hologram-onnx validate model.onnx --check-ops
```

#### Download

Download models from Hugging Face:

```bash
hologram-onnx download stable-diffusion-v1-5 -o models/sd-v1-5/
```

#### Bundle

Bundle multiple `.holo` files:

```bash
hologram-onnx bundle encoder.holo decoder.holo -o pipeline.bundle

# Or from config
hologram-onnx bundle --config my_pipeline.toml -o pipeline.bundle
```

#### Extract

Extract models from a bundle:

```bash
hologram-onnx extract pipeline.bundle -o extracted/
```

#### List

List models in a bundle:

```bash
hologram-onnx list pipeline.bundle
```

## Complete Workflow Example

### 1. Download or Export Model

```bash
# Option A: Download from HuggingFace
hologram-onnx download google/t5-small -o models/t5-small/

# Option B: Export with Optimum
pip install optimum[exporters]
optimum-cli export onnx \
  --model google/t5-small \
  --task text2text-generation-with-past \
  models/t5-small/
```

### 2. Create Configuration

Create `t5_pipeline.toml`:

```toml
name = "t5-small"
description = "T5 text-to-text generation"

[inputs]
text = { type = "text", default = "translate English to French: Hello" }

[models]
encoder = "models/t5-small/encoder_model.onnx"
decoder = "models/t5-small/decoder_model.onnx"

[compiler]
enable_partitioning = true
partition_size = 200
memory_budget = 2048
pack_weights = true

[outputs]
result = { tensor = "encoder_hidden_states", handler = "json" }
```

### 3. Compile Models

```bash
mkdir -p compiled/t5-small
hologram-onnx compile --config t5_pipeline.toml -o compiled/t5-small/
```

This creates:
- `compiled/t5-small/encoder.holo`
- `compiled/t5-small/encoder.weights` (if needed)
- `compiled/t5-small/decoder.holo`
- `compiled/t5-small/decoder.weights` (if needed)

### 4. Run Pipeline

```bash
hologram-onnx run --config t5_pipeline.toml \
  --input text="translate English to German: Good morning" \
  --output results/
```

## Configuration File Format

See [configs/examples/](../configs/examples/) for complete examples:
- `phi-2.toml` - LLM configuration
- `whisper-unified.toml` - Speech recognition (encoder-decoder)
- `t5.toml` - Text-to-text generation (encoder-decoder)

## Troubleshooting

### "Model file not found" when using --config

Make sure the paths in your config are relative to the config file's directory, or use absolute paths.

### "Failed to write .holo file: No such file or directory"

When using `--config`, the output path must be an existing directory:

```bash
mkdir -p output_dir
hologram-onnx compile --config my_config.toml -o output_dir/
```

### "Memory budget exceeded"

Increase the memory budget or enable partitioning:

```bash
hologram-onnx compile model.onnx -o output \
  --partition --partition-size 200 \
  --memory-budget 4096  # 4GB
```

### "Unsupported operation"

Check the operation list in the error message. The operation may not be implemented yet, or you may need a different opset version.

## Performance Tips

1. **Enable partitioning** for large models (3000+ nodes):
   ```bash
   --partition --partition-size 200
   ```

2. **Set memory budget** to avoid OOM:
   ```bash
   --memory-budget 2048  # 2GB
   ```

3. **Use weight packing** (enabled by default in config files) for faster runtime loading

4. **External weights** for large models:
   ```bash
   --weight-threshold 1024  # Smaller = more external
   ```

## See Also

- [Examples README](../examples/README.md) - Code examples
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [Unified Config Documentation](../configs/README.md) - Configuration format details
