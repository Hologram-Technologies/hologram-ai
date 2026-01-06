# T5 Model Usage Guide

This guide shows how to use the new interactive `run` command with T5 models.

## Prerequisites

1. **Compile models** (if not already done):
```bash
# Compile T5 encoder
cargo run -- compile \
  models/t5-small/encoder_model.onnx \
  -o models/t5-small/compiled/encoder \
  --partition --partition-size 200

# Compile T5 decoder
cargo run -- compile \
  models/t5-small/decoder_model.onnx \
  -o models/t5-small/compiled/decoder \
  --partition --partition-size 200
```

2. **Build with text-output feature**:
```bash
cargo build --features text-output
```

## Usage Methods

### Method 1: Direct Execution (Recommended for Quick Tests)

This is the simplest way to test T5 models with a prompt:

```bash
# Run encoder with default prompt
cargo run --features text-output -- run \
  models/t5-small/compiled/encoder.holo \
  --tokenizer models/t5-small/tokenizer.json

# Run encoder with custom prompt
cargo run --features text-output -- run \
  models/t5-small/compiled/encoder.holo \
  --prompt "Tell me a joke" \
  --tokenizer models/t5-small/tokenizer.json

# Run decoder with translation task
cargo run --features text-output -- run \
  models/t5-small/compiled/decoder.holo \
  --prompt "Translate English to French: The cat sits on the mat" \
  --tokenizer models/t5-small/tokenizer.json \
  --max-length 256
```

**Parameters:**
- First argument: Path to `.holo` model file (encoder or decoder)
- `--prompt`: Text input (optional, uses default if not specified)
- `--tokenizer`: Path to `tokenizer.json` (default: `tokenizer.json`)
- `--max-length`: Maximum sequence length (default: `512`)

### Method 2: Config-Based Execution (For Complex Pipelines)

Use a configuration file for multi-model pipelines:

```bash
# Simple config
cargo run --features text-output -- run \
  --config configs/examples/t5-simple.toml \
  -i prompt="Tell me a joke"

# Full pipeline config
cargo run --features text-output -- run \
  --config configs/examples/t5-interactive.toml \
  -i prompt="Translate English to German: Good morning" \
  -i max_length=128
```

**Config files:**
- `configs/examples/t5-simple.toml` - Minimal test configuration
- `configs/examples/t5-interactive.toml` - Full pipeline with generation loop

## Output

The command will show:

1. **Tokenization Details**:
   - Number of input tokens
   - Token IDs (first 20 shown)
   - Padded sequence length

2. **Model Loading**:
   - Model file path
   - Model size in MB
   - Input/output shapes

3. **Execution Flow**:
   - Steps needed for full T5 inference
   - Current status (tokenization ✓, runtime ⚠️)

### Example Output

```
Direct model execution mode
Model: models/t5-small/compiled/encoder.holo
Tokenizer: models/t5-small/tokenizer.json
Prompt: "Tell me a joke"
Loading tokenizer...
Tokenizing input...
Input tokens: 6 tokens
Padded to 128 tokens
Input shape: [batch=1, seq_len=128]
Attention mask shape: [batch=1, seq_len=128]
Loading model: models/t5-small/compiled/encoder.holo
Model size: 347.42 MB

=== T5 Execution Flow ===
✓ Tokenization complete
✓ Input prepared: 6 tokens
✓ Model loaded: 347.42 MB

⚠  Model execution requires hologram runtime integration.
   The .holo format is loaded and ready to execute.
   Tokenization pipeline is working correctly.

To execute:
  1. Encoder input_ids shape: [1, 128]
  2. Encoder attention_mask shape: [1, 128]
  3. Run encoder → get last_hidden_state [1, 128, 512]
  4. Decoder generates output tokens autoregressively
  5. Detokenize output_ids → final text

Example detokenization:
  Sample output: "estthe Le being"
```

## Current Status

✅ **Working:**
- CLI argument parsing
- Tokenization (encoding text → token IDs)
- Model loading (.holo format)
- Input preparation
- Detokenization (token IDs → text)

⚠️ **Pending:**
- Hologram runtime integration for actual .holo execution
- Encoder forward pass
- Decoder autoregressive generation
- Full end-to-end inference

## Testing Different Prompts

### Translation Tasks
```bash
# English to French
cargo run --features text-output -- run encoder.holo \
  --prompt "Translate English to French: Hello, how are you?"

# English to German
cargo run --features text-output -- run encoder.holo \
  --prompt "Translate English to German: Good morning"

# English to Spanish
cargo run --features text-output -- run encoder.holo \
  --prompt "Translate English to Spanish: Thank you very much"
```

### Summarization
```bash
cargo run --features text-output -- run encoder.holo \
  --prompt "Summarize: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
```

### Question Answering
```bash
cargo run --features text-output -- run encoder.holo \
  --prompt "Question: What is the capital of France? Answer:"
```

## Troubleshooting

### Error: Model file not found
```
Solution: Run compile command first (see Prerequisites)
```

### Error: tokenizer.json not found
```
Solution: Specify full path with --tokenizer flag:
  --tokenizer models/t5-small/tokenizer.json
```

### Error: text-output feature required
```
Solution: Build with feature flag:
  cargo build --features text-output
```

## Next Steps

Once hologram runtime integration is complete, this command will:
1. Execute the encoder on tokenized input
2. Run the decoder autoregressively
3. Generate output tokens
4. Detokenize to final text
5. Display the result

For now, it validates the full tokenization pipeline and .holo model loading.
