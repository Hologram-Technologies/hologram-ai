# T5 Quick Start

**Configuration:** This guide uses [t5.toml](./t5.toml) - a unified config file containing all settings for both compilation and execution.

**Full documentation:** See [README.md](./README.md) for detailed guide, examples, and troubleshooting.

## Step 0: Get T5 ONNX Models (Choose One)

### Option A: Download Pre-Converted ONNX from Hugging Face

If an ONNX-converted version exists on Hugging Face:

```bash
# Download T5-small ONNX models from Hugging Face
hologram-ai download "<onnx-model-repo-id>" \
  --output /workspace/models/t5-small

# For gated models, set your HF token first:
export HF_TOKEN=hf_your_token_here
hologram-ai download "gated-model-id" --output /workspace/models/t5-small
```

**Note:** The `download` command only works with repositories that already have ONNX models. If the repository only has PyTorch models, use Option B below.

### Option B: Convert from PyTorch to ONNX (Manual)

If no ONNX version exists, convert using Python:

```bash
pip install transformers torch onnx

python3 << 'EOF'
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Export encoder
encoder_dummy = torch.randint(0, 32128, (1, 128))
torch.onnx.export(
    model.get_encoder(), encoder_dummy,
    "/workspace/models/t5-small/encoder_model.onnx",
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"},
                  "last_hidden_state": {0: "batch", 1: "seq_len"}}
)

# Export decoder
decoder_dummy = torch.randint(0, 32128, (1, 1))
hidden_states = torch.randn(1, 128, 512)
torch.onnx.export(
    model.get_decoder(),
    (decoder_dummy, hidden_states),
    "/workspace/models/t5-small/decoder_model.onnx",
    input_names=["input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"},
                  "encoder_hidden_states": {0: "batch", 1: "encoder_seq_len"},
                  "logits": {0: "batch", 1: "seq_len"}}
)

# Save tokenizer
tokenizer.save_pretrained("/workspace/models/t5-small")
print("✓ T5 models exported to /workspace/models/t5-small/")
EOF
```

## Step 1: Compile Models to .holo Format

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

**Alternative:** Use config-driven compilation (if supported by your version):
```bash
hologram-ai compile-pipeline --config examples/T5/t5.toml --output t5-pipeline.holom
```

## Step 2: Run Text Generation

```bash
# Use default prompt from config
hologram-ai run --config examples/T5/t5.toml

# Custom translation
hologram-ai run --config examples/T5/t5.toml \
  --prompt "translate English to French: Hello, how are you?"

# Custom summarization
hologram-ai run --config examples/T5/t5.toml \
  --prompt "summarize: Large language models are neural networks..."

# Generate joke
hologram-ai run --config examples/T5/t5.toml \
  --prompt "Tell me a joke"
```

## What's Happening

### Compilation
1. Reads ONNX model (encoder_model.onnx / decoder_model.onnx)
2. Translates 70+ ONNX operations to Hologram IR
3. Compiles IR to optimized BackendPlan (SIMD, fusion, parallel ops)
4. Serializes to .holo format with auto-selected weight strategy:
   - < 100MB: Embedded weights (fast loading)
   - 100MB-1GB: Page-aligned weights (memory-mapped)
   - > 1GB: External .weights file

### Execution
1. Tokenizes input text with SentencePiece
2. Runs encoder: `input_ids` → `encoder_hidden_states`
3. Runs decoder autoregressively: generates tokens one by one
4. Decodes tokens back to text

## Configuration File

- **[t5.toml](./t5.toml)** - Unified configuration containing:
  - Tokenizer settings (shared by compilation and execution)
  - Model paths (ONNX source and compiled output)
  - Compilation settings (weight strategy, optimizations)
  - Generation settings (prompt, max tokens, temperature, sampling)
  - Pipeline stages (execution flow)

Edit this file to customize behavior. The unified config eliminates duplication between compilation and execution settings.

## Troubleshooting

### "File not found" during compilation
```bash
# Check that ONNX files exist
ls -lh /workspace/models/t5-small/*.onnx
```

### "File not found" during execution
```bash
# Check that compiled models exist
ls -lh /workspace/models/t5-small/compiled/*.holo

# If missing, run compilation commands above
```

### Generation produces gibberish
1. Verify correct tokenizer: `vocab_path` in `t5.toml` under `[tokenizer]`
2. Try greedy decoding: Set `do_sample = false` in `[generation]` section
3. Reduce temperature: Set `temperature = 0.7` in `[generation]` section

### Out of memory
1. The compiler auto-selects weight strategy
2. For very large models, it uses external weights automatically
3. Check available RAM with `free -h`

## Example Prompts for T5

T5 is a text-to-text model trained on multiple tasks:

**Translation:**
```
translate English to French: The weather is beautiful today
translate English to German: I love programming
translate French to English: Bonjour, comment allez-vous?
```

**Summarization:**
```
summarize: [long article text]
```

**Question Answering:**
```
question: What is the capital of France? context: France is a country in Europe. Its capital is Paris.
```

**Grammar Correction:**
```
grammar: She go to school yesterday
```

**General Text Generation:**
```
Tell me a joke
Write a poem about the ocean
Explain quantum computing in simple terms
```

## Performance Notes

- **First run**: May take longer (JIT compilation, cache warming)
- **Subsequent runs**: Much faster (cached kernels)
- **Batch size**: Currently 1 (single prompt at a time)
- **Speed**: ~10-50 tokens/sec on CPU (depends on hardware)

For production use, consider:
- Compiling with `weight_strategy = "page_aligned"` for faster loading
- Using GPU backend (when available)
- Batching multiple prompts together

## Next Steps

- See [README.md](./README.md) for full documentation
- Explore configuration options in [t5.toml](./t5.toml)
- Check troubleshooting section in the full README
