# CLI -- `hologram-ai`

`hologram-ai` is the native compiler/runner surface for this repository. It
owns model import, optimization, validation, and lowering, then hands the final
graph to `hologram-compiler` to emit a `.holo` archive.

## Commands

### `compile`

Compile an ONNX model into a `.holo` archive:

```bash
cargo run -p hologram-ai -- compile \
  --model models/bert-base-uncased/model.onnx \
  --output /tmp/out \
  --seq-len 8
```

Key flags:
- `--name <stem>` overrides the archive filename stem.
- `--seq-len <n>` fixes symbolic sequence dimensions at compile time.
- `--quantize none|int8|int4` selects compile-time weight quantization.
- `--spatial-scale <n>` downscales 4-D vision inputs before lowering.

### `run`

Run a compiled `.holo` archive:

```bash
cargo run -p hologram-ai -- run /tmp/out/model.holo --fill zeros
```

Use `--input INDEX:HEX` or `--input-file INDEX:PATH` for exact byte-level
inputs, or `--fill zeros|ones|N` to synthesize missing inputs.

For causal-LM generation, `run` switches into text mode when `--prompt` is
present:

```bash
cargo run -p hologram-ai -- run /tmp/tinyllama/model.holo \
  --prompt "Tell me a short joke." \
  --max-tokens 32 \
  --temperature 0
```

When available, `run --prompt` auto-discovers a HuggingFace chat template in
this order:
- `--chat-template <file>` explicit override
- `chat_template.jinja` embedded in the compiled `.holo`
- companion `chat_template.jinja` beside the model / ONNX source
- `chat_template` or `default_chat_template` in `tokenizer_config.json`

Use `--stats` to print latency metrics to stderr during either raw execution or
generation. In generation mode this reports:
- prompt token count and generated token count
- prompt encode time
- prefill session-preparation time
- prefill forward time
- time to first token
- decode throughput after the first token
- total wall-clock time

Use `--decode-top-k <K>` in generation mode to print the first-step top-k token
candidates from the logits row before sampling. This is useful for debugging
prompt formatting and checking whether the model is producing sane next-token
probabilities.

Example:

```bash
cargo run -p hologram-ai -- run /tmp/tinyllama/model.holo \
  --prompt "Tell me a short joke." \
  --max-tokens 1 \
  --temperature 0 \
  --decode-top-k 5 \
  --stats
```

### `download`

Fetch a model repository into `models/`:

```bash
cargo run -p hologram-ai -- download bert-base-uncased
```

Use this instead of external model download tools; it preserves the repository's
expected layout (`model.onnx`, `tokenizer.json`, `config.json`, ...).
