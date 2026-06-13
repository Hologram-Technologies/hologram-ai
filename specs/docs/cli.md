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

### `export-fixture`

Compile a model and emit a deterministic fixture bundle for `holospaces`. The
bundle contains:
- the compiled `.holo`
- deterministic input blobs for a known preset
- expected output blobs and κ-labels
- `manifest.json` describing port order, dtype tags, shapes, and file layout

The emitted `.holo` now also embeds the same fixture manifest, typed inputs,
and expected outputs in archive extension sections. That makes the archive
self-contained: the external directory is still written for `holospaces` and
other file-oriented consumers, but the archive itself carries the witness too.

Example for the checked-in BERT model:

```bash
cargo run -p hologram-ai -- export-fixture \
  --model models/bert-base-uncased/model.onnx \
  --output /tmp/bert-holospaces \
  --preset bert-base-uncased \
  --seq-len 8
```

Available BERT presets:
- `bert-base-uncased`: canonical non-masked token sequence used by the conformance suite
- `bert-base-uncased-masked`: the same sentence with the content token replaced by `[MASK]` for masked-language-model inspection

`bert-base-uncased` synthesizes:
- `input_ids = [101, 2023, 2003, 1037, 3231, 102, 0, 0]`
- `attention_mask = [1, 1, 1, 1, 1, 1, 0, 0]`
- `token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0]`

`bert-base-uncased-masked` synthesizes:
- `input_ids = [101, 2023, 2003, 1037, 103, 102, 0, 0]`
- `attention_mask = [1, 1, 1, 1, 1, 1, 0, 0]`
- `token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0]`

This is the recommended path when the next consumer is `holospaces`: compile in
`hologram-ai`, then hand the emitted `.holo` and fixture inputs/expected κ to
the holospaces engine.

### `run-fixture`

Run the deterministic fixture embedded in a compiled `.holo` archive:

```bash
cargo run -p hologram-ai -- run-fixture /tmp/bert-holospaces/model.holo
```

This command:
- reads the embedded `fixture/manifest.json`
- loads the embedded typed inputs
- executes the archive on those inputs
- verifies the resulting output bytes and κ-labels against the embedded witness

Use `--verbose` to print a short typed preview of each verified output.

Use `--decode-top-k <K>` to inspect top-k token candidates at explicit
positions, or `--masked-top-k <K>` to inspect only `[MASK]` positions from a
masked-language-model fixture:

```bash
cargo run -p hologram-ai -- export-fixture \
  --model models/bert-base-uncased/model.onnx \
  --output /tmp/bert-holospaces-masked \
  --preset bert-base-uncased-masked \
  --seq-len 8

cargo run -p hologram-ai -- run-fixture \
  /tmp/bert-holospaces-masked/model.holo \
  --masked-top-k 5
```

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
