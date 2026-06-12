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

Example for the checked-in BERT model:

```bash
cargo run -p hologram-ai -- export-fixture \
  --model models/bert-base-uncased/model.onnx \
  --output /tmp/bert-holospaces \
  --preset bert-base-uncased \
  --seq-len 8
```

The current preset is `bert-base-uncased`, which synthesizes the canonical
token sequence used by the conformance suite:
- `input_ids = [101, 2023, 2003, 1037, 3231, 102, 0, 0]`
- `attention_mask = [1, 1, 1, 1, 1, 1, 0, 0]`
- `token_type_ids = [0, 0, 0, 0, 0, 0, 0, 0]`

This is the recommended path when the next consumer is `holospaces`: compile in
`hologram-ai`, then hand the emitted `.holo` and fixture inputs/expected κ to
the holospaces engine.

### `run`

Run a compiled `.holo` archive:

```bash
cargo run -p hologram-ai -- run /tmp/out/model.holo --fill zeros
```

Use `--input INDEX:HEX` or `--input-file INDEX:PATH` for exact byte-level
inputs, or `--fill zeros|ones|N` to synthesize missing inputs.

### `download`

Fetch a model repository into `models/`:

```bash
cargo run -p hologram-ai -- download bert-base-uncased
```

Use this instead of external model download tools; it preserves the repository's
expected layout (`model.onnx`, `tokenizer.json`, `config.json`, ...).
