# Text-generation CLI (`hologram-ai run --prompt`)

Autoregressive text generation over a compiled `.holo` causal LM. Implemented in
[`crates/hologram-ai/src/commands/generate.rs`](../../crates/hologram-ai/src/commands/generate.rs)
and wired into `run` in
[`run_cmd.rs`](../../crates/hologram-ai/src/commands/run_cmd.rs).

## Self-describing archive (hologram main `8d0398f`)

The `.holo` now has **open extension sections** and **named/shaped ports**, so
the tokenizer and port identities live *in the archive* — no run-time flags
needed for a compiled model:

- `compile` bakes `tokenizer.json` into the archive as an extension,
  **canonicalized via uor-addr** (`json::canonicalize`, JCS-RFC8785 + NFC) and
  stored with its κ-label (`tokenizer.kappa`) for an integrity check on load.
- `run --prompt` reads the tokenizer from the archive (`session.extension`),
  verifies its content address, and binds `input_ids` / `logits` **by name**.

```
# compiled model is self-describing:
hologram-ai run <archive.holo> --prompt "<text>" \
  [--prompt-template "<…{prompt}…>"] [--max-tokens N] [--temperature T] \
  [--top-k K] [--stop "<s>" …] [--eos ID] [--tokenizer <file>  # optional override]
```

When `--prompt` is absent, `run` is a raw forward pass that executes **any**
compiled model:

```
hologram-ai run <archive.holo> [--input I:HEX] [--input-file I:PATH] [--fill zeros|ones|N] [--verbose]
```

On load it prints each input port (`index: dtype × element_count = bytes`).
Inputs not given explicitly are synthesized from `--fill` (`zeros` is valid for
every dtype; `ones`/numeric encode per the port dtype) — so an arbitrary
multi-input model runs with one command. Explicit inputs are size-checked
against their port; missing inputs without `--fill` are a clear error (no silent
zero-fill). Outputs report `dtype × element_count` and, with `--verbose`, a typed
value preview (f32/f64/i32/i64) or hex. Verified on multi-input / multi-output /
mixed-dtype models in `tests/run_arbitrary_models.rs`.

## LM contract (resolved by name)

- `input_ids` — integer port, shape `[1, seq_len]`; bound by name (no positional
  guess). `attention_mask` / `position_ids`, if present, are synthesized each
  step (1s / `0..cur_len`) at their named ports — multi-input causal LMs work.
- `logits` — f32 port, `[1, seq_len, vocab]`; bound by name.

A model with no `input_ids` / `logits` port (not a causal LM) is reported as a
surfaced validation error — never fabricated output.

## Loop

`encode(template(prompt))` → for each step: take the last `seq_len`-token window,
forward, read the logit row at the last real position, sample (greedy argmax at
`temperature 0`, else softmax + optional top-k via a self-contained SplitMix64),
detokenize the delta and stream it, stop at eos / a `--stop` string / `max_tokens`.
No KV-cache: repeated-prefix work is elided structurally by κ-label inside the
session.

## Embedding & casts (first-class ops)

Embedding lowers to a single **`OpKind::Gather`**`(table, input_ids, axis=0)` —
the int64 ids stay integer, no one-hot, no int→float cast. Numeric dtype
conversions lower to **`OpKind::Cast`** (real ONNX-semantics conversion). Both
are hologram primitives (main `8d0398f`); the prior `OneHot·MatMul` +
`Dequantize`-as-cast workarounds are deleted.

Real int64 token embedding is therefore unblocked: the synthetic LM test runs
**int64 ids at vocab 200** end to end (`tests/generation_synthetic.rs`). Casting
*to* f64 fails loud in the engine (compute domain is f16/bf16/f32) — hologram's
dtype policy, not an hologram-ai limit.
