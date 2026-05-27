# Text-generation CLI (`hologram-ai run --prompt`)

Autoregressive text generation over a compiled `.holo` causal LM. Implemented in
[`crates/hologram-ai/src/commands/generate.rs`](../../crates/hologram-ai/src/commands/generate.rs)
and wired into `run` in
[`run_cmd.rs`](../../crates/hologram-ai/src/commands/run_cmd.rs).

## Why run-time flags (no archive persistence)

The `.holo` format has a **closed section set** (`SectionKind` is a `#[repr(u8)]`
enum, kinds 1–13 in `hologram-archive/src/format.rs`); `HoloWriter` exposes only
typed `set_*` methods and has no custom-section append. The `SECTION_TOKENIZER`
(0x6801) / `SECTION_LLM_META` (0x6811) types in `hologram-ai-common` are u32
codes the current format cannot store — vestigial from an older archive layout.
`compile_with_sections` already silently drops its `ArchiveSections` argument for
this reason. `PortDescriptor` also carries no tensor *name* (only slot /
element_count / dtype), so port roles are positional-by-convention.

So the tokenizer and generation metadata are supplied **at run time** via CLI
flags rather than baked into the archive:

```
hologram-ai run <archive.holo> \
  --prompt "<text>" \
  --tokenizer <tokenizer.json> \
  --prompt-template "<…{prompt}…>" \
  --max-tokens N --temperature T --top-k K --stop "<s>" [--stop "<s2>" …] [--eos ID]
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

## LM contract (resolved positionally)

- input port 0 — `input_ids`, shape `[1, seq_len]`, integer dtype; element count
  is the fixed sequence length baked at compile time.
- output port 0 — `logits`, shape `[1, seq_len, vocab]`, f32; `element_count /
  seq_len` is the vocabulary size.

A model that doesn't match (≠1 input/output, non-integer ids, non-f32 logits,
logit count not divisible by seq_len) is rejected with a clear error — no guess.

## Loop

`encode(template(prompt))` → for each step: take the last `seq_len`-token window,
forward, read the logit row at the last real position, sample (greedy argmax at
`temperature 0`, else softmax + optional top-k via a self-contained SplitMix64),
detokenize the delta and stream it, stop at eos / a `--stop` string / `max_tokens`.
No KV-cache: repeated-prefix work is elided structurally by κ-label inside the
session.

## Embedding lowering fix (in-repo)

`builder.rs::one_hot` (the `OneHot(idx)·W` realization of Gather/Embedding) was
emitting an **i64 `Equal` with implicit broadcast** — the backend binary kernels
are strict element-wise (no broadcast) and the byte-domain `Equal` can't compare
multi-byte integers, so it failed at dispatch. Fixed to:

1. convert indices to f32 via the canonical `Dequantize(scale 1)` `toᶠ³²`,
2. reshape to `[rows,1]` and the `iota` const to `[1,depth]` (matching rank so
   `expand_plan` accepts them), `Expand` both to `[rows,depth]`,
3. compare with a strict-elementwise float `Equal` → 0/1 mask.

This makes runtime-index embedding execute correctly (verified end-to-end on a
synthetic successor-LM in `tests/generation_synthetic.rs`).

## Upstream dependency for real (int64) models — OPEN

The backend's only numeric int→float converter is the dequant kernel
(`(q − z)·s`), which supports the **quantum widths only** (i8 / i4 / u8). Real
LMs use **int64 `input_ids`** with vocab ≫ 256, so the embedding `toᶠ³²` cannot
run today: `Dequantize` over i32/i64 fails completeness/backward and has no
kernel path. The synthetic test therefore uses an **i8 vocab (≤127)**.

To run published TinyLlama/Qwen2-class models end-to-end, hologram needs one of:

- **(preferred)** widen the dequant/`toᶠ³²` path to accept i32/i64 integer
  inputs (read the integer, convert to f32) with completeness + backward, or
- a dedicated runtime-indexed Gather/Embedding kernel, or
- a general integer→float `Cast` kernel (today `Cast` is a byte-reinterpreting
  reshape, not a numeric convert).

Until then, real-model generation is infra-blocked at the token embedding; the
CLI, loop, sampling, and lowering are complete and verified at i8 vocab.
