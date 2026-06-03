> **✅ RESOLVED (hologram main `8d0398f`).** Hologram landed first-class
> `OpKind::Gather` (runtime int indices) and `OpKind::Cast` (numeric int↔float),
> plus named/shaped ports and open extension sections. hologram-ai adopted them:
> embedding now lowers to one `Gather` (int64 ids, no int→float cast) and `Cast`
> is a real numeric conversion — the dequant-widening below is moot. Kept for the
> record. See `specs/notes/generation-cli.md`.

# Upstream request to hologram: dequantize over i32/i64 (unblock LM token embedding)

**Against hologram main `c065c5e`** ("uint8 dequantization"). One small,
self-contained backend change unblocks real-model text generation in hologram-ai.

## Goal

Run published causal LMs (TinyLlama/Qwen2-class) end-to-end through hologram-ai's
generation CLI. Today they compile but fail at execute on the **token embedding**.

## Root cause (verified empirically on `c065c5e`)

A compiled `Gather`/embedding lowers to `OneHot(idx) · W`. The integer
`input_ids` are converted to f32 with the canonical `toᶠ³²` =
`Dequantize(scale = 1, zero_point = 0)` (same converter the dequant primitive
uses). The backend `dequantize` kernel only accepts `quant_dtype ∈ {i4, i8, u8}`;
for **i32/i64 it returns early**, so the forward pass fails at dispatch.

Probe (synthetic successor-LM, `Gather(W[V,V], input_ids[1,S], axis=0)`):

| `input_ids` dtype | compile | execute |
|---|---|---|
| **i8**  | ok | ok — correct rows |
| **i32** | ok | **FAIL: `BackendError`** |
| **i64** | ok | **FAIL: `BackendError`** |

It is **purely a backend-kernel limitation**: compile, completeness, and autodiff
all pass. hologram-ai compiles via `hologram_compiler::compile`, which does **not**
run `append_backward` — so no compiler/graph/autodiff change is needed.

Real LMs use **int64 `input_ids`** with vocab ≫ 256, so i8/u8 can't represent the
ids — i32/i64 support is required. (hologram-ai's `tests/generation_synthetic.rs`
is currently pinned to an i8 vocab ≤ 127 for this reason.)

## Exact locations (`c065c5e`)

`crates/hologram-backend/src/cpu/kernels.rs`
- `dequantize` (fn @ 167) — **the one the embedding uses**:
  - `in_bytes_needed` match @ 170–173: `_ => Err(SlotOutOfRange)` rejects i32/i64.
  - `dequant_at` value match @ ~231–248: reads `inp[i]` (1 byte/elem for i8/u8,
    `inp[i/2]` nibble for i4) — needs **width-aware** reads for i32/i64.
- `DequantActivation` densify path: `in_bytes_needed` @ 300–302; value match @ 321/336.

`crates/hologram-backend/src/cpu/float_kernels.rs`
- `matmul_dequant_float` (@ 311): `in_bytes` @ 327–332 (`"quant_dtype must be
  i8/u8/i4"`); value match @ 372–376.

## Minimal fix

Only the **standalone `dequantize` kernel** must change for embedding (weights
stay i8/i4 through the matmul/activation fused paths). In that kernel:

```rust
// in_bytes_needed:
DTYPE_I32 => n * 4,
DTYPE_I64 => n * 8,

// dequant_at(i) — index by i*width, not inp[i]:
DTYPE_I32 => i32::from_le_bytes(inp[i*4 .. i*4+4].try_into().unwrap()),
DTYPE_I64 => i64::from_le_bytes(inp[i*8 .. i*8+8].try_into().unwrap()) as i32,
```

With `scale = 1, zp = 0` the result is `q as f32` — **exact for |id| < 2²⁴**,
which covers every real vocab (≤ ~256k); i64→i32 is safe for vocab ≪ 2³¹.

Widening `matmul_dequant_float` and the `DequantActivation` densify the same way
is **optional** (consistency only) — embedding doesn't need them.

## Acceptance

- Backend unit test: `dequantize` over an i32 and an i64 buffer with
  `scale = 1, zp = 0` returns each integer as its f32 value.
- End-to-end: a compiled `Gather`/embedding over `input_ids[1,S]` int64 against a
  `[V,D]` table executes and returns the correct rows. Equivalently, flipping
  hologram-ai's `tests/generation_synthetic.rs` synthetic LM from `INT8` to
  `INT64` then passes.

## Alternatives (if preferred over widening dequant)

1. A dedicated **numeric `Cast`(int→float) kernel**. Today `Cast` is a
   byte-reinterpreting reshape (no numeric conversion), so it can't serve here.
2. A first-class **runtime-indexed `Gather`/`Embedding` kernel** — avoids the
   `one_hot · matmul` entirely and is far more efficient at large vocab
   (`O(S·D)` vs `O(S·V·D)`). Larger change, but the right long-term primitive.

The minimal dequant widening requires **zero** hologram-ai changes; options 1–2
would let hologram-ai drop the `one_hot` workaround later. See
`specs/notes/generation-cli.md` for the hologram-ai side.
