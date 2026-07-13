# Upstream request: a `valid_len` scalar for the fused decode attention (κ119)

**Status: OPEN (filed 2026-07-13).**
**To:** the hologram substrate (Hologram-Technologies/hologram, v0.9.0 = `22b0ce1`).
**Precedent:** `upstream-request-decode-attention.md` — the finding → request →
v0.9.0 fix loop this note repeats for the last per-token O(bucket) input.

## The finding

With the v0.9.0 fused decode path adopted (ADR-0019), every per-token input to
the step walk is O(1) — token id, rope rows, ring position — EXCEPT the
additive mask: `[chunk, bucket + chunk]` f32, rebuilt/re-interned every token.
Its information content is:

- one scalar that changes per token: the realized length (`col < pos` visible),
  which the walk ALREADY receives as the 4-byte `decode_pos` operand; and
- a static within-chunk causal triangle (`col − bucket ≤ i`), constant for a
  compiled chunk.

At a 32K bucket that is 128 KiB of mask per token (× stages on the staged
pipeline), BLAKE3-interned each walk so the pool can address it — the same
"resident value re-hashed at the byte boundary" anti-pattern the κ120 KvCacheWrite
move eliminated for the caches (measured there at 28/110/442 ms/tok @2K/8K/32K on
1.5B; the mask is one buffer rather than 2·L, so ~1/(2L) of that tax, growing
with context like everything else that is O(bucket) per token).

We shipped the our-side half: the engine's mask buffer is persistent and updated
incrementally (O(Δpos) per pass instead of an O(bucket) rebuild — `decode.rs`
`refresh_buffers`). The intern hash is the remaining O(bucket) per-token cost,
and only the substrate can remove it: the buffer must be re-addressed each walk
as long as it is an input value.

## The ask

Let the six-input decode attention derive the realized-length visibility law
from a scalar instead of a mask value:

- `Attention(q, k_past, v_past, k_new, v_new, valid_len)` — `valid_len` INT32
  `[1]`, the kernel erases `col ≥ valid_len` in the past region and applies the
  causal triangle in the new region (both laws it already implements — today
  they arrive pre-encoded in the mask bytes); or equivalently an
  `AttentionAttrs` flag declaring "mask = realized-length + causal" so the mask
  operand may be omitted.

This is the structure/content split of the invariance ladder applied once more:
the visibility LAW is structure (compiled once), the realized length is CONTENT
(4 bytes per token). The arbitrary-mask form should remain for genuinely
arbitrary masks; decode never needs one.

## Why it matters

It is the last O(bucket) hash on the per-token path. After it, a decode step's
input traffic is O(1) and the walk cost is the kernel's own — the substrate's
`M = 1` measurement becomes the whole story at any context length.
