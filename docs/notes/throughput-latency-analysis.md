# Throughput & latency analysis — where the browser decode ceiling actually is

**Date:** 2026-07-11. **Scope:** ADR-0018 follow-on. **Substrate:** v0.8.2 =
`f031e8b` (see the pin note below). **Instrument:** `apps/web/scripts/pool-bench.rs`
(`pnpm bench:pool`), extended to decompose the decode step; wasmtime /
`wasm32-wasip1-threads`, 4 participants (codespace's 4 physical cores).

This is the finding→plan record for the request "drastically increase throughput
and eliminate latency so it runs arbitrary models at a modern-chatbot feel." The
headline: **the GEMV worker pool (ADR-0018, shipped) removes the SHORT-context
bottleneck, but the pool speedup collapses as context grows because attention,
the KV-cache recopy, and softmax are serial. Sustaining throughput at the context
lengths users actually chat at needs work the GEMV pool cannot do.**

## 0. Correctness prerequisite (fixed first)

The branch pinned the substrate by `tag = "v0.8.2"`. The local git db was stale,
so cargo mis-resolved that tag to **`18f553d`** — an ancestor of v0.8.1 with **no
prefill pooling** — and, because `holospaces` independently pins an old hologram
at the same `18f553d`, cargo unified our whole decode path down onto it. Net: the
pushed lockfile built an *older-than-main* substrate while the docs claimed
v0.8.2. Fixed by pinning the **explicit rev `f031e8b`** (the real `v0.8.2^{}`,
== substrate `main` HEAD; v0.8.1 = `0120c94` = what our `main` still ships). An
explicit rev cannot mis-resolve against a stale db. Native check + threaded wasm
build both green on the corrected pin; the benchmark below runs against `f031e8b`.

## 1. The GEMV pool win is real — at short context (View 1)

Weight-GEMV decode + prefill/TTFT, serial → pooled (4 participants), real HF
configs, f031e8b:

| model | ~GB | decode tok/s (s→p) | sp | TTFT ms (s→p) | sp |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5 | 30 → 81 | 2.7× | 2427 → 685 | 3.5× |
| Qwen2.5-1.5B | 1.5 | 8.8 → 29 | 3.3× | 8880 → 2321 | 3.8× |
| Qwen2.5-3B | 3.1 | 5.4 → 17 | 3.2× | 19042 → 5216 | 3.7× |
| Qwen2.5-7B | 7.1 | 2.3 → 5.8 | 2.5× | 56216 → 15673 | 3.6× |

**TTFT pools 3.2–4.0× (v0.8.2 prefill pooling)** — the real end-to-end win,
GEMM-bound, not diluted. Decode GEMV pools 2.4–3.3×. These are the *weight*
projections only — a decode step is more than its weight GEMVs.

## 2. The ceiling: the pool speedup ERODES with context (View 2)

Decomposing the real per-token step (µs/token) at chat context lengths L. The
weight GEMV pools; **attention (QK^T/P·V), softmax, and the KV-cache recopy do
not** (attention only crosses the f32 pool threshold at very long L). Attention
*compute* ∝ layers·hidden·L, so it overtakes the fixed weight GEMV as L grows:

Qwen2.5-1.5B (pooled GEMV = 34 ms/token, constant in L):

| L | attn | softmax | KV-recopy | step (ms) | tok/s | pool sp |
|---|---|---|---|---|---|---|
| 128 | 0.9 | 0.2 | 2.7 | 38 | 26 | **3.09×** |
| 2048 | 15 | 3.4 | 3.1 | 56 | 18 | 2.42× |
| 8192 | 62 | 14 | 19 | 129 | 7.7 | **1.66×** |
| 32768 | 250 | 55 | **440** | 778 | 1.3 | **1.14×** |

Qwen2.5-7B tells the same story: 2.45× @128 → 1.46× @8192 → 1.15× @32768, tok/s
5.7 → 1.8 → 0.5. Every model does. **At the long contexts a modern chatbot needs,
the serial attention + KV-recopy dominate and the GEMV pool is nearly idle.**

Two specific serial sinks, from the substrate decode map (f031e8b):

- **The KV cache is recopied every token.** `DecodeRewrite` explodes decode
  attention into explicit nodes; `Concat(past∥new)` + `Transpose` read and
  rewrite the *entire* K and V cache each step — O(L) memory traffic per token,
  hundreds of ms at 32K. A resident, append-in-place KV cache removes it
  outright. (Same root cause feeds a second cost: because `past_k`/`past_v` are
  regenerated graph inputs, the byte `execute` path BLAKE3-re-hashes the whole KV
  cache every step; a resident KV or `execute_addressed` avoids it too.)
- **Attention math is serial.** QK^T/softmax/P·V run single-threaded below the
  pool threshold — the compute half of the long-context ceiling.

Sampling, separately: greedy `argmax` is ~0.1 ms/token, but temperature/top-k
does a **full O(vocab·log vocab) sort = 5.4–6.1 ms/token** (150K vocab) — ~14% of
a short-context step, pure removable waste (a partial top-k is ~argmax cost).

κ weight resolution is NOT a per-token cost (weights are pinned resident at load,
bound by pointer) — ruled out as a suspect.

## 3. Levers, ranked by measured impact

**Upstream (substrate; read-only here — request + measure, as prefill pooling was):**

- **A. Resident / append-in-place KV cache.** Kills the per-token KV recopy
  (440 ms @32K on 1.5B) AND the per-token KV re-hash — the single largest
  long-context cost. Highest-impact fix for "don't cap the input."
- **B. Pool the attention block (QK^T / softmax / P·V).** The compute half of the
  long-context ceiling; the natural successor to the prefill-pooling request.

**Our-side (ship now):**

- **C. Speculative decode default-on (prompt-lookup).** Already built
  (`speculative.rs`), byte-identical, "never worse than not drafting," but OFF by
  default (gated on the `hologram_speculative` localStorage knob; no catalogue
  draft pairings ship). It verifies K draft tokens in ONE M=K pass — which v0.8.2
  now **pools** — so it amortizes the *whole* step (weight GEMV **and** attention
  **and** the KV recopy) over accepted tokens. This is the one our-side lever that
  attacks the long-context ceiling directly. Ceiling = mean-acceptance×; ≈neutral
  on novel text. Needs an acceptance measurement on realistic chat to set the
  default; low-risk because worst case is ≈one plain step.
- **D. int4 weights.** The kernel exists (`matmul_i4_pc_omajor`) but int4 is
  de-advertised. int4 halves bytes/token → ~2× the *short-context* (GEMV-bound)
  decode: 1.5B ~26 → ~45 tok/s @128. Quality-gated re-enable (advertise the tier
  where perplexity holds).
- **E. Sampling: partial top-k instead of the full sort.** Removes 5–6 ms/token
  under temperature sampling. Same emitted token — pure waste removal.
- **F. Eager pool prewarm.** Spawn the pool during model load, off the first
  turn's TTFT.

## 4. Honest framing

The GEMV pool was the right first move and it delivers at short context and on
TTFT. But "optimal performance a user expects from a modern chatbot" at real
context lengths is gated by the serial attention + KV path, not the weight GEMV.
The biggest wins (A, B) are substrate changes; the biggest *our-side* win (C)
already exists and is dormant. Real end-to-end tok/s must be confirmed on the
deploy (the 4-core codespace is too contended for absolute browser numbers) —
these are wasmtime component measurements, representative of scaling, not of one
particular machine's wall-clock.

Related: `upstream-request-prefill-pooling.md` (the prior finding→request→fix loop
this mirrors), ADR-0018.
