# 01 — k-representation: why browser scaling is not classical

The substrate's measured computational advantage is **content-addressed degeneracy**:
distinct computations that produce identical content collapse to the same κ-address, and
the substrate's residency check elides them. This is the same cache-collapse advantage
established by the [UOR-Atlas-UTQC] proof and inherited by [holospaces]; hologram-ai
applies it to AI model workloads.

## What the k-representation buys, concretely

A classical in-browser inference stack is bounded by classical resource assumptions:
the model file must be fetched monolithically, held in the JS/wasm heap, copied into a
runtime, and every decode step recomputes or explicitly caches attention state. Each of
those assumptions is a multiplier on the browser's hard 32-bit ceilings.

Operating on k-representations removes the assumptions structurally:

- **Storage dedup is identity, not policy.** A tensor's address *is* its content hash.
  Two shards, two revisions, or two models sharing a tensor share one OPFS blob. No
  eviction policy, no manifest bookkeeping — equality of κ is equality of content.
- **The archive is structure, not payload.** The compiled `.holo` binds weight constants
  to κ-labels instead of embedding bytes. Compiling, persisting, listing, and shipping
  models costs the *structure*, not the parameters.
- **Compute reuse is derivation, not machinery.** A node's output label is a
  deterministic function of its op signature and operand labels. Re-executing a decode
  step with an unchanged prefix re-derives resident labels and the walk skips them — the
  substrate's CE invariant. hologram-ai therefore ships **no KV-cache**: the reuse the
  KV-cache would hand-manage is what the addressing already provides.

## No classical residency assumption

The classical LLM-in-browser argument — "the model must fit in the tab's address
space" — is a residency assumption, not a law. Operating over k dissolves it, resource
by resource:

- **Network**: shards stream tensor-by-tensor and the content is discarded as retrieved;
  transfer is bounded by nothing but the wire. Unbounded model size.
- **Storage**: the κ-store lives in OPFS — disk, not heap. Its bound is the browser's
  storage quota, a REAL resource measured at runtime (`navigator.storage.estimate()`),
  never a constant. Unique content costs its entropy (the density proof deduplicates
  *identical* content and *identical* compute; it does not repeal Shannon), and identical
  tensors across shards, revisions, and models cost once.
- **Compile**: the archive is a weightless k-form — structure and κ-bindings. Its size is
  a function of the graph, not the parameters. Unbounded model size.
- **Runtime**: execution resolves κ → content **per stage**: the model is partitioned
  into stage archives (embedding, layer blocks, head), each materialized against the
  κ-store, executed, and released. The live working set is a **window** — a parametric
  function of the stage size and context, chosen from the environment — never the model.
  The 32-bit heap bounds the window; it does not bound the model.
- **Compute**: repetition is elided by κ-residency (unchanged decode prefixes, shared
  cones, duplicated tensors). A token's new suffix compute is executed; how fast is a
  *measurement*, reported per environment, never a rejection criterion.

Accordingly, the dictionary tracks:

- `staged-execution` — staged (windowed) execution is **equal** to monolithic execution:
  the stage partition covers the κ-map exactly, and the staged pipeline reproduces the
  monolithic logits, with peak weight residency bounded by the window.
- `decode-elision` — the measured elision witness: consecutive decode steps report
  skipped kernel dispatches for the unchanged prefix cone.
- `memory-guard` — the guard rejects **only genuine resource shortfall**: the κ-store
  bytes the model actually needs versus the measured OPFS quota. Working-set and
  throughput figures are surfaced as information, never used to refuse a model.

Any stronger scaling statement (asymptotics across model families, wall-clock claims) is
an `open` row: measured, reported, never asserted.

[UOR-Atlas-UTQC]: https://github.com/afflom/UOR-Atlas-UTQC
[holospaces]: https://github.com/Hologram-Technologies/holospaces
