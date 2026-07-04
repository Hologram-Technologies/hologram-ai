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

## The honest bound

The advantage claim is bounded exactly as the UTQC proof bounds it: elision pays where
computation carries **repetition** — unchanged decode prefixes, shared constant cones,
duplicated tensors. It does not repeal arithmetic: a token's *new* suffix compute is
executed, and a materialized session still holds the live weight working set.

Accordingly, the dictionary tracks:

- `decode-elision` — a **measured** witness: across consecutive decode steps the session
  reports skipped (elided) kernel dispatches for the unchanged prefix cone. This is the
  k-scaling claim made mechanical, asserted only as measured.
- `memory-guard` — the resource envelope stays a *parametric function of the model's own
  configuration*, checked before the journey starts; the k-representation does not exempt
  the working set from the browser's address space.

Any stronger scaling statement (asymptotics across model families, wall-clock claims) is
an `open` row: measured, reported, never asserted.

[UOR-Atlas-UTQC]: https://github.com/afflom/UOR-Atlas-UTQC
[holospaces]: https://github.com/Hologram-Technologies/holospaces
