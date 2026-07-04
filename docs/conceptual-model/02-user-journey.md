# 02 — The user journey (normative)

The application's contract is a single journey. Its stages are the suites of the BDD
tree ([`features/suites/`](../../features/suites/)); every stage is a dictionary row with
a feature, a witness, and an oracle. The journey is verified in two forms:

- **hermetic** — against a committed tiny-model fixture served locally, with committed
  reference outputs, in headless Chromium. Gating on every push.
- **live** — against a real HuggingFace model at a pinned revision, in headless
  Chromium. Gating on the scheduled architecture matrix.

## Stage S1 — Download

Given any HuggingFace repository id:

1. Resolve the repo's file manifest via the Hub API (`api/models/{id}`), with retry and
   explicit failure surfacing.
2. Classify files: safetensors shards (weights), companions (`config.json`,
   `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`) — each matched by
   its exact basename, never by suffix.
3. **Preflight — the journey validates the model before any shard byte moves:**
   a. `config.json` must name a supported architecture family and carry the family's
      required keys; an unsupported family or malformed config rejects the journey
      naming the family/key, with zero shard bytes transferred.
   b. The tensor manifest is read from the shards' safetensors *headers alone*
      (ranged requests — kilobytes, not weights), and the parametric graph is built
      from config + manifest. A manifest the family cannot realize rejects here.
   c. The resource estimate is derived from the validated config and manifest sizes;
      exceeding the environment budget (a parametric function of the environment,
      never a per-model constant) rejects with the estimate. A config that cannot
      produce an estimate is a preflight failure, never a silent pass.
4. Stream each shard through the persistent download worker: walk the tensor byte
   ranges from the already-parsed header, hash each tensor incrementally, persist it to
   OPFS as `tensors/{κ}.bin`, and **discard the content as it is retrieved** — the
   k-representation is what remains. Peak transient memory is bounded by one tensor,
   never a shard or the model.
5. Persist companions under the model's directory.

Because the graph was built and validated in preflight, the post-stream step is
mechanical — bind the streamed κs into the already-validated graph and emit the k-form
archive. It cannot fail on model validity; there is no separate failure-prone "compile"
stage after the transfer.

## Stage S2 — Compile

1. Build the parametric decoder graph **solely** from `config.json` and the tensor
   manifest (names, shapes, dtypes, κ). The architecture family is selected from
   `config.architectures` via the family registry; an unsupported family fails loud
   naming the family.
2. Compile to a weightless κ-form `.holo`: graph, schedule, ports, tokenizer/generation
   extensions, and the `holospaces.kappa_map` binding each weight constant to its κ.
3. Persist the archive to OPFS under the model directory.

## Stage S3 — Run

1. Materialize: resolve every κ-map entry against the OPFS κ-store; verify each buffer
   re-hashes to its κ; patch the archive's constants into an executable form. A missing
   or corrupt κ aborts with the label.
2. Load the materialized archive into an inference session and execute the forward pass.
3. Correctness authority: natively, the same materialized pipeline must reproduce ONNX
   Runtime logits for the same model within tolerance; in the browser, the hermetic
   fixture must reproduce its committed reference outputs exactly.

## Stage S4 — Chat (the three-message handshake)

Given a materialized session and the model's own chat template and stop conditions:

1. The user sends message 1; the assistant streams a completion.
2. The user sends message 2; the prompt now carries the full transcript; the assistant
   streams a completion. Consecutive decode steps must report elided prefix compute
   (`decode-elision`).
3. The user sends message 3; the assistant streams a completion.

The handshake passes when all three assistant turns complete without error, each turn's
output is non-empty, respects stop conditions, and — on the hermetic fixture — matches
the committed reference transcript deterministically (temperature 0, fixed seed).

## Failure discipline

Every stage failure is surfaced to the user with the stage, the reason, and the
offending identity (repo id, family name, κ, budget figure). No stage publishes partial
state as success; no stage silently falls back.
