# hologram-ai

**An in-browser AI application on the hologram substrate**: it downloads models
from HuggingFace, compiles them into content-addressed (κ-form) `.holo`
archives, and runs them — entirely client-side, with the k-representation
discipline inherited from [holospaces]. The same pipeline is available natively
(CLI + library), where its external-authority conformance is anchored.

This repository is **docs-as-code / BDD-driven / V&V-gated**, following the
[UOR-Atlas-UTQC] methodology: the conceptual model is authored once — as prose
*and* as typed data — every feature begins as a Gherkin definition, and every
claim is validated against an authority this repository did not author.

## Where things are

| Path | Role |
|---|---|
| [`docs/conceptual-model/`](docs/conceptual-model/) | the conceptual authority (prose): what the system is, the k-representation principle, the normative user journey, the status discipline |
| [`docs/architecture/`](docs/architecture/) | how it is realized: the docs-as-code flow, workspace organization, the k-form pipeline |
| [`docs/adrs/`](docs/adrs/) | architecture decision records |
| [`model/`](model/) | the conceptual model as typed data: the dictionary (one row per capability), the status ledger, the oracle registry, the parametric use-cases |
| [`features/`](features/) | Gherkin definitions — one feature per dictionary row (BDD-first) |
| [`oracles/`](oracles/) | committed external validation artifacts + checksums |
| [`crates/`](crates/) | the Rust workspace (see the crate table in [ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)) |
| [`apps/web/`](apps/web/) | the browser application (React + the wasm binding) + the browser BDD suite |
| [`xtask/`](xtask/) | automation: oracle verification, pin checks, the conformance ledger, fixture generation |

## The user journey

The application's contract is one journey, verified end-to-end in real
Chromium — hermetically on every push, against live HuggingFace on the
scheduled matrix:

1. **Download** — any HuggingFace repository; safetensors shards stream
   tensor-by-tensor into the OPFS κ-store (`tensors/{κ}.bin`), gated by a
   config-derived memory guard.
2. **Compile** — a parametric decoder graph built solely from the model's own
   `config.json` + tensor manifest, compiled to a weightless k-form `.holo`
   bound to its weights by κ-labels.
3. **Run** — the archive materializes against the κ-store (every buffer
   re-hashes to its κ — content addressing is the integrity check) and
   executes in an inference session. Autoregressive reuse is content-addressed
   elision; there is no KV-cache.
4. **Chat** — a three-message handshake with streamed tokens, the model's own
   template, and its declared stop conditions.

## Quickstart

```sh
just            # list tasks
just vv         # the full local gate (fmt, lint, test, bdd, honesty, oracles, journey, …)
just bdd        # the Rust Gherkin suites (default lane)
just journey    # the hermetic browser journey in headless Chromium
just report     # the conformance ledger
```

Native CLI (host shell):

```sh
cargo run -p hologram-ai -- compile --model model.onnx --output out/
cargo run -p hologram-ai -- run --model out/model.holo --fill ones
cargo run -p hologram-ai -- download <hf-repo-id>
```

## Verification & validation

Every dictionary row carries a status that is a *contract on what the suite
may assert* (`verified` / `build` / `open` — see
[03-status-discipline.md](docs/conceptual-model/03-status-discipline.md)), an
oracle from the registry, and a Gherkin feature executed by the Rust cucumber
runner or the browser (cucumber-js + Playwright) runner. A mechanical
**honesty meta-gate** enforces the model ⇄ features ⇄ witnesses links and
forbids asserting open claims. `just vv` runs the whole gate set; CI mirrors
it job-for-job, and the Pages deployment requires the browser journey green.

External authorities: ONNX Runtime and the official ONNX node-test corpus,
the BLAKE3 test vectors, the reference `safetensors` and HuggingFace
`tokenizers` implementations, GGML quantization goldens, the live HuggingFace
Hub at pinned revisions, and the pinned [hologram]/[holospaces] substrate
witnesses. See [`model/oracles.toml`](model/oracles.toml).

[hologram]: https://github.com/Hologram-Technologies/hologram
[holospaces]: https://github.com/Hologram-Technologies/holospaces
[UOR-Atlas-UTQC]: https://github.com/afflom/UOR-Atlas-UTQC
