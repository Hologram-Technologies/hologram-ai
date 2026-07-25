# Upstream contract: uor-r4

Base revision: `f1b4859e65363eda9aa7dbeb0db467d93c8f4b02` (UOR-Foundation/uor-r4 main)
Patch branch: `feature/typed-integration-facade`, tip `384a0e9` (worktree `../uor-r4-facade`)
Status: implemented and gate-green; consumed by path until merged upstream.
Patch-ready diff: `docs/upstream/patches/uor-r4-typed-facade.patch` (verified
to apply cleanly against the base revision).

## What hologram-ai needs (and why)

hologram-ai integrates with uor-r4 as a **library** — never by invoking its
CLI, constructing `cargo run` argument strings, or parsing stdout. The patch
adds one additive crate, `uor-r4-api`, wrapping existing functionality. It
must not redesign compilation, change scoring/quality behavior, remove
legacy paths, add `.holo` logic, add networking, or add SDK-specific types.

## Required surface (approximate names; ownership boundaries are normative)

```rust
// Typed compilation over a verified local source directory.
pub struct CompileRequest {
    pub source_dir: PathBuf,       // config.json + model.safetensors + tokenizer.json
    pub work_dir: PathBuf,         // private resumable workspace
    pub options: CompileOptions,   // seconds/target/sequence_length/depths/k0/budgets
}
pub enum CompileOutcome {
    Complete(CompiledModel),
    Incomplete { /* typed resumability signal, not stdout text */ },
}
pub struct CompiledModel {
    pub graph: Vec<u8>,               // scored deployable R4G1 bytes
    pub signature_artifact: Vec<u8>,  // input-projection artifact (semantic role)
    pub tokenizer: Option<Vec<u8>>,
    pub score_report: Vec<u8>,
    pub compile_report: Vec<u8>,
    pub provenance: CompileProvenance, // options + versions + component digests
}
pub fn compile(req: &CompileRequest, progress: &mut dyn FnMut(ProgressEvent))
    -> Result<CompileOutcome, CompileError>;

// Engine from bytes — no filesystem, no sidecar paths.
pub struct EngineParts<'a> {
    pub graph: &'a [u8],
    pub signature_artifact: &'a [u8],
    pub tokenizer: Option<&'a [u8]>,
    pub score_report: Option<&'a [u8]>,
}
impl R4Engine {
    pub fn load(parts: EngineParts<'_>) -> Result<Self, LoadError>;
    pub fn predict_next_into(&mut self, window: &[u32], out: &mut PredictOutput)
        -> Result<(), InferenceError>;
    pub fn generate_into(&mut self, seed: &[u32], output_tokens: &mut [u32])
        -> Result<GenerateStatus, InferenceError>;
    pub fn reset(&mut self);
    pub fn abi_version(&self) -> AbiVersion;
}

// Tokenizer from bytes (compile-side export already exists).
impl Tokenizer { pub fn from_bytes(bytes: &[u8]) -> io::Result<Self>; }
```

Semantics hologram-ai relies on:

- graph bytes are validated (two-stage + CIDs) before engine initialization;
- abstention is a typed outcome; no fabricated tokens;
- steady-state predict/generate steps perform zero heap allocation after
  engine/state initialization;
- structured error enums (`CompileError`, `LoadError`, `InferenceError`),
  no `unwrap`/`expect` on recoverable paths;
- deterministic outputs: no HashMap-order, clock, or RNG dependence.

## Previously missing upstream (gaps this patch closes — all delivered in
## `384a0e9`)

1. Typed compile request/response (today: three CLI-shaped stages taking
   `&[String]`, `Result<(), String>`, conventional filenames; resumability
   signalled via stdout text).
2. Bytes-based engine constructor bundling graph + signature artifact +
   tokenizer (today: root-package `R4g1State::load(paths)` reading sidecars).
3. `Tokenizer::from_bytes` (today: path-only `try_load`).
4. Typed generation in a library crate (today: root package only).
5. ABI/version handshake API (today: raw constants only).
6. Structured errors at compile/load boundaries (today: `String`).

## Known follow-ups (not in this patch)

- `uor-r4-graph-runtime`'s `predict_distribution` hardcodes vocab/tokenizer
  constants; the std `GraphScorer` is the model-agnostic path hologram-ai
  uses. A no_std model-agnostic scorer is required for the browser phase.
- Reusable contract checks (P-4 source scan, allocation census) are
  test-only upstream; hologram-ai asserts its own wrapper census meanwhile.
