# Plan 061: GGUF Removal, SD v1.5 Correctness, Semantic Output Metadata, SDXL

**Status:** In progress
**Created:** 2026-04-07
**Branches (one per repo, same name):**
- `hologram-ai`: `feat/sd-correctness-and-output-types` (off `main`)
- `hologram` (base): `feat/sd-correctness-and-output-types` (off `main`)

Both branches must exist before any work starts. Stage 0 (GGUF removal) and Stage B (extending `ModelMetaSection`) require coordinated edits across the two repos; we land them as a paired PR set.

**SPRINT.md tracking:** [specs/SPRINT.md](../SPRINT.md) is updated as each stage lands — a "GGUF removed" line, v1.5 correctness done with the golden-image gate, the semantic output metadata milestone, and SDXL pipeline status. Each stage's verification step explicitly includes a SPRINT.md update.

## Context

Earlier triage claimed SD v1.5 was "fully working" because [sd_pipeline_e2e.rs](../../crates/hologram-ai/tests/sd_pipeline_e2e.rs) compiles + runs without panicking. **That was wrong.** The test only asserts the output PPM exists and is >1000 bytes — it never validates image content. The actual generated image **has some structure but is mostly noise** — there's *some* signal (so VAE isn't outputting all zeros and the latent isn't completely garbage), but the signal is weak/wrong. This is a numerical correctness bug, not a missing-feature bug.

The companion question is whether this work gives us **output types** — text vs. image vs. audio. Important framing: models always output **raw bytes** (the underlying f32/i64 tensor). What's missing is *metadata for downstream consumers* describing how to interpret those bytes — "this output is an RGB NCHW image in [-1, 1]" vs. "this is logits over a 32000-token vocab" vs. "this is a 768-dim text embedding". Today the archive carries a coarse `ModelKind` enum (`hologram/crates/hologram-archive/src/section/model_meta.rs:14-33`) that says *what kind of model* it is, but there is no per-output descriptor. Without that, the `hologram run` command (which already exists at `hologram/crates/hologram-cli/src/commands/run_cmd.rs`) can't generically post-process outputs into the right artifact (PNG, text, WAV) — it has to assume bytes-on-stdout. This work adds the missing per-output semantic metadata so any consumer (including `hologram run` and our integration tests) can do the right thing.

These two concerns are related: a correctness fix is the immediate blocker, and an output-type system is the foundation for generically supporting top-20 HF models (text → text, text → image, text+image → text, audio → text, etc.) without bespoke per-model code.

This plan ships **Stage 0 (GGUF removal) → A (v1.5 correctness) → B (semantic output metadata) → C (SDXL)**. Stage 0 is a prerequisite cleanup: remove the `hologram-ai-gguf` crate and all its consumers, with a TinyLlama ONNX ~40 tok/s regression gate to prove the removal didn't break the LLM baseline. Semantic output types thread through Stages A → C as metadata that downstream consumers (tests today, `hologram run` later) use to interpret the raw output bytes correctly.

---

## What we already know

- **Failure mode:** "Some structure but mostly noise." There's signal in the output, so VAE isn't completely broken and the latent isn't totally random — but conditioning isn't taking effect strongly enough. Suspect order: (1) VAE drift / fused-kernel bug, (2) text encoder producing weak/wrong embeddings, (3) UNet cross-attention not applying conditioning, (4) CFG scale/sign, (5) scheduler off-by-one in alpha indexing. Pure-noise causes (channel layout, all-zero VAE) are *less* likely now since there's structure.
- **Known suspect (12 days old, must verify):** The inline binary fast path in `tape.rs` may size outputs from `output_byte_hint` (inflated by concretization to context_length=2048) instead of actual broadcast result, producing 1-float outputs where 825 are expected. Even if patched since, the *class* of bug (compile-time hints diverging from runtime shapes) is what to look for.
- **TinyLlama ONNX works** at ~43 tok/s baseline (recorded 2026-04-04). This is the regression baseline GGUF removal must not break.
- **`hologram` CLI has a generic `run` command** at `hologram/crates/hologram-cli/src/commands/run_cmd.rs` — 795 lines, takes a `.holo` + inputs and runs it, including text generation. The compiler-only constraint applies to *hologram-ai*, not hologram base. So end-to-end image generation can be wired into `hologram run` once we have semantic output metadata to know "this output is an image, save it as PNG" — but Stage A keeps the SD pipeline as an integration test for tightest test-loop iteration; promoting it to a `hologram run` feature is a Stage B.5 follow-up.
- **`convert.rs`** at [crates/hologram-ai/src/download/convert.rs](../../crates/hologram-ai/src/download/convert.rs) already runs inline Python via a venv to do `torch.onnx.export`. Stage A.1's reference-latent capture extends this pattern with a new inline Python script — no new tooling, no new dev-env requirements.
- ModelKind enum already covers TextLlm, TextEncoder, Vision, Audio, ImageGen, AudioGen, VideoGen, MultiModal, Generic. CLI parses `kind` from manifest at [cli.rs:330-344](../../crates/hologram-ai/src/cli.rs#L330-L344). What's *missing* is (1) auto-detection from a single ONNX file, (2) per-output structured descriptors, (3) flow into single-model compilation.

---

## Stage 0 — Remove the GGUF crate (with a TinyLlama ONNX baseline gate)

**Goal:** `hologram-ai-gguf` is gone — crate, dependency, enum variants, CLI inspect path, lib re-exports — and TinyLlama ONNX still runs at ≥40 tok/s to prove no infrastructure regressed in the cleanup.

ONNX TinyLlama works (~43 tok/s baseline). GGUF was historically a parallel import path; we'll never use it again. Removing it shrinks the surface area, kills a class of import-divergence bugs, and forces every future model to come in through the ONNX path we're actually investing in.

### 0.1 — Establish the baseline (read-only first)

The correct baseline command uses `hologram-ai run` (the `hologram-ai` binary, not the `hologram` base CLI), with deterministic sampling and a chat-formatted prompt:
```
RUST_LOG=info cargo run --release -- run \
  models/TinyLlama-1.1B-Chat-v1.0/model.holo \
  --prompt "Question: What is the capital of France? Answer:" \
  --max-tokens 15 --temperature 0.0 --stop $'\n'
```
Two things matter: `--temperature 0.0` forces argmax (deterministic, avoids sampling-noise gibberish), and the `Question: ... Answer:` format matches TinyLlama-Chat's fine-tuning. With default sampling and a bare prompt, output looks broken even though the model is fine.

**Baseline established (2026-04-07): 42.9 tok/s decode, coherent output** (`The capital of France is Paris.`). Gate: ≥40 tok/s with coherent English. The `hologram` base CLI's `run` subcommand is a simpler runner that doesn't handle multi-component LLM pipelines — use `hologram-ai run` for any LLM correctness or perf work.

### 0.2 — Remove GGUF (mechanical)

Files to delete or edit (verified by grep):

- **Delete entirely:** `crates/hologram-ai-gguf/` (whole crate directory)
- **Workspace `Cargo.toml`:** remove the `hologram-ai-gguf` member entry and the `[workspace.dependencies]` line
- **`crates/hologram-ai/Cargo.toml`** line 19: remove `hologram-ai-gguf.workspace = true`
- **`crates/hologram-ai/src/lib.rs`** line 21: remove `pub use hologram_ai_gguf::{import_gguf, GgufImportOptions};`
- **`crates/hologram-ai/src/compiler.rs`** lines 991-993: remove the `ModelSource::GgufPath` match arm; remove the `GgufPath` variant from the `ModelSource` enum entirely (it's safe since ADR-0016 says we can break APIs freely)
- **`crates/hologram-ai/src/validate.rs`** line 214: remove the `"gguf"` extension branch; replace with an error
- **`crates/hologram-ai/src/cli.rs`** lines 257-258: remove `inspect_gguf` function and its call site in `main`
- Any GGUF references in `download/`, `convert.rs`, model paths, tests, fixtures
- `models/TinyLlama-1.1B-Chat-v1.0-GGUF/` directory: leave on disk (gitignored) but remove any test that references it

After each delete, run `cargo build -p hologram-ai` to find the next reference. Iterate until clean.

### 0.3 — Baseline gate

After all deletions, re-run the exact same TinyLlama ONNX command from 0.1. **Acceptance: tok/s ≥ 40, output is coherent English.** If tok/s drops below 40, bisect with `git bisect` against the deletion commits — almost certainly the deletion accidentally cut a shared utility. Do not move on to Stage A until the gate passes.

### Stage 0 verification
```
cargo build --workspace
cargo test --workspace
cargo clippy -- -D warnings
# Baseline gate (must match the 0.1 command exactly)
RUST_LOG=info cargo run --release -- run \
  models/TinyLlama-1.1B-Chat-v1.0/model.holo \
  --prompt "Question: What is the capital of France? Answer:" \
  --max-tokens 15 --temperature 0.0 --stop $'\n'
# Must report ≥ 40 tok/s and coherent output (e.g. "The capital of France is Paris.")
```

Then update [specs/SPRINT.md](../SPRINT.md): under "Cleanup", add `[x] Removed hologram-ai-gguf crate (TinyLlama ONNX gate: ≥40 tok/s)`.

---

## Stage A — Fix v1.5 image correctness

**Goal:** `cargo test -p hologram-ai --features e2e -- sd_pipeline_generates_image` produces a recognizable cat-like image (not noise) for prompt "a photo of a cat" with 20 DDIM steps.

### A.1 — Lock down VAE in isolation (read-only, half a day)

The fastest bisection: run VAE on a **known-good latent** that we did *not* generate ourselves.

1. Extend [crates/hologram-ai/src/download/convert.rs](../../crates/hologram-ai/src/download/convert.rs) with a new inline Python script `SD_REFERENCE_CAPTURE_SCRIPT` that uses `diffusers` to:
   - Run SD v1.5 for "a photo of a cat" with a fixed seed
   - Save the **scaled latent** at the moment before VAE decode (`latent / 0.18215`) to `models/stable-diffusion-v1-5/known_good_latent.bin` (16384 f32 LE)
   - Save the **reference VAE output** to `known_good_image.bin` ([1,3,512,512] f32 LE)
2. New test `sd_vae_known_good_latent` (extending [sd_vae_e2e.rs](../../crates/hologram-ai/tests/sd_vae_e2e.rs)) loads the reference latent, runs our VAE, and:
   - Asserts max abs error vs reference < 1e-2 (loose — we use checkpointing + BLAS)
   - Saves PPM via the same `save_ppm` helper

**This single test will tell us:**
- If the PPM is **noise** → VAE itself or post-processing is broken (drill into A.2)
- If the PPM is **a cat** → VAE is fine; the bug is upstream in CLIP/UNet/scheduler (drill into A.3)
- If max abs error is huge but image looks plausible → VAE has subtle drift, possibly checkpointing or a fused kernel

### A.2 — VAE-side root cause (only if A.1 PPM is noise)

In priority order:
1. **Latent → save_ppm channel layout.** Verify the byte layout is C-contiguous channel-first for `[1, 3, H, W]`; if NHWC sneaks in anywhere, three channels collapse to one and we get noise.
2. **Output normalization.** `(v + 1) / 2` is correct *only if* VAE outputs ~[-1, 1]. Print `min/max/mean/std` for the VAE output. If we see [-100, 100] something fused a wrong activation; if all-zeros, something earlier short-circuited.
3. **Spatial-scale compile-time decision.** The pipeline test compiles VAE with `spatial_scale = Some(2)` ([sd_pipeline_e2e.rs:248-253](../../crates/hologram-ai/tests/sd_pipeline_e2e.rs#L248-L253)). Cross-check the compiled graph's input shape against `vae_inputs.set_with_shape`.
4. **Verify the binary fast path bug** is or isn't still live. Look for `output_byte_hint` usage in tape's inline binary dispatch; assert the hint matches actual broadcast size. If it doesn't, fix in hologram base (size from inputs, use hint only as a pre-allocation upper bound).
5. **Checkpoint-disabled run.** Set `vae_tape.checkpoint_enabled = false` and re-run. If image suddenly works, the recompute path has a stale-input bug.

For each candidate, run the A.1 known-good-latent test as the oracle.

### A.3 — Upstream root cause (in progress 2026-04-08)

VAE is responding to the latent (output PPM stats: mean=121.7, std=23.4, ratio of block_std to pixel_diff = 3.18 — strongly *structured*, not noise). The bug is upstream: the latent isn't denoising into meaningful structure. Bisect upstream:

1. ~~**Scheduler beta schedule**~~ — **FIXED.** `ddpm_alpha_bars` was using a plain *linear* beta interpolation, but SD v1.5's `scheduler_config.json` specifies `beta_schedule: "scaled_linear"` (interpolate the *square roots* of beta, then square). At t=999 this gives alpha_bar = 0.00466 instead of 0.00158 — a 2.95× difference that compounds across the DDIM `x0 = (x_t - sqrt(1-ab)*eps) / sqrt(ab)` step into severely wrong predictions. Verified against diffusers reference values.
2. ~~**dispatch_where div-by-zero**~~ — **FIXED in hologram base.** [gather_concat.rs:161](../../../hologram/crates/hologram-exec/src/float_dispatch/gather_concat.rs#L161) panicked with "remainder with divisor of zero" when `cond` was empty but `n > 0`. Fix: return empty Vec when any operand is empty (broadcast against zero-length axis = empty result). This was hit by the Q8 text encoder path during the SD pipeline.
3. **Single-step UNet vs ORT.** (*Not yet exercised — needs Python ORT env.*) Set `n_steps = 1`, t = 999. Capture noise prediction. In Python, run the same UNet ONNX through ORT with the same noisy latent + zero text embeddings. Cosine similarity must be > 0.999.
4. **Text encoder vs ORT.** Same pattern with the CLIP encoder for an empty string. Q8 path likely also has subtle drift even after the panic fix.
5. **CFG guidance scale.** Pipeline uses 7.5 with `uncond + scale * (cond - uncond)` — sign and order verified correct.
6. **PRNG seed.** Box-Muller LCG over `i` indexes — should produce N(0,1) per pixel, but verify by checking the per-step latent stats printed by the test.
7. **Spatial-scale concern noted.** [sd_pipeline_e2e.rs:248-253](../../crates/hologram-ai/tests/sd_pipeline_e2e.rs#L248-L253) compiles VAE with `spatial_scale = Some(2)` but feeds runtime input `[1, 4, 64, 64]` at line 410. If `spatial_scale=2` bakes the compiled input to `[1, 4, 32, 32]`, this is a runtime/compile shape mismatch. The 8-day-old `output.ppm` is 512×512 which doesn't match a `spatial_scale=2` output (256×256), so either the test code changed since the PPM was generated, or the runtime tolerates the mismatch silently. Investigate after the scheduler fix is verified.

Stop bisection at the first remaining divergent component once the scheduler + dispatch_where fixes are verified end-to-end.

### A.4 — Strengthen the pipeline test so this never silently regresses

The current assertion is `meta.len() > 1000` — useless. Replace with:

1. Save a **golden image** (`tests/golden/sd_v15_cat.bin`, [3,512,512] u8) once the pipeline is fixed.
2. Assert per-pixel mean abs error against golden < 5 (out of 255). Tolerant enough for BLAS reordering, tight enough to catch noise.
3. Assert image **statistics**: `std(luminance) > 20` (catches blank/uniform output), `max - min > 100` (catches near-noise).
4. **Semantic output assertion (depends on Stage B).** Once Stage B has landed, the same test loads the VAE `.holo` archive and asserts `archive.meta().io_signature.outputs[0].semantic == SemanticRole::Image { layout: Nchw, color: Rgb, value_range: MinusOneOne }`. This is the *first* concrete consumer of the semantic-output system and proves the metadata round-trips correctly through compilation.

This is the regression gate — if Stage B or C breaks something, this test will scream. Stage A.4 ships with assertions 1–3 immediately and gains assertion 4 the moment Stage B merges.

### Stage A critical files
- [crates/hologram-ai/tests/sd_pipeline_e2e.rs](../../crates/hologram-ai/tests/sd_pipeline_e2e.rs) — strengthen assertions, add golden comparison
- [crates/hologram-ai/tests/sd_vae_e2e.rs](../../crates/hologram-ai/tests/sd_vae_e2e.rs) — add `sd_vae_known_good_latent` test
- [crates/hologram-ai/src/download/convert.rs](../../crates/hologram-ai/src/download/convert.rs) — add `SD_REFERENCE_CAPTURE_SCRIPT`
- *Probably:* `hologram/crates/hologram-exec/src/tape.rs` (sibling repo) — fast-path output sizing fix
- *Probably:* `hologram/crates/hologram-exec/src/float_dispatch/spatial.rs` — Resize / VAE-related kernels if a bug surfaces there
- `models/stable-diffusion-v1-5/known_good_latent.bin`, `known_good_image.bin` — gitignored test fixtures

Then update [specs/SPRINT.md](../SPRINT.md): under "Stable Diffusion support", add `[x] v1.5 image quality regression fix — golden image gate in sd_pipeline_e2e.rs`.

---

## Stage B — Semantic output type metadata system

**Goal:** Every `.holo` archive carries a structured **per-output semantic role** so any consumer can answer "what does this model take and what does it produce, semantically?" without reading the graph. `hologram-ai info model.holo` prints the per-input and per-output semantic types. The system must extend cleanly to the top-20 HF model categories (text-gen, text-encode, image-gen, image-classify, audio-transcribe, audio-gen, multi-modal, etc.).

This is *not* just "set ModelKind correctly" — that gives us a coarse model-level category. We need **per-output descriptors** that say things like "output 0 is an RGB NCHW image in [-1, 1]", "output 1 is logits over a 32000-token vocab", or "output 0 is a 768-dim text embedding". Per-output is the right granularity because many models (e.g. CLIP, OpenCLIP-G in SDXL) emit *multiple* semantically distinct outputs from one graph.

### B.1 — Design: extend ModelMetaSection in hologram base

Add a new field to `hologram/crates/hologram-archive/src/section/model_meta.rs`:

```rust
pub struct ModelMetaSection {
    pub kind: ModelKind,                   // existing
    pub arch: String,                      // existing
    // ... existing fields ...
    pub io_signature: IoSignature,         // NEW
}

pub struct IoSignature {
    pub inputs: Vec<IoTensor>,
    pub outputs: Vec<IoTensor>,
}

pub struct IoTensor {
    pub name: String,                      // graph entrypoint name
    pub semantic: SemanticRole,            // NEW — see below
    pub dtype: DType,
    pub shape: Vec<DimSpec>,               // each dim is Concrete(u32) or Symbolic(name)
}

pub enum SemanticRole {
    // Text family
    TokenIds { vocab_size: u32, max_len: u32 },
    TextEmbedding { dim: u32 },
    Logits { vocab_size: u32 },
    // Image family
    Image { layout: ImageLayout, color: ColorSpace, value_range: ValueRange },
    ImageLatent { channels: u32, scaling_factor: f32 },
    // Audio family
    AudioWaveform { sample_rate: u32, channels: u32 },
    AudioMelSpectrogram { n_mels: u32, n_frames_dim: usize },
    // Diffusion-specific
    DiffusionTimestep,
    CrossAttentionStates { hidden_dim: u32 },
    // Catch-all
    Generic,
}

pub enum ImageLayout { Nchw, Nhwc }
pub enum ColorSpace { Rgb, Bgr, Grayscale, Latent }
pub enum ValueRange { ZeroOne, MinusOneOne, Standardized, Raw }
```

**Why this shape:** It's *structural enough* to drive generic post-processing (e.g., a `.holo → PNG` converter doesn't need to know what model it is, just that an output is `Image { Nchw, Rgb, MinusOneOne }`), but *open-ended enough* via `Generic` to never block compilation. Each variant carries the minimum info needed to actually consume the tensor.

**Append-only enums:** `SemanticRole` variants must always be appended at the end (rkyv stability). Document this in the new file.

### B.2 — Detection in hologram-ai

Two paths:

1. **Manifest-driven (existing extension).** Extend the manifest schema with optional `[[component.outputs]]` blocks specifying semantic role. For the SD pipeline, this lets us say "vae_decoder output 0 is `Image { Nchw, Rgb, MinusOneOne }`".

2. **Auto-detection from ONNX (new).** A new pass `crates/hologram-ai-common/src/opt/io_signature_inference.rs`:
   - Walk the ONNX graph's `model.metadata_props` and `graph.output` entries
   - Apply heuristics: shape `[N, 3, H, W]` with float dtype after a Tanh or Sigmoid → `Image`; shape `[N, vocab_size]` after a Softmax/Gather → `Logits`; shape `[N, dim]` from a pooler → `TextEmbedding`; etc.
   - Cross-reference architecture detection (CLIP / UNet / VAE / GPT-style) for higher-confidence labeling
   - Always falls back to `Generic`

3. **Manifest overrides auto-detection.** Test path uses manifest; CLI single-file compile uses auto-detection.

### B.3 — CLI surface

Extend `hologram-ai info`:
```
$ hologram-ai info models/stable-diffusion-v1-5/vae_decoder/model.holo
Kind: ImageGen
Arch: stable-diffusion-vae-decoder
Inputs:
  latent_sample: f32 [1, 4, 64, 64] — ImageLatent { channels: 4, scaling_factor: 0.18215 }
Outputs:
  sample: f32 [1, 3, 512, 512] — Image { Nchw, Rgb, MinusOneOne }
```

### B.4 — Tests

- `io_signature_roundtrip` (hologram base) — rkyv encode/decode
- `io_signature_append_only` (hologram base) — variants are stable across rkyv versions
- `infer_io_signature_sd_vae` (hologram-ai) — compile vae_decoder, assert detected `Image` output
- `infer_io_signature_clip` (hologram-ai) — compile text_encoder, assert detected `TextEmbedding` output
- `infer_io_signature_tinyllama` (hologram-ai) — already-tested LLM, assert `Logits` output

### Stage B critical files
- `hologram/crates/hologram-archive/src/section/model_meta.rs` (hologram base) — extend section
- `crates/hologram-ai-common/src/opt/io_signature_inference.rs` (new)
- `crates/hologram-ai-common/src/sections/meta.rs` — manifest output overrides
- `crates/hologram-ai/src/cli.rs` — manifest schema + `info` printing
- `crates/hologram-ai/src/compiler.rs` — wire detected/manifested signature into archive

Then update [specs/SPRINT.md](../SPRINT.md): add a new top-level item `[x] Semantic output metadata system — IoSignature + SemanticRole, per-output detection, hologram-ai info prints types`.

---

## Stage C — SDXL pipeline (follow-on)

**Goal:** Same e2e pipeline pattern as v1.5 but for SDXL: dual text encoders, added time conditioning, 1024×1024 output.

Now that v1.5 is correct (Stage A) and we have semantic output metadata (Stage B), SDXL is straightforward. Every kernel SDXL needs already exists.

### Deltas from v1.5

| | v1.5 | SDXL |
|---|---|---|
| Text encoders | 1× CLIP-L (768d) | CLIP-L + OpenCLIP-G; concat to 2048d cross-attn states; OpenCLIP-G also yields pooler_output |
| UNet inputs | sample, timestep, encoder_hidden_states | + `text_embeds` [1,1280] + `time_ids` [1,6] |
| Latent | [1,4,64,64] | [1,4,128,128] |
| VAE scaling | 0.18215 | 0.13025 |
| Scheduler | DDIM same betas | Same |

### Steps
1. `optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task stable-diffusion-xl models/stable-diffusion-xl-base-1.0/`
2. Add four compile tests mirroring v1.5: `sdxl_text_encoder_e2e.rs`, `sdxl_text_encoder_2_e2e.rs`, `sdxl_unet_e2e.rs`, `sdxl_vae_e2e.rs`
3. Triage any compile failures using existing infra in `crates/hologram-ai-onnx/src/op_map.rs` and `crates/hologram-ai-common/src/lower/strategy.rs`
4. Add `sdxl_pipeline_e2e.rs` with: dual tokenizer, dual text encoder, hidden-state concat, `text_embeds`/`time_ids` construction, DDIM loop with CFG, VAE decode, golden-image assertion (Stage A.4 pattern)
5. **Exercise Stage B semantic types end-to-end** on every SDXL component:
   - `sdxl_text_encoder_e2e` asserts output 0 is `TextEmbedding { dim: 768 }`
   - `sdxl_text_encoder_2_e2e` asserts output 0 is `TextEmbedding { dim: 1280 }` *and* output 1 is a pooled `TextEmbedding { dim: 1280 }` (the dual-output case is exactly why per-output semantics matter)
   - `sdxl_unet_e2e` asserts inputs include `DiffusionTimestep` and `CrossAttentionStates { hidden_dim: 2048 }`, output is `ImageLatent { channels: 4, scaling_factor: 0.13025 }`
   - `sdxl_vae_e2e` asserts output is `Image { Nchw, Rgb, MinusOneOne }` at 1024×1024
   - `sdxl_pipeline_e2e` asserts the *full chain* of semantic types matches the manifest, proving the output-type system can describe a complete multi-component pipeline

### Risk knobs
- Memory: VAE at 1024×1024 may need `spatial_scale=2` compile + `checkpoint_enabled = true`
- UNet OOM: compile at smaller spatial first, scale up incrementally

### Stage C critical files
- `crates/hologram-ai/tests/sdxl_text_encoder_e2e.rs`, `sdxl_text_encoder_2_e2e.rs`, `sdxl_unet_e2e.rs`, `sdxl_vae_e2e.rs`, `sdxl_pipeline_e2e.rs` (all new)
- *Conditional:* op_map / strategy if a new op surfaces

Then update [specs/SPRINT.md](../SPRINT.md): under "Stable Diffusion support", add `[x] SDXL base 1.0 pipeline — dual text encoders, 1024×1024, golden image gate`.

---

## Non-goals

- **`run` CLI command in hologram-ai** — forbidden by ADR-0016 (compiler-only). Pipelines stay as integration tests; hologram base's `run` command is the future home for image-output once Stage B's semantic types land.
- **Quantization correctness audit** — Q8/Q4 issues are tracked separately. Stage A uses the f32 variant; we revisit Q8 once the f32 path is golden.
- **DiT (SD3, Flux)** — separate plan; rectified-flow scheduling and transformer-based diffusion are a much larger lift.
- **Refiner model** for SDXL — base alone is sufficient for "working".
- **Top-20 HF model expansion beyond SDXL** — Stage B is the *foundation* for that expansion, but adding individual models is per-model work tracked separately.

---

## End-to-end verification

```
# Stage 0
cargo run --release -p hologram --bin hologram -- run \
  models/TinyLlama-1.1B-Chat-v1.0/model.holo \
  --prompt "What is the capital of France?" --max-tokens 50
# Must report ≥ 40 tok/s, coherent English

# Stage A
cargo test -p hologram-ai --features e2e -- sd_vae_known_good_latent --nocapture
cargo test -p hologram-ai --features e2e -- sd_pipeline_generates_image --nocapture
# Visually inspect models/stable-diffusion-v1-5/output.ppm — it must look like a cat
# Then the golden assertion locks it in

# Stage B
cargo test -p hologram-archive io_signature
cargo test -p hologram-ai infer_io_signature
hologram-ai info models/stable-diffusion-v1-5/vae_decoder/model.holo
# Should print Image{Nchw,Rgb,MinusOneOne} for the output

# Stage C
cargo test -p hologram-ai --features e2e -- sdxl --nocapture
# Visually inspect models/stable-diffusion-xl-base-1.0/output.ppm

# Always
cargo clippy -- -D warnings
cargo fmt --check
```

---

## Open questions

- Stage B extends `ModelMetaSection` in hologram base — is this within the umbrella of the existing archive ADR, or does it warrant a new ADR? Working assumption: it's an additive section change, no new ADR needed.
- Stage B cross-repo commit ordering — land hologram base changes first and rev the hologram-ai dependency, or land them simultaneously via a workspace dependency override during development? Stage 0 has no hologram base changes; only Stage B and possibly Stage A.2 fast-path fix touch hologram base.
