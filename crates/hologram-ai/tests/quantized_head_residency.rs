//! Quantized chunked-head residency V&V (row `quantized-transit`, chunked head).
//!
//! The deployed 1.5B log showed the LM-head stages re-materializing on EVERY
//! generated token — a `RuntimeError`-adjacent thrash costing ~30–45 s/token —
//! because a TIED head is a bf16 matmul whose whole-panel F32 Cast image
//! (~1.8 GB across the head chunks) cannot stay resident under the wasm address
//! ceiling, so admission evicts it and it rebuilds each step. The int8 tier
//! that removed the other projections' F32 images had structurally EXCLUDED the
//! head: its matcher only quantized whole (un-ranged) externals, and head
//! chunks bind ranged slices.
//!
//! This test proves the fix: a chunked head joins the int8 tier — each vocab
//! chunk crystallizes its own per-chunk artifact (keyed by κ + byte range) and
//! becomes a dequant-fused int8 matmul with NO F32 panel. Under a footprint
//! ceiling, the whole model then stays resident, so generating N tokens costs
//! exactly `stage_count` materializations — each stage ONCE — not `stage_count`
//! per token. Verified for BOTH a tied head (shares the embedding κ, the 1.5B
//! case) and an untied head (its own `lm_head.weight`), so the property is the
//! family-general one, not fit to one model.

mod common;

use common::families::{dummy_bf16_bytes, is_norm, Dims, FamilyScale, FAITHFUL_FAMILIES, QWEN2};

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A per-call discriminator so concurrently-running tests never share a κ-store
/// directory (a shared dir races on content-addressed file writes).
static STORE_SEQ: AtomicUsize = AtomicUsize::new(0);

fn unique_store_dir(tag: &str) -> std::path::PathBuf {
    let seq = STORE_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("hai-{tag}-{}-{seq}", std::process::id()))
}

use hologram_ai::materialize::DirKappaStore;
use hologram_ai::quantized::{crystallize_quantized, crystallize_quantized_range};
use hologram_ai::staged::{head_quant_chunks, quantizable_weights, GrowableStagedSession};
use hologram_ai::DecodeSession;
use hologram_ai_common::lower::{quant_key, QuantMap};
use hologram_ai_common::DType;

const PROMPT: [i64; 5] = [1, 2, 3, 5, 8];
const GEN_STEPS: usize = 6;

/// A small model in every dimension EXCEPT a vocabulary large enough that the
/// head partitions into several chunks (`vocab · hidden` exceeds one layer
/// stage's element count → `head_chunk_rows < vocab`). Parametric: the chunk
/// count follows the config arithmetic, never a magic constant. Everything else
/// stays tiny so the whole int8 pipeline compiles and decodes in well under a
/// second — this test exercises the CHUNKED-head path the modest coverage scale
/// (single-chunk head) does not.
fn chunked_head_dims() -> Dims {
    let mut d = Dims::MODEST;
    d.vocab_size = 16_384;
    d
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Cosine similarity of two equal-length logit rows — the faithfulness metric
/// between the float-head reference and the int8-head result.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

/// The outcome of driving the pipeline for `scale`: how many head chunks the
/// head partitioned into, the total stage materializations over feed +
/// generate, the pipeline stage count, the generated tokens, and the final
/// logit row.
struct Outcome {
    head_chunks: usize,
    total_materializations: u64,
    stage_count: usize,
    tokens: Vec<i64>,
    final_logits: Vec<f32>,
}

/// Build a footprint-bounded staged decode session for `scale` — the
/// attention/MLP projections always int8, the head int8 iff `quantize_head` —
/// feed `PROMPT`, generate `GEN_STEPS` greedy tokens on the STEP runner (no
/// seeder — this isolates one runner's residency, the per-token thrash
/// surface), and report the outcome. With `quantize_head = false` the head is
/// the classic bf16 chunked matmul (the reference); with `true` each chunk is a
/// dequant-fused int8 matmul (the fix).
fn drive(scale: &FamilyScale, quantize_head: bool) -> Outcome {
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    // κ-store: the wide bf16 weights. Unique per call — parallel tests must not
    // share a store dir (they would race on content-addressed writes).
    let store_dir = unique_store_dir(&format!("qhead-{arch}"));
    std::fs::create_dir_all(&store_dir).expect("creating κ-store temp dir");
    let mut dir = DirKappaStore::new(&store_dir);
    let mut kappas: Vec<String> = Vec::with_capacity(keys.len());
    for (name, dims) in keys.iter().zip(&shapes) {
        let kappa = dir
            .insert(&dummy_bf16_bytes(name, dims))
            .unwrap_or_else(|e| panic!("{arch}: persisting weight {name}: {e:#}"));
        kappas.push(kappa);
    }

    // int8 tier — the whole projection chain: the 196-style attention/MLP
    // weights (whole κ), THEN the head chunks (per-chunk artifacts keyed by κ +
    // range). This is the browser download's quantized tier end to end.
    let idx_of: HashMap<&str, usize> = kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let mut quant = QuantMap::new();
    let eligible = quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
        .unwrap_or_else(|e| panic!("{arch}: quantizable_weights: {e:#}"));
    for wide in &eligible {
        let i = idx_of[wide.as_str()];
        assert!(
            !is_norm(&keys[i]),
            "{arch}: a norm weight is not quantizable"
        );
        let (out, inf) = (shapes[i][0], shapes[i][1]);
        let entry = crystallize_quantized(&mut dir, wide, DType::BF16, out, inf)
            .unwrap_or_else(|e| panic!("{arch}: crystallizing {wide}: {e:#}"));
        quant.insert(wide.clone(), entry);
    }
    let head_targets = head_quant_chunks(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
        .unwrap_or_else(|e| panic!("{arch}: head_quant_chunks: {e:#}"));
    let head_chunks = head_targets.len();
    if quantize_head {
        for target in &head_targets {
            let entry = crystallize_quantized_range(
                &mut dir,
                &target.kappa,
                target.offset,
                target.len,
                DType::BF16,
                target.out_features,
                target.in_features,
            )
            .unwrap_or_else(|e| panic!("{arch}: crystallizing head chunk: {e:#}"));
            quant.insert(
                quant_key(&target.kappa, Some((target.offset, target.len))),
                entry,
            );
        }
    }

    // Footprint-bounded session (the wasm residency model). The budget is
    // generous for this tiny int8 model — the POINT is that the int8 head has
    // no F32 panel, so the whole model fits and holds resident; the OLD bf16
    // head could not (its panels alone dwarf the model).
    let mut session = GrowableStagedSession::new(
        config_json.clone(),
        keys.clone(),
        kappas.clone(),
        shapes.clone(),
        dtypes.clone(),
        None,
        one,
        Box::new(DirKappaStore::new(&store_dir)),
    )
    .unwrap_or_else(|e| panic!("{arch}: growable session: {e:#}"));
    session.set_residency_budget(1 << 30); // 1 GiB — ample for the int8 model
    session.set_bound_by_footprint(true);
    session.set_quant_map(quant);

    let ctx = scale.dims.max_position_embeddings as usize;
    let want = PROMPT.len();
    let step = session
        .decode_runner_for(want)
        .unwrap_or_else(|e| panic!("{arch}: decode step runner: {e:#}"));
    let stage_count = step.stage_count();
    let mut decode = DecodeSession::new(step, scale.dims.rope_theta as f32, ctx as u64)
        .unwrap_or_else(|e| panic!("{arch}: decode session: {e:#}"));

    // Feed WITHOUT a seeder: prompt steps one at a time on the step runner, so
    // token 1 materializes every stage and tokens 2.. are resident hits.
    let mut row = decode
        .feed(&PROMPT)
        .unwrap_or_else(|e| panic!("{arch}: prefill: {e:#}"));
    let mut tokens = Vec::with_capacity(GEN_STEPS);
    for s in 0..GEN_STEPS {
        let next = argmax(&row) as i64;
        tokens.push(next);
        row = decode
            .step(next)
            .unwrap_or_else(|e| panic!("{arch}: decode step {s}: {e:#}"));
    }

    assert!(
        row.iter().all(|x| x.is_finite()),
        "{arch}: int8-head decode produced non-finite logits"
    );
    assert_eq!(
        row.len(),
        scale.dims.vocab_size as usize,
        "{arch}: logits width must equal the vocabulary"
    );

    Outcome {
        head_chunks,
        total_materializations: decode.runner().materialization_count(),
        stage_count,
        tokens,
        final_logits: row,
    }
}

/// The chunked int8 head stays resident: generating many tokens costs exactly
/// `stage_count` materializations — each stage materialized ONCE across the
/// whole feed + generate — instead of re-materializing the head stages every
/// token (the deployed thrash). Holds for a tied head (shares the embedding κ)
/// and an untied head (its own weight) alike.
#[test]
fn quantized_head_stays_resident_across_decode_steps() {
    for &layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(layout, chunked_head_dims());
        let arch = scale.arch();

        let out = drive(&scale, true);

        assert!(
            out.head_chunks >= 2,
            "{arch}: the scale must actually CHUNK the head (got {} chunk(s)) — else the \
             ranged-artifact path is untested",
            out.head_chunks
        );
        // The whole model (int8 body + int8 head) is resident: every stage
        // materialized exactly once over feed + {GEN_STEPS} steps. A bf16 head
        // would re-materialize its chunks each token → far more than one pass.
        assert_eq!(
            out.total_materializations, out.stage_count as u64,
            "{arch}: expected each of {} stages to materialize exactly ONCE across feed + \
             {GEN_STEPS} decode steps (no per-token head re-materialization); saw {} \
             materializations — the head is thrashing",
            out.stage_count, out.total_materializations
        );
        assert_eq!(out.tokens.len(), GEN_STEPS);
        assert!(
            out.tokens
                .iter()
                .all(|&t| (t as u64) < scale.dims.vocab_size),
            "{arch}: a generated token is out of vocabulary range"
        );

        // Reproducible run to run (a broken residency/splice shows as drift).
        let again = drive(&scale, true);
        assert_eq!(
            out.tokens, again.tokens,
            "{arch}: int8-head decode is not reproducible run-to-run"
        );

        eprintln!(
            "[quantized-head] {arch}: {} head chunk(s), {} stages, \
             {} materializations over feed + {GEN_STEPS} steps (each stage once) — resident, no thrash",
            out.head_chunks, out.stage_count, out.total_materializations
        );
    }
}

/// The int8 head is a FAITHFUL quantization of the bf16 head, not merely
/// internally consistent: the same model decoded with a float chunked head (the
/// reference) and with the int8 chunked head (the fix) produces near-identical
/// logits — the int8 tier is the model's own established regime (all 196
/// projections are already int8), and the head is one more linear layer, so its
/// per-channel int8 form tracks the float head the way `int8_accuracy` proves a
/// linear layer does. Verified for tied and untied heads.
#[test]
fn int8_head_logits_track_the_float_head() {
    for &layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(layout, chunked_head_dims());
        let arch = scale.arch();

        let float_head = drive(&scale, false);
        let int8_head = drive(&scale, true);
        assert!(int8_head.head_chunks >= 2, "{arch}: head must chunk");

        let sim = cosine(&float_head.final_logits, &int8_head.final_logits);
        assert!(
            sim >= 0.99,
            "{arch}: int8-head logits must track the float head (cosine {sim:.5} < 0.99) — \
             the head's per-channel int8 form is unfaithful"
        );
        eprintln!("[quantized-head] {arch}: int8-vs-float head logits cosine = {sim:.5}");
    }
}

/// The VERIFY (speculative-decode, all-positions) head joins the int8 tier under
/// the SAME κ+range keys as the decode head — the load-bearing cross-plan
/// alignment. The quant map is derived once from the WINDOW plan
/// (`head_quant_chunks` → `build_parametric_stage_graphs`) but consumed by the
/// VERIFY plan (`build_parametric_verify_stage_graphs`); they align only because
/// the head chunking and byte-range offsets are seq-independent. If they ever
/// diverged the verify head would silently keep its wide bf16 binding — the
/// exact F32-panel thrash on the speculative path. This asserts they DON'T.
#[test]
fn verify_head_chunks_are_int8_under_the_window_derived_keys() {
    use hologram_ai::materialize::kappa_requirements;
    use hologram_ai::staged::compile_verify_stages;

    let scale = FamilyScale::new(QWEN2, chunked_head_dims());
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    let store_dir = unique_store_dir("verify");
    std::fs::create_dir_all(&store_dir).expect("temp dir");
    let mut dir = DirKappaStore::new(&store_dir);
    let kappas: Vec<String> = keys
        .iter()
        .zip(&shapes)
        .map(|(name, dims)| dir.insert(&dummy_bf16_bytes(name, dims)).expect("persist"))
        .collect();

    // WINDOW-plan-derived head-chunk targets → per-chunk int8 artifacts.
    let targets = head_quant_chunks(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
        .expect("head_quant_chunks");
    assert!(targets.len() >= 2, "{arch}: the head must chunk");
    let mut quant = QuantMap::new();
    let mut artifact_kappas = std::collections::HashSet::new();
    for t in &targets {
        let entry = crystallize_quantized_range(
            &mut dir,
            &t.kappa,
            t.offset,
            t.len,
            DType::BF16,
            t.out_features,
            t.in_features,
        )
        .expect("crystallize head chunk");
        artifact_kappas.insert(entry.0.clone());
        quant.insert(quant_key(&t.kappa, Some((t.offset, t.len))), entry);
    }
    let head_kappa = targets[0].kappa.clone();

    // Compile the VERIFY plan with those WINDOW-derived keys.
    let ctx = scale.dims.max_position_embeddings as usize;
    let bucket = hologram_ai::engine::geometric_window(PROMPT.len().max(1), ctx) as u64;
    let draft = 2u64; // a K=2 speculative draft
    let stages = compile_verify_stages(
        &config_json,
        &keys,
        &kappas,
        &shapes,
        &dtypes,
        bucket,
        draft,
        one,
        Some(&quant),
    )
    .expect("verify stages compile");

    // No head-chunk verify stage may still bind a RANGE of the wide head κ; each
    // binds its per-chunk int8 artifact — proving the window keys matched the
    // verify plan's ranged bindings.
    let mut wide_head_ranges = 0usize;
    let mut artifact_hits = 0usize;
    for archive in &stages {
        for req in kappa_requirements(archive).expect("κ-map parses") {
            if req.kappa == head_kappa && req.range.is_some() {
                wide_head_ranges += 1;
            }
            if artifact_kappas.contains(&req.kappa) {
                artifact_hits += 1;
            }
        }
    }
    assert_eq!(
        wide_head_ranges, 0,
        "{arch}: the verify head must be int8 — no ranged wide-head binding may survive \
         (window→verify key misalignment would leave it a bf16 F32-panel matmul)"
    );
    assert!(
        artifact_hits >= targets.len(),
        "{arch}: every verify head chunk must bind its per-chunk int8 artifact"
    );
    eprintln!(
        "[quantized-head] {arch}: verify head is int8 across {} chunk(s) under the window-derived keys",
        targets.len()
    );
}
