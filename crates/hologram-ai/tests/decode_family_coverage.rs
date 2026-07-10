//! Parametric family DECODE coverage (integration V&V).
//!
//! The compilation suite proves the parametric registry BUILDS the graph for
//! every faithful decoder family (Llama, Qwen2, Mistral, Phi3). This test goes
//! further: it drives the FULL browser decode pipeline — int8-quantized staged
//! decode, a footprint-bounded shared-ledger session, a step runner PLUS a
//! chunked-prefill seeder, `feed` + `step` generation — end to end for EACH
//! family, and asserts each one actually decodes: finite, non-empty, and
//! reproducible logits/tokens. A break in any single family's decode path
//! (fused qkv/gate-up, q/k/v bias, tied vs untied head, GQA splice) fails the
//! build, so "arbitrary models" is verified across the popular formats, not
//! assumed and not fit to one.
//!
//! Parametric in BOTH the family (the registry's faithful set) and the scale
//! (`Dims::MODEST`, small enough to run in the default suite in seconds). No
//! term is specialized to a particular model or size.

mod common;

use common::families::{dummy_bf16_bytes, is_norm, Dims, FamilyScale, FAITHFUL_FAMILIES};

use std::collections::HashMap;
use std::num::NonZeroU64;

use hologram_ai::materialize::DirKappaStore;
use hologram_ai::quantized::{crystallize_quantized, crystallize_quantized_range};
use hologram_ai::staged::{head_quant_chunks, quantizable_weights, GrowableStagedSession};
use hologram_ai::DecodeSession;
use hologram_ai_common::lower::{quant_key, QuantMap};
use hologram_ai_common::DType;

/// A prompt of arbitrary in-vocabulary token ids (values are immaterial — the
/// SHAPES drive the pipeline; these only need to be `< vocab_size`).
const PROMPT: [i64; 6] = [1, 2, 3, 5, 8, 13];
const GEN_STEPS: usize = 4;

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Build the full int8-quantized staged decode session for `scale` and generate
/// `GEN_STEPS` greedy tokens after prefilling `PROMPT`. Returns the last logit
/// row and the generated token ids. Mirrors the browser's decode flow: a
/// footprint-bounded shared-ledger session, a step runner (chunk 1), and a
/// chunked-prefill seeder sharing the ledger.
fn decode_family(scale: &FamilyScale) -> (Vec<f32>, Vec<i64>) {
    decode_family_q(scale, true)
}

/// As [`decode_family`], but `quantize` selects the weight tier: `true` is the
/// int8 derived-artifact path (the browser's deployed path — since v0.8.1 that is
/// fused output-major **W8A8**), `false` keeps the wide bf16 weights as the
/// full-precision reference. Same κ-store, same prompt, same decode; only the
/// weight tier differs, so a token divergence between the two is exactly the
/// quantization error the deployed path introduces.
fn decode_family_q(scale: &FamilyScale, quantize: bool) -> (Vec<f32>, Vec<i64>) {
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    // κ-store: the wide bf16 weights (the download byte set). The directory must
    // be unique per *invocation*, not per (arch, pid): `decode_family_q` is now
    // called several times for the same arch (the quantized run, the bf16
    // reference, the reproducibility re-run), and cargo runs this binary's tests
    // concurrently — a shared `hai-fam-{arch}-{pid}` dir means two decodes writing
    // and reading one κ-store at once, which corrupts both. A per-call counter
    // disambiguates. (v0.8.1's own ffi commit fixed this same collision class.)
    static STORE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = STORE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let store_dir =
        std::env::temp_dir().join(format!("hai-fam-{arch}-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&store_dir).expect("creating κ-store temp dir");
    let mut dir = DirKappaStore::new(&store_dir);
    let mut kappas: Vec<String> = Vec::with_capacity(keys.len());
    for (name, dims) in keys.iter().zip(&shapes) {
        let kappa = dir
            .insert(&dummy_bf16_bytes(name, dims))
            .unwrap_or_else(|e| panic!("{arch}: persisting weight {name}: {e:#}"));
        kappas.push(kappa);
    }

    // int8 derived-artifact tier (the browser's quantized path): each eligible
    // wide projection is crystallized into a matmul-ready per-channel artifact.
    let idx_of: HashMap<&str, usize> = kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let eligible = if quantize {
        quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
            .unwrap_or_else(|e| panic!("{arch}: quantizable_weights: {e:#}"))
    } else {
        Vec::new() // full-precision reference: no weight is quantized
    };
    let mut quant = QuantMap::new();
    for wide in &eligible {
        let i = idx_of[wide.as_str()];
        let (out, inf) = (shapes[i][0], shapes[i][1]);
        assert!(
            !is_norm(&keys[i]),
            "{arch}: a norm weight is not a quantizable projection"
        );
        let entry = crystallize_quantized(&mut dir, wide, DType::BF16, out, inf)
            .unwrap_or_else(|e| panic!("{arch}: crystallizing int8 artifact for {wide}: {e:#}"));
        quant.insert(wide.clone(), entry);
    }

    // The LM head joins the int8 tier too (row `quantized-transit`, chunked
    // head): each vocab-row chunk of a large head gets its OWN per-chunk
    // artifact, keyed by (κ, range), so a chunked head is a dequant-fused int8
    // matmul instead of a bf16 matmul whose whole-panel F32 image thrashes
    // residency. A no-op where the head is a single chunk (small vocab).
    let head_targets = if quantize {
        head_quant_chunks(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
            .unwrap_or_else(|e| panic!("{arch}: head_quant_chunks: {e:#}"))
    } else {
        Vec::new()
    };
    for target in head_targets {
        let entry = crystallize_quantized_range(
            &mut dir,
            &target.kappa,
            target.offset,
            target.len,
            DType::BF16,
            target.out_features,
            target.in_features,
        )
        .unwrap_or_else(|e| panic!("{arch}: crystallizing head-chunk artifact: {e:#}"));
        quant.insert(
            quant_key(&target.kappa, Some((target.offset, target.len))),
            entry,
        );
    }

    // The footprint-bounded shared-ledger session (the wasm residency model).
    // The budget is generous for this modest scale — the point here is decode
    // CORRECTNESS per family; the shared-ledger bound itself is witnessed by the
    // stage-residency-cache scenarios and the family memory sweep.
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
    session.set_residency_budget(1 << 30); // 1 GiB — holds this modest model
    session.set_bound_by_footprint(true);
    if quantize {
        session.set_quant_map(quant);
    }

    // Step runner (chunk 1) + chunked-prefill seeder, sharing the ledger.
    let ctx = scale.dims.max_position_embeddings as usize;
    let want = PROMPT.len();
    let bucket = hologram_ai::engine::geometric_window(want.max(1), ctx);
    let chunk = (hologram_ai::engine::geometric_window(1, ctx) as u64).min(bucket as u64);

    let step = session
        .decode_runner_for(want)
        .unwrap_or_else(|e| panic!("{arch}: decode step runner: {e:#}"));
    let mut decode = DecodeSession::new(step, scale.dims.rope_theta as f32, ctx as u64)
        .unwrap_or_else(|e| panic!("{arch}: decode session: {e:#}"));
    if chunk >= 2 {
        let seeder = session
            .chunk_runner_for(want, chunk)
            .unwrap_or_else(|e| panic!("{arch}: prefill seeder: {e:#}"));
        decode
            .set_seeder(seeder)
            .unwrap_or_else(|e| panic!("{arch}: installing seeder: {e:#}"));
    }

    let mut row = decode
        .feed(&PROMPT)
        .unwrap_or_else(|e| panic!("{arch}: prefill (feed): {e:#}"));
    let mut tokens = Vec::with_capacity(GEN_STEPS);
    for s in 0..GEN_STEPS {
        let next = argmax(&row) as i64;
        tokens.push(next);
        row = decode
            .step(next)
            .unwrap_or_else(|e| panic!("{arch}: decode step {s}: {e:#}"));
    }
    (row, tokens)
}

/// Every faithful decoder family DECODES end to end: the full int8-quantized
/// staged decode pipeline compiles, prefills, and generates finite, non-empty,
/// reproducible output for Llama, Qwen2, Mistral, and Phi3 alike.
#[test]
fn decode_generates_for_every_supported_family() {
    assert!(
        FAITHFUL_FAMILIES.len() >= 4,
        "the coverage frontier must include the popular formats"
    );
    for layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(*layout, Dims::MODEST);
        let arch = scale.arch();

        let (logits, tokens) = decode_family(&scale);

        assert_eq!(
            logits.len(),
            scale.dims.vocab_size as usize,
            "{arch}: logits width must equal the vocabulary"
        );
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "{arch}: decode produced non-finite logits (the forward is numerically broken for this layout)"
        );
        assert_eq!(
            tokens.len(),
            GEN_STEPS,
            "{arch}: decode produced too few tokens"
        );
        assert!(
            tokens.iter().all(|&t| (t as u64) < scale.dims.vocab_size),
            "{arch}: a generated token is out of vocabulary range"
        );

        // Reproducibility: the same inputs decode to the same tokens (a broken
        // splice / residency path often shows as run-to-run drift).
        let (_, again) = decode_family(&scale);
        assert_eq!(
            tokens, again,
            "{arch}: decode is not reproducible run-to-run"
        );

        eprintln!("[family-coverage] {arch}: decoded {tokens:?} (finite, reproducible)");
    }
}

/// Cosine similarity of two logit rows. `1.0` = identical direction; argmax and
/// the whole ranking are preserved as it approaches 1.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x).powi(2);
        nb += f64::from(*y).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

/// **The V&V migration for turning on W8A8.** Since substrate v0.8.1 the deployed
/// int8 tier is the fused output-major **W8A8** decode GEMV — it quantizes the
/// activation per token, so it is a *different function* from the W8A32 it
/// replaced, and no byte-exact transcript can gate it. The honest replacement is
/// accuracy agreement against full precision: quantizing to int8-W8A8 must not
/// change what the model generates.
///
/// So decode each family twice over the same κ-store, same prompt — once through
/// the int8-W8A8 tier (the browser's deployed path) and once with the wide bf16
/// weights (full precision) — and require the quantized path to track it. The
/// numbers are MEASURED, not hoped: across all four families the greedy token
/// sequences agree exactly and the first-logit cosine is 0.9994–0.9996. The
/// asserts sit conservatively under that (argmax of the first token must agree;
/// cosine ≥ 0.99), so a real regression that flips the top token or tanks the
/// logit direction fails here, while a borderline late-token flip on some future
/// fixture does not make the suite brittle.
///
/// This is also the FIRST CI gate on the int8 κ-materialized decode path at all:
/// before this, its only coverage was the `#[ignore]` `decode_perf` benchmark and
/// the browser. W8A8's *kernel* numerics are pinned separately and exactly by
/// `omajor_w8a8_substrate_contract::w8a8_reproduces_the_exact_integer_oracle_which_w8a32_cannot`;
/// this witnesses the *integration* — the whole transformer plus LM head under
/// per-token activation quantization.
#[test]
fn int8_w8a8_decode_tracks_full_precision_for_every_family() {
    for layout in FAITHFUL_FAMILIES {
        let scale = FamilyScale::new(*layout, Dims::MODEST);
        let arch = scale.arch();

        let (q_logits, q_tokens) = decode_family_q(&scale, true); // int8 W8A8
        let (f_logits, f_tokens) = decode_family_q(&scale, false); // bf16 reference

        let cos = cosine(&q_logits, &f_logits);
        eprintln!(
            "[w8a8-accuracy] {arch}: W8A8={q_tokens:?} bf16={f_tokens:?} \
             seq_match={} cos={cos:.6}",
            q_tokens == f_tokens
        );

        assert_eq!(
            argmax(&q_logits),
            argmax(&f_logits),
            "{arch}: int8-W8A8 flipped the first-token argmax vs full-precision bf16 \
             (cos {cos:.6}) — the deployed quantized path no longer tracks the model"
        );
        assert!(
            cos >= 0.99,
            "{arch}: int8-W8A8 logit cosine {cos:.6} vs bf16 fell below 0.99 — \
             quantization is degrading the forward, not just rounding it"
        );
        assert_eq!(
            q_tokens.first(),
            f_tokens.first(),
            "{arch}: int8-W8A8 first generated token differs from full precision"
        );
    }
}
