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
use hologram_ai::quantized::crystallize_quantized;
use hologram_ai::staged::{quantizable_weights, GrowableStagedSession};
use hologram_ai::DecodeSession;
use hologram_ai_common::lower::QuantMap;
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
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    // κ-store: the wide bf16 weights (the download byte set).
    let store_dir = std::env::temp_dir().join(format!("hai-fam-{arch}-{}", std::process::id()));
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
    let eligible = quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
        .unwrap_or_else(|e| panic!("{arch}: quantizable_weights: {e:#}"));
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
    session.set_quant_map(quant);

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
