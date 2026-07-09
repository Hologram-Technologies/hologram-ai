//! Cross-model residency + warm draft-session reuse for catalogue-paired
//! speculative decode (row `speculative-draft-pairing`).
//!
//! A catalogue-paired DRAFT model is a SECOND `GrowableStagedSession`. These
//! witnesses nail the two native invariants the browser wiring rests on:
//!
//!  * SHARED LEDGER. [`GrowableStagedSession::share_residency_with`] makes the
//!    target and draft charge residency admission against ONE ledger, so their
//!    COMBINED footprint — not each session's alone — is what the budget bounds.
//!    Without it, two `bound_by_footprint` sessions each fill the budget and
//!    TOGETHER exceed it — the wasm 4 GiB address-space over-commit (a
//!    `RuntimeError: unreachable` allocation abort). The control demonstrates
//!    exactly that over-commit; the shared ledger prevents it.
//!  * WARM REUSE. [`ModelDrafter::into_session`] returns the draft decode
//!    session after a turn, so a warm caller reuses its resident pipeline across
//!    turns (the browser's `DecodeChatSession` does) instead of rebuilding it.
//!
//! Parametric in the family and small in scale (`Dims::MODEST`) — runs in the
//! default suite in seconds; no term is fit to a particular model or size.

mod common;

use common::families::{dummy_bf16_bytes, is_norm, Dims, FamilyScale};

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};

use hologram_ai::materialize::DirKappaStore;
use hologram_ai::quantized::{crystallize_quantized, crystallize_quantized_range};
use hologram_ai::speculative::{Drafter, ModelDrafter};
use hologram_ai::staged::{head_quant_chunks, quantizable_weights, GrowableStagedSession};
use hologram_ai::DecodeSession;
use hologram_ai_common::lower::{quant_key, QuantMap};
use hologram_ai_common::DType;

const PROMPT: [i64; 6] = [1, 2, 3, 5, 8, 13];
const GEN_STEPS: usize = 4;

/// A unique κ-store dir per built session: parallel tests (and the two sessions
/// of one test) must not race on content-addressed writes into a shared dir.
fn unique_store_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("hai-pairing-{tag}-{}-{n}", std::process::id()))
}

/// Build a full int8-quantized staged session for `scale` with `budget` resident
/// bytes as a HARD address ceiling (`bound_by_footprint`) — the browser's wasm
/// residency model. Returns the session (the archive factory, whose ledger the
/// runners charge) and the model context length.
fn build_session(scale: &FamilyScale, budget: u64) -> (GrowableStagedSession, usize) {
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    let store_dir = unique_store_dir(arch);
    std::fs::create_dir_all(&store_dir).expect("creating κ-store temp dir");
    let mut dir = DirKappaStore::new(&store_dir);
    let mut kappas: Vec<String> = Vec::with_capacity(keys.len());
    for (name, dims) in keys.iter().zip(&shapes) {
        let kappa = dir
            .insert(&dummy_bf16_bytes(name, dims))
            .unwrap_or_else(|e| panic!("{arch}: persisting weight {name}: {e:#}"));
        kappas.push(kappa);
    }

    // int8 derived-artifact tier (the browser's quantized path).
    let idx_of: HashMap<&str, usize> = kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let mut quant = QuantMap::new();
    for wide in quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
        .unwrap_or_else(|e| panic!("{arch}: quantizable_weights: {e:#}"))
    {
        let i = idx_of[wide.as_str()];
        let (out, inf) = (shapes[i][0], shapes[i][1]);
        assert!(
            !is_norm(&keys[i]),
            "{arch}: a norm weight is not a projection"
        );
        let entry = crystallize_quantized(&mut dir, &wide, DType::BF16, out, inf)
            .unwrap_or_else(|e| panic!("{arch}: crystallizing int8 artifact: {e:#}"));
        quant.insert(wide.clone(), entry);
    }
    for target in head_quant_chunks(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
        .unwrap_or_else(|e| panic!("{arch}: head_quant_chunks: {e:#}"))
    {
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
    session.set_residency_budget(budget);
    session.set_bound_by_footprint(true);
    session.set_quant_map(quant);
    let ctx = scale.dims.max_position_embeddings as usize;
    (session, ctx)
}

/// Wire a decode session over `session` and drive PROMPT + GEN_STEPS greedily,
/// so its stages materialize and accrue residency on `session`'s ledger. Returns
/// the driven decode session (holds the runner; keep it alive to keep the
/// residency it charged).
fn drive(
    session: &mut GrowableStagedSession,
    scale: &FamilyScale,
    ctx: usize,
) -> DecodeSession<hologram_ai::staged::StagedRunner<'static>> {
    let step = session
        .decode_runner_for(PROMPT.len())
        .unwrap_or_else(|e| panic!("decode step runner: {e:#}"));
    let mut decode = DecodeSession::new(step, scale.dims.rope_theta as f32, ctx as u64)
        .unwrap_or_else(|e| panic!("decode session: {e:#}"));
    let mut row = decode.feed(&PROMPT).expect("prefill");
    for _ in 0..GEN_STEPS {
        let next = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as i64)
            .unwrap_or(0);
        row = decode.step(next).expect("decode step");
    }
    decode
}

/// The shared ledger bounds the target+draft PAIR to the budget; the control
/// (two independent ledgers) over-commits.
#[test]
fn shared_ledger_bounds_the_model_pair() {
    let scale = FamilyScale::llama(Dims::MODEST);

    // One model's full resident footprint, measured with an unbounded budget.
    let (mut solo, ctx) = build_session(&scale, u64::MAX);
    let _held = drive(&mut solo, &scale, ctx);
    let single_fp = solo.peak_resident_footprint();
    assert!(single_fp > 0, "a driven model must hold some residency");

    // A budget that fits ONE model with headroom, but NOT two.
    let budget = single_fp + single_fp / 2;

    // SHARED: the draft adopts the target's ledger BEFORE it wires a runner, so
    // both charge one ledger and their combined footprint is bounded by `budget`.
    let (mut target, ctx) = build_session(&scale, budget);
    let _t = drive(&mut target, &scale, ctx);
    let (mut draft, dctx) = build_session(&scale, budget);
    target.share_residency_with(&mut draft);
    let _d = drive(&mut draft, &scale, dctx);

    let shared_peak = target.peak_resident_footprint();
    assert_eq!(
        shared_peak,
        draft.peak_resident_footprint(),
        "a shared ledger is ONE ledger — both sessions must read the identical peak"
    );
    assert!(
        shared_peak <= budget,
        "the shared ledger must bound the PAIR to the budget: peak {shared_peak} > budget {budget}"
    );

    // CONTROL: independent ledgers each fill the budget, so their real
    // simultaneous residency exceeds it — the over-commit the shared ledger
    // prevents. (Both sessions stay alive, so both hold their residency at once.)
    let (mut t2, ctx) = build_session(&scale, budget);
    let _t2 = drive(&mut t2, &scale, ctx);
    let (mut d2, dctx) = build_session(&scale, budget);
    let _d2 = drive(&mut d2, &scale, dctx);
    let unshared_sum = t2.peak_resident_footprint() + d2.peak_resident_footprint();
    assert!(
        unshared_sum > budget,
        "independent ledgers over-commit: two models resident at once is {unshared_sum} > budget \
         {budget} — this is exactly what share_residency_with must prevent"
    );
}

/// `ModelDrafter::into_session` reclaims the draft session warm: after a
/// propose/commit cycle the reclaimed session continues stepping, its realized
/// sequence intact and its resident pipeline unrebuilt.
#[test]
fn model_drafter_reclaims_a_warm_session() {
    let scale = FamilyScale::llama(Dims::MODEST);
    let (mut session, ctx) = build_session(&scale, u64::MAX);
    let runner = session
        .decode_runner_for(PROMPT.len())
        .expect("draft runner");
    let draft_session = DecodeSession::new(runner, scale.dims.rope_theta as f32, ctx as u64)
        .expect("draft decode session");

    let mut drafter = ModelDrafter::new(draft_session);
    drafter.prefill(&PROMPT).expect("draft prefill");
    let proposal = drafter.propose(&PROMPT, 3).expect("draft propose");
    assert_eq!(
        proposal.len(),
        3,
        "the drafter proposes exactly `cap` tokens"
    );
    // Accept 1 of 3, commit a bonus — the loop's sync after partial acceptance.
    drafter.commit(1, 42).expect("draft commit");

    // Reclaim the warm session and keep using it: the reclaimed session's
    // realized sequence is intact and its resident pipeline is unrebuilt.
    let mut reclaimed = drafter.into_session();
    assert_eq!(
        reclaimed.realized_len(),
        PROMPT.len() + 2,
        "prompt ({}) + 1 accepted + 1 bonus committed",
        PROMPT.len()
    );
    let row = reclaimed
        .step(7)
        .expect("the reclaimed warm session still steps");
    assert!(
        row.iter().all(|x| x.is_finite()),
        "warm step yields finite logits"
    );
    assert!(
        reclaimed.runner().materialization_count() > 0,
        "into_session keeps the RESIDENT pipeline — a rebuilt-from-scratch session \
         would report zero materializations"
    );
}
