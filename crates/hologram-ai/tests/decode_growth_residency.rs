//! Bucket-growth residency (row `lazy-constant-residency`): a decode session
//! that outgrows its bucket must FREE the outgoing runner's resident stages
//! BEFORE it compiles and materializes the wider bucket — never hold both at
//! once.
//!
//! WHY IT MATTERS (a real deployed crash): Qwen2.5-1.5B decoded ~10 tokens fine
//! at bucket 64, then aborted with a bare `RuntimeError: unreachable` the instant
//! the sequence crossed 64 and the session grew to bucket 128. That is a
//! `memory.grow` past the wasm 4 GiB ceiling — the grow-only linear memory was
//! asked to hold the old runner's full resident set (~1.9 GB of int8 stages)
//! AND the new bucket's stage compilation at once. The residency ledger cannot
//! foresee it because compilation / module memory is not stage-weight residency;
//! the structural fix is to free the outgoing runner first so the allocator
//! reuses that space for the new bucket.
//!
//! This witnesses the invariant natively: at every bucket regrowth the growable's
//! CURRENT resident footprint is zero at the moment the wider bucket is built —
//! the outgoing runner (and the stale seeder) were freed first. Parametric in the
//! family, small in scale; no term is fit to a model or size.

mod common;

use common::families::{dummy_bf16_bytes, Dims, FamilyScale};

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use hologram_ai::materialize::DirKappaStore;
use hologram_ai::staged::GrowableStagedSession;
use hologram_ai::{DecodeSession, RopeSpec};
use hologram_ai_common::DType;

fn unique_store_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("hai-growth-{tag}-{}-{n}", std::process::id()))
}

/// A footprint-bounded (wasm-regime) staged session over wide bf16 weights — no
/// quant tier needed; the growth-residency law is dtype-independent.
fn build_session(scale: &FamilyScale) -> GrowableStagedSession {
    let arch = scale.arch();
    let config_json = scale.config_json();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = scale.manifest().into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    let store_dir = unique_store_dir(arch);
    std::fs::create_dir_all(&store_dir).expect("creating κ-store temp dir");
    let dir = DirKappaStore::new(&store_dir);
    let mut kappas: Vec<String> = Vec::with_capacity(keys.len());
    for (name, dims) in keys.iter().zip(&shapes) {
        kappas.push(
            dir.insert(&dummy_bf16_bytes(name, dims))
                .unwrap_or_else(|e| panic!("{arch}: persisting {name}: {e:#}")),
        );
    }

    let mut session = GrowableStagedSession::new(
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        None,
        one,
        Box::new(DirKappaStore::new(&store_dir)),
    )
    .unwrap_or_else(|e| panic!("{arch}: growable session: {e:#}"));
    // Generous budget so stages stay resident (footprint > 0 before growth), but
    // the HARD-ceiling regime so eviction returns the footprint to the ledger.
    session.set_residency_budget(u64::MAX);
    session.set_bound_by_footprint(true);
    session
}

#[test]
fn growth_frees_the_outgoing_runner_before_rebuilding() {
    let scale = FamilyScale::llama(Dims::MODEST);
    let ctx = scale.dims.max_position_embeddings as usize;
    let session = Rc::new(RefCell::new(build_session(&scale)));

    // Record the growable's CURRENT resident footprint at each rebuild — the
    // moment the wider bucket is compiled. The fix makes this zero.
    let at_rebuild: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));

    let step = session
        .borrow_mut()
        .decode_runner_for(1)
        .expect("initial step runner (bucket = MIN_WINDOW)");
    let g = Rc::clone(&session);
    let rec = Rc::clone(&at_rebuild);
    let mut decode = DecodeSession::new(
        step,
        RopeSpec::plain(scale.dims.rope_theta as f32),
        ctx as u64,
    )
    .expect("decode session")
    .with_rebuild(Box::new(move |bucket| {
        // Read BEFORE building the wider runner — sequential borrows.
        rec.borrow_mut().push(g.borrow().resident_footprint());
        g.borrow_mut().decode_runner_for(bucket as usize)
    }));

    let bucket0 = decode.geometry().bucket;
    // Prefill a couple of tokens so the step runner materializes (footprint > 0),
    // then step past the initial bucket to force at least one growth.
    decode.feed(&[1, 2]).expect("prefill");
    assert!(
        session.borrow().resident_footprint() > 0,
        "the step runner must be resident before growth (else the test proves nothing)"
    );
    for t in 0..(bucket0 as i64 + 4) {
        decode.step((t % 7) + 1).expect("decode step");
    }

    let recorded = at_rebuild.borrow();
    assert!(
        !recorded.is_empty(),
        "stepping past bucket {bucket0} must have grown the bucket at least once"
    );
    for (i, &footprint) in recorded.iter().enumerate() {
        assert_eq!(
            footprint, 0,
            "growth #{i}: the outgoing runner's residency must be FREED before the wider \
             bucket is compiled — {footprint} bytes were still resident (the old resident set \
             + the new bucket's compilation is the wasm 4 GiB over-commit that aborts growth)"
        );
    }
    eprintln!(
        "[growth-residency] {} growth(s), footprint at rebuild always 0 (outgoing runner freed first)",
        recorded.len()
    );
}

/// Every growth — not just the first — frees ALL auxiliary residency: the
/// outgoing step runner AND the prefill seeder, across MULTIPLE consecutive
/// growths (64 → 128 → 256). This exercises the seeder-drop path (a warm turn's
/// step runner and seeder are both resident until growth) and proves the handoff
/// holds at every bucket transition, not only the first.
#[test]
fn every_growth_frees_the_step_runner_and_the_seeder() {
    // One layer keeps ~130 single-position steps fast while still staging.
    let scale = FamilyScale::llama(Dims::MODEST.with_layers(1));
    let ctx = scale.dims.max_position_embeddings as usize;
    let session = Rc::new(RefCell::new(build_session(&scale)));
    let at_rebuild: Rc<RefCell<Vec<u64>>> = Rc::new(RefCell::new(Vec::new()));

    let step = session
        .borrow_mut()
        .decode_runner_for(1)
        .expect("initial step runner");
    let g = Rc::clone(&session);
    let rec = Rc::clone(&at_rebuild);
    let mut decode = DecodeSession::new(
        step,
        RopeSpec::plain(scale.dims.rope_theta as f32),
        ctx as u64,
    )
    .expect("decode session")
    .with_rebuild(Box::new(move |bucket| {
        rec.borrow_mut().push(g.borrow().resident_footprint());
        g.borrow_mut().decode_runner_for(bucket as usize)
    }));

    // Install a prefill SEEDER (chunk > 1) so BOTH a step runner and a seeder are
    // resident before the first growth — growth must free both.
    let bucket0 = decode.geometry().bucket;
    let chunk = (hologram_ai::engine::geometric_window(1, ctx) as u64).min(bucket0 as u64);
    assert!(
        chunk >= 2,
        "the seeder chunk must batch more than one position"
    );
    let seeder = session
        .borrow_mut()
        .chunk_runner_for(1, chunk)
        .expect("prefill seeder");
    decode.set_seeder(seeder).expect("install the seeder");

    decode
        .feed(&[1, 2, 3])
        .expect("prefill materializes step + seeder");
    assert!(
        session.borrow().resident_footprint() > 0,
        "step runner and seeder must be resident before growth"
    );

    // Step past TWO bucket boundaries (64 → 128 → 256).
    for t in 0..(bucket0 as i64 * 2 + 8) {
        decode.step((t % 7) + 1).expect("decode step");
    }

    let recorded = at_rebuild.borrow();
    assert!(
        recorded.len() >= 2,
        "expected at least two growths (64 → 128 → 256), saw {}",
        recorded.len()
    );
    for (i, &footprint) in recorded.iter().enumerate() {
        assert_eq!(
            footprint, 0,
            "growth #{i}: ALL auxiliary residency (step runner + seeder) must be freed before \
             the wider bucket is compiled — {footprint} bytes still resident"
        );
    }
    eprintln!(
        "[growth-residency] {} growths with a seeder installed, footprint at rebuild always 0",
        recorded.len()
    );
}

/// A DECLARED generation budget sizes the first bucket for the WHOLE turn, so
/// the turn never regrows mid-way — the full-stage re-materialization the
/// deployed log paid for. Scale-free: the rule is the same relation between
/// prompt, declared budget, and context at any magnitude (the numbers here are
/// only small enough to run fast). An UNDECLARED budget keeps the geometric
/// ladder — witnessed by the growth tests above, which start at the prompt's
/// window and cross it.
#[test]
fn a_declared_generation_budget_never_regrows_mid_turn() {
    const GEN_BUDGET: usize = 30;
    let scale = FamilyScale::llama(Dims::MODEST.with_layers(1));
    let ctx = scale.dims.max_position_embeddings as usize;
    let session = Rc::new(RefCell::new(build_session(&scale)));
    let grew = Rc::new(RefCell::new(false));

    // The prompt ALONE would size a 64 bucket, which `GEN_BUDGET` more tokens
    // would cross; declaring the budget sizes one 128 bucket for the whole turn.
    let prompt: Vec<i64> = (0..60).map(|t| (t % 7) + 1).collect();
    assert_eq!(hologram_ai::engine::geometric_window(prompt.len(), ctx), 64);
    let want = hologram_ai::engine::decode_bucket_for_turn(prompt.len(), GEN_BUDGET, ctx);
    assert_eq!(
        want, 128,
        "prompt + declared budget sizes one bucket up front"
    );

    let step = session
        .borrow_mut()
        .decode_runner_for(want)
        .expect("budget-sized step runner");
    let g = Rc::clone(&session);
    let flag = Rc::clone(&grew);
    let mut decode = DecodeSession::new(
        step,
        RopeSpec::plain(scale.dims.rope_theta as f32),
        ctx as u64,
    )
    .expect("decode session")
    .with_rebuild(Box::new(move |bucket| {
        *flag.borrow_mut() = true;
        g.borrow_mut().decode_runner_for(bucket as usize)
    }));

    decode.feed(&prompt).expect("prefill");
    for t in 0..GEN_BUDGET as i64 {
        decode.step((t % 7) + 1).expect("decode step");
    }
    assert!(
        !*grew.borrow(),
        "a turn generating within its DECLARED budget must never regrow (no re-materialization)"
    );
    eprintln!("[growth-residency] declared budget {GEN_BUDGET} sized one {want} bucket; no regrow");
}
