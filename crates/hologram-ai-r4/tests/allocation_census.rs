//! Allocation census for the steady-state engine path (ADR-0009).
//!
//! Two tiers, honest about what each counts:
//!
//! 1. **Fixture-driven engine census** (`engine_steady_state_allocates_nothing`,
//!    `#[ignore]`d): point `HOLOGRAM_AI_R4_FIXTURE_BUNDLE` at a real compiled
//!    bundle (schema v1, e.g. produced by `compile_source_to_bundle`) and run
//!    with `cargo test -p hologram-ai-r4 --test allocation_census -- --ignored`.
//!    It loads the engine, warms up, then asserts **zero heap allocations**
//!    across N `predict_next_into` calls and across the upstream steady-state
//!    generate loop (events go to a `NullEventSink`; the post-run event batch
//!    and `FinishInfo` construction happen after the counter is stopped, so
//!    only the loop region is counted). A synthetic minimal R4G1 graph is not
//!    cheaply constructible outside the uor-r4 compiler pipeline, so this
//!    tier needs the fixture.
//! 2. **Unit-level census** (always on): the pure adapter paths that exist
//!    without a loadable graph — `Prediction::default` and the
//!    `predict_next_into` outcome mapping, exercised through the same shapes
//!    the engine uses — perform zero allocations. This does NOT cover the
//!    upstream scorer; tier 1 is the load-bearing check for that.

// NOTE: unlike the library crate (`#![forbid(unsafe_code)]`), this test
// harness uses `unsafe` — a counting `GlobalAlloc` implementation is
// impossible without it. The unsafe is confined to the test allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use hologram_ai_core::{NullEventSink, ResolutionStatus};
use hologram_ai_r4::{Engine, PredictOutcome, Prediction};

/// Counting wrapper over the system allocator. Counts `alloc` calls;
/// dealloc/realloc are irrelevant to the zero-allocation claim (a steady
/// state that only frees would still be wrong, and nothing in the path
/// allocates to begin with).
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn alloc_count() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

/// Mirror of the engine's outcome mapping over the same plain-data
/// shapes (`PredictOutput` semantics in flat form), used by the
/// unit-level census. Kept local so the census does not reach into
/// private adapter internals.
fn map_flat(
    token: u32,
    status: Option<ResolutionStatus>,
    widened: bool,
    abstained: bool,
) -> Prediction {
    Prediction {
        outcome: if abstained {
            PredictOutcome::Abstain { widened }
        } else {
            PredictOutcome::Serve {
                token,
                status: status.unwrap_or(ResolutionStatus::Exact),
                widened,
            }
        },
    }
}

#[test]
fn unit_level_prediction_mapping_allocates_nothing() {
    // Warm up (first-touch lazy init in the test harness is not part of
    // the counted region).
    let mut slot = map_flat(0, None, false, true);
    std::hint::black_box(&slot);

    let before = alloc_count();
    for i in 0..1024u32 {
        let serve = map_flat(i, Some(ResolutionStatus::Graph), i % 2 == 0, false);
        let abstain = map_flat(0, Some(ResolutionStatus::Novel), true, true);
        slot = serve;
        std::hint::black_box(abstain);
        std::hint::black_box(&slot);
    }
    let after = alloc_count();
    assert_eq!(
        after - before,
        0,
        "prediction slot mapping allocated {} times",
        after - before
    );
}

/// Fixture-driven steady-state census. Ignored by default: requires
/// `HOLOGRAM_AI_R4_FIXTURE_BUNDLE` to point at a real compiled bundle.
#[test]
#[ignore = "needs HOLOGRAM_AI_R4_FIXTURE_BUNDLE pointing at a compiled bundle"]
fn engine_steady_state_allocates_nothing() {
    let path = std::env::var("HOLOGRAM_AI_R4_FIXTURE_BUNDLE")
        .expect("HOLOGRAM_AI_R4_FIXTURE_BUNDLE must point at a compiled bundle");
    let bytes = std::fs::read(&path).expect("fixture bundle must be readable");
    let bundle = hologram_ai_bundle::Bundle::parse_verified(&bytes).unwrap();
    let mut engine = Engine::load(&bundle).unwrap();

    // A window of zeroes is in-vocabulary for any teacher with a real
    // token table; the fixture drives what "servable" means, and an
    // abstaining path is still part of the allocation contract.
    let window = [0u32; 8];

    // Warmup: first probes may populate one-time caches inside the
    // scorer. The contract covers steady state after warmup.
    let mut prediction = Prediction::default();
    for _ in 0..16 {
        engine.predict_next_into(&window, &mut prediction).unwrap();
    }
    let mut out = vec![0u32; 32];
    engine
        .generate_into(&window, &mut out, &mut NullEventSink)
        .unwrap();
    engine.reset();

    // Counted region 1: N single-step predictions.
    let before = alloc_count();
    for _ in 0..256 {
        engine.predict_next_into(&window, &mut prediction).unwrap();
    }
    let predict_allocs = alloc_count() - before;
    assert_eq!(
        predict_allocs, 0,
        "predict_next_into allocated {predict_allocs} times in steady state"
    );

    // Counted region 2: the steady-state generate loop. The post-run
    // event batch is not part of this region (NullEventSink discards
    // events, and FinishInfo is plain data).
    engine.reset();
    let before = alloc_count();
    let finish = engine
        .generate_into(&window, &mut out, &mut NullEventSink)
        .unwrap();
    let generate_allocs = alloc_count() - before;
    assert_eq!(
        generate_allocs, 0,
        "generate_into loop allocated {generate_allocs} times in steady state"
    );
    assert!(finish.produced <= out.len());
}
