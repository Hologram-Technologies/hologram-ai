//! PV — peak-memory reproduction of the browser (wasm) decode pipeline at
//! multi-billion-parameter *config scale*, swept ACROSS decoder families and
//! driven NATIVELY so the suite can see what busts the wasm 4 GiB
//! address-space ceiling — parametric in the family AND the scale, fit to no
//! single model.
//!
//! The browser (wasm32) aborts with `RuntimeError: unreachable` — an allocation
//! abort — on the FIRST DECODE STEP of a large model: all staged decode stages
//! materialize, one token emits, then wasm memory cannot grow past 4 GiB.
//! Fixture-scale conformance tests (hidden 64, 2 layers) never approach that
//! ceiling, so the suite is blind to it. This harness drives the SAME pipeline
//! (a `GrowableStagedSession` with `residency_budget = 3 GiB` and
//! `bound_by_footprint(true)`, an int8-quantized decode step runner, a
//! chunked-prefill seeder sharing the session's one residency ledger) at a
//! representative large config — over EACH faithful family (Llama, Qwen2,
//! Mistral, Phi3: tied vs untied head, fused vs separate qkv/gate-up, with and
//! without q/k/v bias) — and MEASURES the true peak for each:
//!
//!   * `rss_bytes()` — process RSS from `/proc/self/statm`, sampled at every
//!     stage materialization and every decode step (materialization is the
//!     spike). This is the honest peak the box must hold.
//!   * `GrowableStagedSession::peak_resident_footprint()` — what the shared
//!     residency ledger *thinks* is resident (resident stage weights + retained
//!     transients). It does NOT track the compiled stage modules.
//!   * cumulative heap-allocation volume during the decode region (the counting
//!     allocator), a secondary signal.
//!
//! The DELTA (`rss − resident_footprint`) is the memory the ledger does not
//! track: the compiled stage archives (2 runners × N stage modules), the
//! retained F32 head scratch, activations, and the runtime. Whether the peak
//! busts 4 GiB because of *resident weights* (the shared-ledger fix should bound
//! these to ~3 GiB) or because of the *untracked* module/scratch memory is the
//! question this harness answers with numbers.
//!
//! Gated behind `HOLOGRAM_AI_LARGE=1` (the byte/compile cost is large). Run:
//!   HOLOGRAM_AI_LARGE=1 cargo test --release -p hologram-ai \
//!     --test decode_memory_large -- --nocapture --test-threads=1
//!
//! `HOLOGRAM_AI_DECODE_LAYERS` (default `4`) sets the transformer-layer count —
//! the scale knob applied to the shared `Dims::LARGE` template (no per-model
//! numbers). Set `HOLOGRAM_AI_DECODE_LAYERS=28` for a full large config, or a
//! comma-separated sweep `HOLOGRAM_AI_DECODE_LAYERS=4,8,16,28` to print the
//! memory-vs-scale trend and locate the 4 GiB crossing. `HOLOGRAM_AI_FAMILIES`
//! (default all four) restricts which families are swept.

mod common;

use common::families::{dummy_bf16_bytes, is_norm, Dims, FamilyScale, FAITHFUL_FAMILIES};

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::time::Instant;

use hologram_ai::materialize::DirKappaStore;
use hologram_ai::quantized::crystallize_quantized;
use hologram_ai::staged::{quantizable_weights, GrowableStagedSession};
use hologram_ai::{DecodeSession, RopeSpec};
use hologram_ai_common::lower::QuantMap;
use hologram_ai_common::DType;

// A counting global allocator (modelled on `hologram-ai-conformance`'s ZA
// harness `alloc.rs`, which is gated behind its `structural` feature — inlined
// here so this test stays self-contained). Installed as this binary's global
// allocator, so the decode region's cumulative heap-allocation volume is
// measurable. A #[global_allocator] can be set once per test binary; this is
// the only one here. The byte counter is CUMULATIVE (dealloc does not decrement
// it), so it is an allocation-VOLUME signal, not a live peak — rss_bytes() is
// the peak.
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct CountingAllocator {
        inner: System,
    }
    impl CountingAllocator {
        pub const fn new() -> Self {
            Self { inner: System }
        }
    }

    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            self.inner.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            self.inner.dealloc(ptr, layout)
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOC_BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
            self.inner.realloc(ptr, layout, new_size)
        }
    }

    #[derive(Clone, Copy, Default)]
    pub struct AllocStats {
        pub allocations: usize,
        pub bytes: usize,
    }

    pub fn reset() {
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        ALLOC_BYTES.store(0, Ordering::Relaxed);
    }
    pub fn snapshot() -> AllocStats {
        AllocStats {
            allocations: ALLOC_COUNT.load(Ordering::Relaxed),
            bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        }
    }
    /// Guard against a vacuous measurement: prove the allocator is actually the
    /// installed global (a capacity Vec must move the counters).
    pub fn assert_allocator_installed() {
        reset();
        let mut v = Vec::<u8>::with_capacity(std::hint::black_box(64));
        v.push(std::hint::black_box(0));
        let _ = std::hint::black_box(&v);
        assert!(
            snapshot().allocations > 0,
            "the counting allocator is not installed as #[global_allocator]"
        );
    }
}

#[global_allocator]
static GLOBAL: counting::CountingAllocator = counting::CountingAllocator::new();

/// One per-stage materialization sample: `(stage, stage_count, weight_bytes, rss_bytes)`.
type StageSample = (usize, usize, u64, u64);

const GIB: f64 = (1u64 << 30) as f64;
const STRUCTURAL_CEILING: u64 = 4 << 30; // wasm32 address space
const RUNTIME_RESERVE: u64 = 1 << 30; // activations + K/V + runtime
const RESIDENCY_BUDGET: u64 = STRUCTURAL_CEILING - RUNTIME_RESERVE; // 3 GiB

/// Process resident set size (bytes), from `/proc/self/statm` — the coarse but
/// honest measure of the peak memory the box holds. Same instrument as
/// `perf_contract_large.rs`.
fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|p| p.parse::<u64>().ok())
        })
        .map(|pages| pages * 4096)
        .unwrap_or(0)
}

fn mem_available_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

// ─────────────────────────────── measurement ─────────────────────────────────

struct ScaleResult {
    arch: &'static str,
    layers: u64,
    total_params: u64,
    manifest_tensors: usize,
    wide_weight_bytes: u64, // bf16 weights on disk (the download's byte set)
    artifact_bytes: u64,    // int8+scales derived-artifact bytes on disk
    quantized_weights: usize,
    stage_count: usize, // decode-step pipeline stage modules
    seeder_stage_count: usize,
    module_bytes: u64,       // compiled archive bytes (step + seeder runners)
    rss_baseline: u64,       // before building the runners
    rss_after_build: u64,    // runners built, weights not yet materialized
    rss_peak: u64,           // over the whole feed + decode steps
    resident_footprint: u64, // session.peak_resident_footprint()
    step_runner_peak_weight: u64,
    decode_alloc_bytes: u64, // cumulative heap alloc during feed + steps
    decode_alloc_count: usize,
    generated: Vec<i64>,
    per_stage_rss: Vec<(usize, usize, u64, u64)>, // (stage, count, weight_bytes, rss)
}

/// Build `scale` (a decoder family at a chosen scale), compile the int8-
/// quantized staged decode pipeline (the browser path), drive a short prefill +
/// a few decode steps under the 3 GiB footprint-bounded ledger, and measure the
/// peak. Parametric in the family and the layer count.
fn measure_scale(scale: FamilyScale) -> ScaleResult {
    let arch = scale.arch();
    let layers = scale.dims.layers;
    let config_json = scale.config_json();
    let manifest = scale.manifest();
    let (keys, shapes): (Vec<String>, Vec<Vec<u64>>) = manifest.into_iter().unzip();
    let dtypes = vec![DType::BF16; keys.len()];
    let one = NonZeroU64::new(1).expect("1 is non-zero");

    let total_params: u64 = shapes.iter().map(|d| d.iter().product::<u64>()).sum();
    eprintln!(
        "\n╔═ {arch}  layers={layers}  ({} tensors, {:.3}B params) ═╗",
        keys.len(),
        total_params as f64 / 1e9
    );

    // ── κ-store: seed the wide bf16 weights (the download's byte set) ──────────
    let store_dir = std::env::temp_dir().join(format!(
        "hai-decode-mem-{}-{arch}-l{layers}",
        std::process::id()
    ));
    std::fs::create_dir_all(&store_dir).expect("creating κ-store temp dir");
    let mut dir = DirKappaStore::new(&store_dir);

    let t = Instant::now();
    let mut kappas: Vec<String> = Vec::with_capacity(keys.len());
    let mut wide_weight_bytes: u64 = 0;
    for (name, dims) in keys.iter().zip(&shapes) {
        let bytes = dummy_bf16_bytes(name, dims);
        wide_weight_bytes += bytes.len() as u64;
        let kappa = dir.insert(&bytes).expect("persisting a dummy weight");
        kappas.push(kappa);
        // `bytes` dropped here — seeding must not retain the whole weight set.
    }
    // Distinctness guard on the LOAD-BEARING weights: every non-norm tensor
    // (embed + projections + biases — the terms that dominate memory) must own a
    // distinct κ, else content-address dedup would under-count resident memory
    // (see `name_seed`). The norm weights are intentionally the constant 1.0 and
    // DO dedup to one κ — a few KiB, immaterial — so they are excluded.
    let big: Vec<&String> = keys
        .iter()
        .zip(&kappas)
        .filter(|(name, _)| !is_norm(name))
        .map(|(_, k)| k)
        .collect();
    let distinct: std::collections::HashSet<&&String> = big.iter().collect();
    assert_eq!(
        distinct.len(),
        big.len(),
        "load-bearing weights deduplicated ({} distinct κ for {} non-norm tensors) — \
         memory would be under-counted",
        distinct.len(),
        big.len()
    );
    eprintln!(
        "  seeded {} wide bf16 weights ({:.3} GiB on disk) in {:?}",
        keys.len(),
        wide_weight_bytes as f64 / GIB,
        t.elapsed()
    );

    // ── quantized derived-artifact tier (the browser's int8 path) ─────────────
    // `quantizable_weights` names the wide κs the staged plan can rewrite onto
    // int8 artifacts and fully retire; the tied embedding / chunked head stay
    // wide. For each, `crystallize_quantized` resolves the wide bf16, derives the
    // matmul-ready per-channel int8 artifact, and persists it — minting its κ.
    let t = Instant::now();
    let idx_of: std::collections::HashMap<&str, usize> = kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let eligible = quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, one)
        .expect("quantizable_weights over the Qwen2 config");
    let mut quant = QuantMap::new();
    let mut artifact_bytes: u64 = 0;
    for wide_kappa in &eligible {
        let i = *idx_of
            .get(wide_kappa.as_str())
            .expect("eligible κ is in the manifest");
        let shape = &shapes[i];
        assert_eq!(shape.len(), 2, "a quantizable weight is a 2-D projection");
        let (out, inf) = (shape[0], shape[1]);
        let entry = crystallize_quantized(&mut dir, wide_kappa, DType::BF16, out, inf)
            .expect("crystallizing the int8 artifact");
        // Artifact layout is `q_i8(out·in) ‖ scales_f32(4·out)` — its byte size
        // is exactly this, no need to resolve (and re-allocate) it.
        artifact_bytes += out * inf + out * 4;
        quant.insert(wide_kappa.clone(), entry);
    }
    eprintln!(
        "  crystallized {} int8 artifacts ({:.3} GiB on disk) in {:?}",
        quant.len(),
        artifact_bytes as f64 / GIB,
        t.elapsed()
    );

    // ── compiled-module byte count: exactly the archives the session will build
    //    for the decode step runner and the chunked-prefill seeder. Weightless
    //    k-forms, but this is the "34 stage modules × 2 runners" the wasm engine
    //    compiles into executable modules — the ledger does not track them. ─────
    let ctx = scale.dims.max_position_embeddings as usize;
    let want = 16usize; // short prompt
    let bucket = hologram_ai::engine::geometric_window(want.max(1), ctx);
    let chunk = (hologram_ai::engine::geometric_window(1, ctx) as u64).min(bucket as u64);

    let t = Instant::now();
    let step_archives = hologram_ai::staged::compile_decode_stages(
        &config_json,
        &keys,
        &kappas,
        &shapes,
        &dtypes,
        bucket as u64,
        one,
        Some(&quant),
    )
    .expect("decode step stages compile");
    let seeder_archives = hologram_ai::staged::compile_chunk_stages(
        &config_json,
        &keys,
        &kappas,
        &shapes,
        &dtypes,
        bucket as u64,
        chunk,
        one,
        Some(&quant),
    )
    .expect("chunk seeder stages compile");
    let stage_count = step_archives.len();
    let seeder_stage_count = seeder_archives.len();
    let module_bytes: u64 = step_archives.iter().map(|a| a.len() as u64).sum::<u64>()
        + seeder_archives.iter().map(|a| a.len() as u64).sum::<u64>();
    eprintln!(
        "  compiled decode modules: step {stage_count} stages + seeder {seeder_stage_count} stages \
         = {:.1} MiB archives (bucket {bucket}, chunk {chunk}) in {:?}",
        module_bytes as f64 / (1u64 << 20) as f64,
        t.elapsed()
    );
    drop(step_archives);
    drop(seeder_archives);

    // ── the browser session: 3 GiB footprint-bounded shared ledger, int8 map ──
    let mut session = GrowableStagedSession::new(
        config_json.clone(),
        keys.clone(),
        kappas.clone(),
        shapes.clone(),
        dtypes.clone(),
        None, // context = the model's own max_position_embeddings
        one,
        Box::new(DirKappaStore::new(&store_dir)),
    )
    .expect("the growable staged session builds");
    session.set_residency_budget(RESIDENCY_BUDGET);
    session.set_bound_by_footprint(true);
    session.set_quant_map(quant);

    // Per-stage RSS sampling: materialization is where memory spikes. The
    // observer fires once per stage as it materializes (then stays resident).
    let per_stage: Rc<RefCell<Vec<StageSample>>> = Rc::new(RefCell::new(Vec::new()));
    let peak: Rc<RefCell<u64>> = Rc::new(RefCell::new(0));
    {
        let per_stage = Rc::clone(&per_stage);
        let peak = Rc::clone(&peak);
        session.set_stage_observer(Box::new(move |stage, count, weight_bytes| {
            let rss = rss_bytes();
            if rss > *peak.borrow() {
                *peak.borrow_mut() = rss;
            }
            per_stage
                .borrow_mut()
                .push((stage, count, weight_bytes, rss));
        }));
    }

    let rss_baseline = rss_bytes();
    *peak.borrow_mut() = rss_baseline;

    // Build the decode step runner (chunk 1) at the 64-token bucket, then the
    // chunked-prefill seeder over the SAME bucket — both wired by this session,
    // so they share its ONE address-space residency ledger (the shared-ledger
    // fix). This is exactly what the browser holds at once on the first turn.
    let step = session
        .decode_runner_for(want)
        .expect("the decode step runner builds");
    let mut decode = DecodeSession::new(
        step,
        RopeSpec::plain(scale.dims.rope_theta as f32),
        ctx as u64,
    )
    .expect("the decode session opens");
    if chunk >= 2 {
        let seeder = session
            .chunk_runner_for(want, chunk)
            .expect("the chunked-prefill seeder builds");
        decode.set_seeder(seeder).expect("the seeder installs");
    }
    let rss_after_build = rss_bytes();
    if rss_after_build > *peak.borrow() {
        *peak.borrow_mut() = rss_after_build;
    }

    // A short prompt of in-vocabulary dummy token ids, then a few decode steps.
    // The FIRST feed materializes every stage (the spike the browser aborts on);
    // subsequent steps run warm off the resident set.
    let toks: Vec<i64> = (0..want as u64)
        .map(|i| ((i * 2_654_435_761) % scale.dims.vocab_size) as i64)
        .collect();

    counting::reset();
    let mut row = decode
        .feed(&toks)
        .expect("the prompt prefills (first decode materialization)");
    {
        let rss = rss_bytes();
        if rss > *peak.borrow() {
            *peak.borrow_mut() = rss;
        }
    }
    let mut generated = Vec::new();
    for _ in 0..4 {
        let n = argmax(&row) as i64;
        generated.push(n);
        row = decode.step(n).expect("the decode session steps");
        let rss = rss_bytes();
        if rss > *peak.borrow() {
            *peak.borrow_mut() = rss;
        }
    }
    let decode_alloc = counting::snapshot();

    let resident_footprint = session.peak_resident_footprint();
    let step_runner_peak_weight = decode.runner().peak_resident_weight_bytes();
    let rss_peak = *peak.borrow();
    let per_stage_rss = per_stage.borrow().clone();

    // Drop the session (and its runners) before cleaning up the store.
    drop(decode);
    drop(session);
    let _ = std::fs::remove_dir_all(&store_dir);

    ScaleResult {
        arch,
        layers,
        total_params,
        manifest_tensors: keys.len(),
        wide_weight_bytes,
        artifact_bytes,
        quantized_weights: eligible.len(),
        stage_count,
        seeder_stage_count,
        module_bytes,
        rss_baseline,
        rss_after_build,
        rss_peak,
        resident_footprint,
        step_runner_peak_weight,
        decode_alloc_bytes: decode_alloc.bytes as u64,
        decode_alloc_count: decode_alloc.allocations,
        generated,
        per_stage_rss,
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

fn report(r: &ScaleResult) {
    eprintln!(
        "\n── layers={} · {:.3}B params · {} tensors · {} int8 artifacts ──",
        r.layers,
        r.total_params as f64 / 1e9,
        r.manifest_tensors,
        r.quantized_weights
    );
    eprintln!("  decode step pipeline : {} stages", r.stage_count);
    eprintln!("  prefill seeder       : {} stages", r.seeder_stage_count);
    eprintln!(
        "  wide bf16 weights    : {:.3} GiB (download byte set)",
        gib(r.wide_weight_bytes)
    );
    eprintln!("  int8 artifacts       : {:.3} GiB", gib(r.artifact_bytes));
    eprintln!(
        "  compiled modules     : {:.1} MiB archives (step + seeder, 2 runners)",
        r.module_bytes as f64 / (1u64 << 20) as f64
    );
    eprintln!("  generated tokens     : {:?}", r.generated);

    // The per-stage RSS trace of the FIRST feed — where the spike is.
    if !r.per_stage_rss.is_empty() {
        let count = r.per_stage_rss[0].1;
        eprintln!("  per-stage RSS on first feed (materialization spike):");
        for (stage, _cnt, wbytes, rss) in r.per_stage_rss.iter().take(count.max(1)) {
            eprintln!(
                "      stage {:>2}/{:<2}  weights {:>8.1} MiB   RSS {:>7.3} GiB",
                stage + 1,
                count,
                *wbytes as f64 / (1u64 << 20) as f64,
                gib(*rss)
            );
        }
    }

    let delta = r.rss_peak.saturating_sub(r.resident_footprint);
    eprintln!("  ── measured peak ──");
    eprintln!(
        "  RSS baseline (pre-build)     : {:.3} GiB",
        gib(r.rss_baseline)
    );
    eprintln!(
        "  RSS after runners built      : {:.3} GiB  (+{:.3} GiB modules, no weights yet)",
        gib(r.rss_after_build),
        gib(r.rss_after_build.saturating_sub(r.rss_baseline))
    );
    eprintln!(
        "  RSS PEAK (feed + steps)      : {:.3} GiB",
        gib(r.rss_peak)
    );
    eprintln!(
        "  ledger peak_resident_footprint: {:.3} GiB  (tracked: resident weights + transients)",
        gib(r.resident_footprint)
    );
    eprintln!(
        "  step runner peak weight bytes : {:.3} GiB  (actual κ-store bytes resolved resident)",
        gib(r.step_runner_peak_weight)
    );
    eprintln!(
        "  DELTA rss − ledger footprint  : {:.3} GiB  (UNTRACKED: modules + F32 head scratch + activations + runtime)",
        gib(delta)
    );
    eprintln!(
        "  decode-region heap alloc vol  : {:.3} GiB cumulative over {} allocations (not a live peak)",
        gib(r.decode_alloc_bytes),
        r.decode_alloc_count
    );
    let ceil = gib(STRUCTURAL_CEILING);
    if r.rss_peak > STRUCTURAL_CEILING {
        eprintln!(
            "  >>> PEAK {:.3} GiB EXCEEDS the wasm 4 GiB ceiling by {:.3} GiB <<<",
            gib(r.rss_peak),
            gib(r.rss_peak - STRUCTURAL_CEILING)
        );
    } else {
        eprintln!(
            "  peak {:.3} GiB is under the {:.0} GiB ceiling (headroom {:.3} GiB)",
            gib(r.rss_peak),
            ceil,
            gib(STRUCTURAL_CEILING - r.rss_peak)
        );
    }
}

#[test]
fn decode_peak_memory_across_families() {
    if std::env::var("HOLOGRAM_AI_LARGE").as_deref() != Ok("1") {
        eprintln!("SKIP: set HOLOGRAM_AI_LARGE=1 to run the large-config decode-memory sweep");
        return;
    }
    counting::assert_allocator_installed();

    let layer_list: Vec<u64> = match std::env::var("HOLOGRAM_AI_DECODE_LAYERS").ok() {
        Some(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<u64>().ok())
            .filter(|&l| l >= 1)
            .collect(),
        None => vec![4], // reduced default; set a larger count for the full sweep
    };
    assert!(
        !layer_list.is_empty(),
        "HOLOGRAM_AI_DECODE_LAYERS parsed to no layers"
    );

    // Scale tier: `large` ≈ 20 B params, `xl` ≈ 500 B+ (its single embed tensor
    // alone exceeds the residency budget — the weight-tier paging frontier). The
    // layer knob turns depth down for a feasible native run; the WIDTH (the
    // ceiling-driving embed/head/per-layer stages) stays at the tier's scale.
    let dims = match std::env::var("HOLOGRAM_AI_TIER").ok().as_deref() {
        Some("xl") | Some("extra-large") | Some("extra_large") => Dims::EXTRA_LARGE,
        _ => Dims::LARGE,
    };

    // Which families to sweep — default ALL faithful families (the point is that
    // memory is bounded for every layout, not one). `HOLOGRAM_AI_FAMILIES` (a
    // comma-separated arch-name substring filter) narrows it on a small box.
    let family_filter = std::env::var("HOLOGRAM_AI_FAMILIES").ok();
    let families: Vec<_> = FAITHFUL_FAMILIES
        .iter()
        .filter(|l| {
            family_filter
                .as_ref()
                .is_none_or(|f| f.split(',').any(|t| l.arch.contains(t.trim())))
        })
        .collect();
    assert!(
        !families.is_empty(),
        "HOLOGRAM_AI_FAMILIES matched no faithful family"
    );

    eprintln!(
        "box: MemAvailable {:.1} GiB. wasm ceiling = {:.0} GiB (STRUCTURAL_CEILING); \
         residency budget = {:.0} GiB (ceiling − 1 GiB reserve), bound_by_footprint = true.\n\
         Sweeping families {:?} × layers {:?} on the shared Dims::LARGE template.",
        gib(mem_available_bytes()),
        gib(STRUCTURAL_CEILING),
        gib(RESIDENCY_BUDGET),
        families.iter().map(|l| l.arch).collect::<Vec<_>>(),
        layer_list,
    );

    let mut results = Vec::new();
    for layout in &families {
        for &layers in &layer_list {
            let scale = FamilyScale::new(**layout, dims.with_layers(layers));
            results.push(measure_scale(scale));
        }
    }

    eprintln!("\n════════════════════ memory-vs-scale summary ════════════════════");
    eprintln!(
        "{:>20}  {:>6}  {:>9}  {:>7}  {:>10}  {:>10}  {:>11}  {:>10}  {:>9}",
        "family",
        "layers",
        "params",
        "stages",
        "wide bf16",
        "int8 art",
        "RSS peak",
        "ledger",
        "delta"
    );
    for r in &results {
        eprintln!(
            "{:>20}  {:>6}  {:>7.3}B  {:>7}  {:>8.3}Gi  {:>8.3}Gi  {:>9.3}Gi  {:>8.3}Gi  {:>7.3}Gi",
            r.arch,
            r.layers,
            r.total_params as f64 / 1e9,
            r.stage_count,
            gib(r.wide_weight_bytes),
            gib(r.artifact_bytes),
            gib(r.rss_peak),
            gib(r.resident_footprint),
            gib(r.rss_peak.saturating_sub(r.resident_footprint)),
        );
    }
    for r in &results {
        report(r);
    }

    // The contract the wasm build enforces structurally (an allocation abort):
    // the peak must stay under the 4 GiB address-space ceiling for EVERY family.
    // If any exceeds, FAIL loud with the breakdown so the fix targets the right
    // term — resident weights (the ledger) or untracked module/scratch memory.
    let worst = results
        .iter()
        .max_by_key(|r| r.rss_peak)
        .expect("at least one scale ran");
    assert!(
        worst.rss_peak <= STRUCTURAL_CEILING,
        "decode peak RSS {:.3} GiB for {} at {} layers EXCEEDS the wasm 4 GiB ceiling by {:.3} GiB. \
         Breakdown: ledger-tracked resident footprint {:.3} GiB, UNTRACKED (compiled modules \
         + F32 head scratch + activations + runtime) {:.3} GiB. \
         {} decode stages × 2 runners compiled to {:.1} MiB of archives.",
        gib(worst.rss_peak),
        worst.arch,
        worst.layers,
        gib(worst.rss_peak - STRUCTURAL_CEILING),
        gib(worst.resident_footprint),
        gib(worst.rss_peak.saturating_sub(worst.resident_footprint)),
        worst.stage_count,
        worst.module_bytes as f64 / (1u64 << 20) as f64,
    );
}
