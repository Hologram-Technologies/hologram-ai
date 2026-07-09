//! Staged (windowed) execution over k — dictionary row `staged-execution`.
//!
//! The classical "whole model resident in memory" assumption is a residency
//! assumption, not a law (`docs/conceptual-model/01-k-representation.md`).
//! This module removes it: the parametric decoder is compiled as a sequence
//! of **stage archives** (embedding, decoder-layer blocks, head), each an
//! ordinary k-form `.holo` whose weight constants are κ-bound placeholders.
//! Execution materializes one stage against the κ-store, runs it, hands its
//! output activations to the next stage, and **drops the session before the
//! next stage materializes** — peak weight residency is a parametric WINDOW
//! (the largest stage × context), never the model. The model itself lives in
//! the κ-store (OPFS in the browser, a directory natively).
//!
//! Staged and monolithic execution are the same computation: the stage
//! graphs are emitted by the same layer-emission recipe as the monolithic
//! graph and run the same kernels in the same per-layer order, so the staged
//! pipeline reproduces the monolithic logits **byte-for-byte** (the
//! `staged-execution` witness), with the head-stage boundary placed on the
//! fused final-norm operands — see
//! [`hologram_ai_safetensors::parametric::build_parametric_stage_graphs`].

use std::collections::HashMap;
use std::num::NonZeroU64;

use anyhow::{anyhow, bail, ensure, Context, Result};
use hologram_ai_common::{shape_from_concrete, AiParam, DType, TensorInfo};
use hologram_archive::{decode_ports, HoloLoader, SectionKind};
use hologram_exec::OutputBuffer;

use crate::engine::{LmSession, SessionProvider};
use crate::materialize::{materialize_archive_with, KappaStore};
use crate::runner::{HoloRunner, PortInfo};
use crate::{ModelCompiler, ModelSource};

/// Compile the parametric decoder as **stage archives** (k-form `.holo`
/// bytes, in execution order): embedding, `ceil(L / layers_per_stage)`
/// decoder-layer blocks, head. Mirrors the monolithic streamed compile
/// (`ModelSource::SafetensorsStreamed`): every stage's weights are
/// [`AiParam::External`] κ-bindings, so the archives are weightless structure
/// and the κ-store holds the parameters exactly once — a tensor consumed by
/// two stages (the tied embedding) binds the same κ in both stage κ-maps.
///
/// The manifest is the parallel `keys`/`kappas`/`shapes`/`dtypes` slices of
/// the streamed download; `context_length` follows the monolithic rule
/// (`Some(n)` validated against the model's effective ceiling, `None` = the
/// model's own trained context). Fails loud if any manifest tensor is
/// consumed by no stage — the partition never silently drops a weight.
pub fn compile_stages(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    context_length: Option<u64>,
    layers_per_stage: NonZeroU64,
) -> Result<Vec<Vec<u8>>> {
    compile_stages_with(
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        context_length,
        layers_per_stage,
        None,
    )
}

/// [`compile_stages`] with a quantized derived-artifact tier (row
/// `quantized-transit`): after κ-binding, each stage graph's κ-bound MatMul
/// weights whose wide κ has a recorded quantized derivation are rewritten
/// onto ranged bindings into the artifact's κ
/// ([`hologram_ai_common::lower::quantize_external_matmul_weights`]) — the
/// wide κ leaves the stage's κ-map, so it never transits and never
/// materializes. Coverage is per weight: κs absent from the map (or consumed
/// outside the projection chain) keep their wide binding.
#[allow(clippy::too_many_arguments)]
pub fn compile_stages_with(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    context_length: Option<u64>,
    layers_per_stage: NonZeroU64,
    quant: Option<&hologram_ai_common::lower::QuantMap>,
) -> Result<Vec<Vec<u8>>> {
    let graphs = bound_stage_graphs(
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        context_length,
        layers_per_stage,
    )?;
    let mut archives = Vec::with_capacity(graphs.len());
    for (stage, mut graph) in graphs.into_iter().enumerate() {
        if let Some(quant) = quant {
            hologram_ai_common::lower::quantize_external_matmul_weights(&mut graph, quant)
                .with_context(|| format!("quantizing stage {stage} onto derived artifacts"))?;
        }
        let archive = ModelCompiler::default()
            .compile(ModelSource::AiGraph(graph))
            .with_context(|| format!("compiling stage {stage}"))?;
        archives.push(archive.bytes);
    }
    Ok(archives)
}

/// [`compile_stages_with`] for the **decode-step** plan (row `decode-plan`,
/// staged realization): the same partition, cut points, and κ-map coverage,
/// assembled at seq = 1 with each layer stage's attention decomposed over a
/// fixed `bucket`-row carried past. One archive set serves every step;
/// growing past the bucket is a recompile at a larger bucket.
#[allow(clippy::too_many_arguments)]
pub fn compile_decode_stages(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    bucket: u64,
    layers_per_stage: NonZeroU64,
    quant: Option<&hologram_ai_common::lower::QuantMap>,
) -> Result<Vec<Vec<u8>>> {
    compile_chunk_stages(
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        bucket,
        1,
        layers_per_stage,
        quant,
    )
}

/// [`compile_decode_stages`] parametric in the chunk (row `chunked-prefill`):
/// `chunk` positions per pass over the carried past — the prefill-seeding
/// pipeline that amortizes the weight stream across the chunk.
#[allow(clippy::too_many_arguments)]
pub fn compile_chunk_stages(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    bucket: u64,
    chunk: u64,
    layers_per_stage: NonZeroU64,
    quant: Option<&hologram_ai_common::lower::QuantMap>,
) -> Result<Vec<Vec<u8>>> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).context("parsing config.json")?;
    let graphs = hologram_ai_safetensors::parametric::build_parametric_chunk_stage_graphs(
        &config,
        keys,
        dtypes,
        bucket,
        chunk,
        layers_per_stage,
    )?;
    bind_quantize_compile(graphs, keys, kappas, shapes, dtypes, quant)
}

/// The **staged verify pipeline** of row `speculative-decode`: `chunk` positions
/// per pass whose head emits logits at every position (over any vocabulary — the
/// head still chunks). The browser's paged/staged runner verifies a `K`-token
/// draft in one `M = K` pass; K/V is spliced from that same pass. Same κ
/// bindings and quant map as the decode pipeline, so it shares the weight tier.
#[allow(clippy::too_many_arguments)]
pub fn compile_verify_stages(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    bucket: u64,
    chunk: u64,
    layers_per_stage: NonZeroU64,
    quant: Option<&hologram_ai_common::lower::QuantMap>,
) -> Result<Vec<Vec<u8>>> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).context("parsing config.json")?;
    let graphs = hologram_ai_safetensors::parametric::build_parametric_verify_stage_graphs(
        &config,
        keys,
        dtypes,
        bucket,
        chunk,
        layers_per_stage,
    )?;
    bind_quantize_compile(graphs, keys, kappas, shapes, dtypes, quant)
}

/// Bind manifest κs onto stage graphs, optionally rewrite their matmuls onto
/// quantized artifacts, and compile each — the shared back half of the chunk,
/// decode, and verify pipelines.
fn bind_quantize_compile(
    graphs: Vec<hologram_ai_common::AiGraph>,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    quant: Option<&hologram_ai_common::lower::QuantMap>,
) -> Result<Vec<Vec<u8>>> {
    let graphs = bind_manifest_kappas(graphs, keys, kappas, shapes, dtypes)?;
    let mut archives = Vec::with_capacity(graphs.len());
    for (stage, mut graph) in graphs.into_iter().enumerate() {
        if let Some(quant) = quant {
            hologram_ai_common::lower::quantize_external_matmul_weights(&mut graph, quant)
                .with_context(|| format!("quantizing stage {stage} onto artifacts"))?;
        }
        let archive = ModelCompiler::default()
            .compile(ModelSource::AiGraph(graph))
            .with_context(|| format!("compiling stage {stage}"))?;
        archives.push(archive.bytes);
    }
    Ok(archives)
}

/// The stage graphs of the staged plan with every manifest κ bound — the
/// shared front half of [`compile_stages_with`] and [`quantizable_weights`].
/// Fails loud if any manifest tensor is consumed by no stage.
#[allow(clippy::too_many_arguments)]
fn bound_stage_graphs(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    context_length: Option<u64>,
    layers_per_stage: NonZeroU64,
) -> Result<Vec<hologram_ai_common::AiGraph>> {
    let config: serde_json::Value =
        serde_json::from_str(config_json).context("parsing config.json")?;
    let graphs = hologram_ai_safetensors::parametric::build_parametric_stage_graphs(
        &config,
        keys,
        dtypes,
        context_length,
        layers_per_stage,
    )?;
    bind_manifest_kappas(graphs, keys, kappas, shapes, dtypes)
}

/// Bind the κ of every manifest tensor each stage declares; fail loud if any
/// manifest tensor is consumed by no stage.
fn bind_manifest_kappas(
    mut graphs: Vec<hologram_ai_common::AiGraph>,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
) -> Result<Vec<hologram_ai_common::AiGraph>> {
    ensure!(
        keys.len() == kappas.len() && keys.len() == shapes.len() && keys.len() == dtypes.len(),
        "manifest slices disagree: {} keys, {} κs, {} shapes, {} dtypes",
        keys.len(),
        kappas.len(),
        shapes.len(),
        dtypes.len()
    );
    let mut bound = vec![false; keys.len()];
    for graph in graphs.iter_mut() {
        // Bind the κ of every manifest tensor this stage declares. Only
        // declared names are bound — a stage's κ-map is exactly the weights
        // its layers consume, which is what the partition witness checks.
        let name_to_id: HashMap<String, u32> = graph
            .tensor_names
            .iter()
            .map(|(id, name)| (name.clone(), *id))
            .collect();
        for (i, key) in keys.iter().enumerate() {
            let Some(&id) = name_to_id.get(key) else {
                continue;
            };
            // A chunked stage binds a RANGE of the tensor: the builder
            // recorded `kappa_range:<name>` metadata and declared the CHUNK
            // shape itself — honor both; the κ stays the whole tensor's.
            let range = graph
                .metadata
                .get(&format!("kappa_range:{key}"))
                .and_then(|v| match v {
                    hologram_ai_common::MetaValue::Str(s) => {
                        let (off, len) = s.split_once('+')?;
                        Some((off.parse().ok()?, len.parse().ok()?))
                    }
                    _ => None,
                });
            let info =
                match range {
                    Some(_) => graph.tensor_info.get(&id).cloned().unwrap_or_else(|| {
                        TensorInfo::new(dtypes[i], shape_from_concrete(&shapes[i]))
                    }),
                    None => TensorInfo::new(dtypes[i], shape_from_concrete(&shapes[i])),
                };
            graph.tensor_info.insert(id, info.clone());
            graph.params.insert(
                id,
                AiParam::External {
                    kappa: kappas[i].clone(),
                    info,
                    range,
                },
            );
            bound[i] = true;
        }
    }

    if let Some(i) = bound.iter().position(|b| !b) {
        bail!(
            "manifest tensor `{}` is consumed by no stage graph — the staged \
             partition must cover the model's tensors exactly",
            keys[i]
        );
    }
    Ok(graphs)
}

/// The wide κs the staged plan can rewrite onto quantized artifacts AND
/// fully retire in EVERY stage that consumes them (row `quantized-transit`,
/// browser tier `quantized-rest`): the κs whose wide blobs go gas-phase
/// once their artifacts crystallize. A κ that any stage keeps wide-bound
/// (a tied embedding's Gather, a chunked head's ranged bindings) is
/// excluded — its blob stays load-bearing.
#[allow(clippy::too_many_arguments)]
pub fn quantizable_weights(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    context_length: Option<u64>,
    layers_per_stage: NonZeroU64,
) -> Result<Vec<String>> {
    let graphs = bound_stage_graphs(
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        context_length,
        layers_per_stage,
    )?;
    let mut eligible = std::collections::BTreeSet::new();
    let mut kept_wide = std::collections::HashSet::new();
    for graph in &graphs {
        let per_stage = hologram_ai_common::lower::quantizable_external_weights(graph)?;
        let stage_set: std::collections::HashSet<&String> = per_stage.iter().collect();
        // κs this stage binds but cannot fully retire stay wide everywhere.
        for param in graph.params.values() {
            if let AiParam::External { kappa, .. } = param {
                if !stage_set.contains(kappa) {
                    kept_wide.insert(kappa.clone());
                }
            }
        }
        eligible.extend(per_stage);
    }
    Ok(eligible
        .into_iter()
        .filter(|k| !kept_wide.contains(k))
        .collect())
}

/// The head-chunk quantization targets of the staged plan (row
/// `quantized-transit`, chunked head): the vocab-row ranges of a large LM head
/// that the int8 tier derives into per-chunk artifacts, so a chunked head
/// joins the int8 tier instead of remaining a bf16 matmul whose whole-panel F32
/// image thrashes residency. Distinct from [`quantizable_weights`], which lists
/// whole projection κs to RETIRE: a head chunk's κ (a tied head's is the
/// embedding table's) stays wide for the embedding Gather — only its slice is
/// crystallized, never the tensor. Deduplicated by (κ, range), so the several
/// stages that reference one chunk report it once.
pub fn head_quant_chunks(
    config_json: &str,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    dtypes: &[DType],
    context_length: Option<u64>,
    layers_per_stage: NonZeroU64,
) -> Result<Vec<hologram_ai_common::lower::HeadChunkTarget>> {
    let graphs = bound_stage_graphs(
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        context_length,
        layers_per_stage,
    )?;
    let mut seen = std::collections::HashSet::new();
    let mut targets = Vec::new();
    for graph in &graphs {
        for target in hologram_ai_common::lower::ranged_external_matmul_weights(graph) {
            let key = hologram_ai_common::lower::quant_key(
                &target.kappa,
                Some((target.offset, target.len)),
            );
            if seen.insert(key) {
                targets.push(target);
            }
        }
    }
    Ok(targets)
}

/// A κ-store adapter that tallies the bytes it resolves — the per-stage
/// weight-residency instrument of the [`StagedRunner`].
struct CountingStore<'s> {
    inner: &'s mut dyn KappaStore,
    bytes: u64,
}

impl KappaStore for CountingStore<'_> {
    fn resolve(&mut self, kappa: &str) -> Result<Vec<u8>> {
        let content = self.inner.resolve(kappa)?;
        self.bytes += content.len() as u64;
        Ok(content)
    }

    fn invalidate(&mut self, kappa: &str) {
        self.inner.invalidate(kappa);
    }

    fn resolve_range(&mut self, kappa: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        let content = self.inner.resolve_range(kappa, offset, len)?;
        self.bytes += content.len() as u64;
        Ok(content)
    }
}

/// Resolves the k-form archive bytes of stage `i` — a `Vec` of precompiled
/// archives natively, an OPFS read in the browser. Archives are weightless
/// k-forms (structure + κ-bindings), so resolving one moves no parameters.
pub type StageResolver<'a> = Box<dyn FnMut(usize) -> Result<Vec<u8>> + 'a>;

/// A factory producing a fresh `Send` κ-store — one per paged stage session,
/// backing the weight provider independently of the runner's own `store`.
/// Native: a `DirKappaStore` over the same path. Browser: a cloned OPFS
/// store (single-threaded, so `Send`-safe). The provider inherits the
/// store's verify/invalidate/seek tiers.
pub type ResolverFactory<'a> = Box<dyn Fn() -> crate::runner::PagedStore + 'a>;

/// Weight-tier paging configuration (row `lazy-constant-residency`).
struct PagedWeights<'a> {
    /// Resident paged-weight byte budget per stage session (`0` = unbounded).
    budget: usize,
    /// Produces the provider's κ-store, `Send` per hologram's `load_paged`.
    make_resolver: ResolverFactory<'a>,
}

/// Per-stage observer: `(stage, stage_count, materialized_weight_bytes)`,
/// called after a stage materializes and before it executes.
pub type StageObserver<'a> = Box<dyn FnMut(usize, usize, u64) + 'a>;

/// A [`StageObserver`] shared across the regrown runners of a
/// [`GrowableStagedSession`].
type SharedStageObserver = std::rc::Rc<std::cell::RefCell<dyn FnMut(usize, usize, u64)>>;

/// Admission probe: consulted before a stage session is kept resident,
/// with the byte MARGIN the pipeline must keep free — the structural
/// transient bound of its largest stage (archive copy + materialized copy +
/// loaded constants + up-to-2× dtype widening ≤ 4× the stage's raw weight
/// bytes). `false` means the environment measurably lacks that headroom —
/// the session drops (strict windowing) instead of risking the heap. A
/// fixed margin crashed a 1.5B model at its head stage while smaller
/// stages held the room; the margin is a function of the MODEL, so the
/// probe receives it. Raw κ-byte budgets under-count a live session's true
/// footprint; only the environment can answer whether it has room, so
/// admission asks it directly.
pub type AdmissionProbe<'a> = Box<dyn Fn(u64) -> bool + 'a>;

/// The structural transient bound of a stage's materialize-and-execute:
/// the raw copies (archive + materialized image, plus slack for the loaded
/// runner) and TWO full F32 images of the stage's elements — the kernel
/// widens a narrow-dtype panel to F32 and holds a pre-transposed scratch of
/// the same size. Parametric in elements, not bytes: a bf16 stage's
/// execution image is twice its storage.
fn stage_transient_bound(stage_weight_bytes: u64, stage_elements: u64) -> u64 {
    stage_weight_bytes
        .saturating_mul(3)
        .saturating_add(stage_elements.saturating_mul(8))
}

/// Windowed execution over the stage archives of [`compile_stages`].
///
/// One token-window forward pass runs the stages in order: resolve the
/// stage's k-form archive, materialize it against the κ-store, load a
/// [`HoloRunner`], execute it with the previous stage's output bytes as
/// input, and **drop the session before the next stage materializes**. Stage
/// 0 consumes the `input_ids` window; the final stage produces the logits.
/// Peak resident weight bytes are therefore bounded by the largest stage —
/// the window — never the model, and the per-stage accounting
/// ([`Self::stage_weight_bytes`], [`Self::peak_resident_weight_bytes`])
/// measures exactly that.
///
/// The combined address-space residency of every runner that shares it — the
/// step plan, the prefill seeder, and any verify plan of one decode session.
/// `footprint` is the live resident bytes summed across all siblings;
/// `max_walk` is the largest single-stage footprint any sibling has walked (the
/// headroom the reserve must leave for the next walk); `peak` is the high-water
/// `footprint` for instrumentation/witnesses. A footprint-bounded runner admits
/// a stage only while `footprint + this + max_walk <= budget`, so no
/// combination of concurrent runners over-commits the ceiling.
#[derive(Default)]
pub(crate) struct ResidencyLedger {
    footprint: u64,
    max_walk: u64,
    peak: u64,
}

/// A handle every sibling runner of one session shares.
pub(crate) type SharedResidency = std::rc::Rc<std::cell::RefCell<ResidencyLedger>>;

/// Implements [`LmSession`] + [`SessionProvider`], so
/// [`generate_stream`](crate::commands::generate::generate_stream) drives it
/// unchanged.
///
/// # Shared residency ledger
///
/// A single decode turn holds SEVERAL runners at once against ONE address
/// space — the step plan (chunk 1), the chunked-prefill seeder (chunk `C`),
/// and, under speculation, the verify plan. Each would otherwise gate residency
/// against its own budget and, together, over-commit the wasm 4 GiB ceiling. So
/// footprint-bounded runners spawned by one [`GrowableStagedSession`] share a
/// residency ledger: admission is charged against the COMBINED resident
/// footprint of every sibling, and a dropped runner returns its share. The
/// address space is one resource, accounted as one.
pub struct StagedRunner<'a> {
    resolve_stage: StageResolver<'a>,
    store: Box<dyn KappaStore + 'a>,
    /// Weight-tier paging (row `lazy-constant-residency`), if enabled: each
    /// stage loads PAGED against a residency budget instead of materializing
    /// its weights whole — the arena is a bounded window over the κ-store, so
    /// a stage whose weights exceed the window still runs. The resolver
    /// factory produces a `Send` κ-resolver per stage session (independent of
    /// this runner's `store`, which paging still uses to build the paged
    /// archive and resolve any ranged sub-tensor bindings inline).
    paged: Option<PagedWeights<'a>>,
    stage_count: usize,
    input_ports: Vec<PortInfo>,
    output_ports: Vec<PortInfo>,
    /// The compiled `input_ids` window (element count) of stage 0.
    window: usize,
    /// Weight bytes materialized per stage — written on execution (a stage
    /// that has not run yet reports 0).
    stage_weight_bytes: Vec<u64>,
    /// The largest single-stage weight residency observed across executions.
    peak_resident_weight_bytes: u64,
    /// Observer called after each stage materializes, before it executes:
    /// `(stage, stage_count, materialized_weight_bytes)`. Lets a UI surface
    /// per-stage progress instead of a silent first-token wait.
    on_stage: Option<StageObserver<'a>>,
    /// Residency budget (bytes) for keeping materialized stage sessions
    /// across forward passes — `0` (the default) is strict windowing: every
    /// pass rematerializes every stage. See [`Self::set_residency_budget`].
    residency_budget: u64,
    /// Materialized stage sessions held under the budget, with `(weight_bytes,
    /// footprint_bytes)`. `weight_bytes` is the packed archive size (the
    /// weight-residency metric the witnesses assert); `footprint_bytes` is the
    /// session's TRUE runtime footprint (weights PLUS the intermediates the
    /// substrate buffer pool keeps resident — e.g. a float LM-head chunk's
    /// whole-panel F32 Cast+Transpose image, many times its packed weight).
    /// `None` = not resident (drops after its pass, freeing that footprint).
    resident: Vec<Option<(HoloRunner, u64, u64)>>,
    /// Weight bytes currently held by `resident` (the cache metric).
    resident_bytes: u64,
    /// This runner's OWN contribution to the shared ledger's `footprint` — the
    /// TRUE runtime footprint its resident stages hold. Tracked locally so the
    /// runner can return exactly its share to the ledger when a stage is evicted
    /// or the whole runner is dropped.
    resident_footprint: u64,
    /// The address-space residency ledger this runner shares with its session
    /// siblings (step / seeder / verify). Admission is gated on the COMBINED
    /// footprint here, not this runner's alone, so concurrent runners never
    /// over-commit the ceiling. A standalone runner gets its own fresh ledger.
    residency: SharedResidency,
    /// When set, the residency budget is a HARD address-space ceiling (the
    /// wasm32 tab): admission is gated on the shared true-footprint ledger plus
    /// a largest-walk reserve, not on the packed weight bytes. Default `false` —
    /// the budget is a κ-store-bandwidth cache limit and a 64-bit host has no
    /// address ceiling to respect.
    bound_by_footprint: bool,
    /// Stage materializations performed over this runner's lifetime — the
    /// bandwidth instrument (`resident` hits don't count).
    materialization_count: u64,
    /// Per stage, per input port: where its bytes come from. Built once at
    /// construction from the archives' own port names (see [`Feed`]).
    feeds: Vec<Vec<Feed>>,
    /// Per non-final stage, per output port: where its bytes go (see [`Sink`]).
    sinks: Vec<Vec<Sink>>,
    /// Number of intermediate-stage outputs surfaced as trailing pipeline
    /// outputs (after the final stage's own outputs).
    surfaced_count: usize,
    /// Kernels dispatched / elided across all stages of the most recent
    /// forward pass (class CE, summed per stage) — the decode attribution
    /// instrument of the `performance-contract` row.
    last_dispatched: u64,
    last_skipped: u64,
    /// Environment headroom probe consulted at admission (see
    /// [`AdmissionProbe`]). `None` = admission by byte budget alone.
    admission_probe: Option<AdmissionProbe<'a>>,
    /// Expected (raw weight bytes, element count) per stage, computed by the
    /// session from the manifest BEFORE anything materializes — the largest
    /// transient bound drives the admission margin. Falls back to measured
    /// bytes as stages run.
    expected_stage_bytes: Vec<(u64, u64)>,
    /// The session verified-κ set (row `session-verified-kappa`): a κ
    /// verifies at its first materialization this session; later
    /// rematerializations are read-only I/O. Shared across regrows of a
    /// growable session — the session, not the window, is the trust scope.
    verified: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,
}

/// Where a stage input's bytes come from, resolved once at construction by
/// port NAME: an input the immediately previous stage produces is a carried
/// activation; anything else is a pipeline-level input (auxiliaries like
/// `last_pos`, and the decode plan's `rope_cos`/`rope_sin`/`decode_mask`/
/// `past_k_l`/`past_v_l`), deduplicated by name — one pipeline port may feed
/// many stages.
enum Feed {
    Pipeline(usize),
    Carried(String),
}

/// Where a non-final stage output's bytes go: into the next stage (an input
/// there shares its name) or surfaced as a trailing pipeline output (the
/// decode plan's `k_new_l`/`v_new_l`).
enum Sink {
    Carry(String),
    Surface(usize),
}

impl<'a> StagedRunner<'a> {
    /// Build a runner over `stage_count` stages resolved on demand through
    /// `resolve_stage`, materializing κs against `store`. Reads the LM port
    /// contract from the k-form archives' port sections (weight-free): stage
    /// 0 must declare an `input_ids` input and the final stage a `logits`
    /// output. Routing between stages is by port NAME (`Feed`/`Sink`): the
    /// archives' own ports are the pipeline's contract, never positional
    /// guessing.
    pub fn new(
        stage_count: usize,
        mut resolve_stage: StageResolver<'a>,
        store: Box<dyn KappaStore + 'a>,
    ) -> Result<Self> {
        ensure!(
            stage_count >= 1,
            "a staged pipeline needs at least one stage"
        );

        let stage_ports = |resolve_stage: &mut StageResolver<'a>,
                           stage: usize,
                           kind: SectionKind|
         -> Result<Vec<PortInfo>> {
            let archive = resolve_stage(stage)
                .with_context(|| format!("resolving the stage {stage} archive"))?;
            archive_ports(&archive, kind).with_context(|| format!("reading stage {stage} ports"))
        };

        // Stage 0's inputs are pipeline inputs verbatim.
        let mut input_ports = stage_ports(&mut resolve_stage, 0, SectionKind::Inputs)?;
        let mut feeds: Vec<Vec<Feed>> = vec![(0..input_ports.len()).map(Feed::Pipeline).collect()];
        let mut sinks: Vec<Vec<Sink>> = Vec::new();
        let mut surfaced_ports: Vec<PortInfo> = Vec::new();
        let mut prev_outputs = stage_ports(&mut resolve_stage, 0, SectionKind::Outputs)?;

        for stage in 1..stage_count {
            let ins = stage_ports(&mut resolve_stage, stage, SectionKind::Inputs)?;
            feeds.push(
                ins.iter()
                    .map(|p| {
                        if prev_outputs.iter().any(|o| o.name == p.name) {
                            Feed::Carried(p.name.clone())
                        } else {
                            // Pipeline-level input, deduplicated by name (the
                            // decode plan's shared position ports feed every
                            // layer stage from one pipeline port).
                            let idx = input_ports
                                .iter()
                                .position(|q| q.name == p.name)
                                .unwrap_or_else(|| {
                                    input_ports.push(p.clone());
                                    input_ports.len() - 1
                                });
                            Feed::Pipeline(idx)
                        }
                    })
                    .collect(),
            );
            // The previous stage's outputs either carry into this stage or
            // surface as trailing pipeline outputs.
            sinks.push(
                prev_outputs
                    .iter()
                    .map(|o| {
                        if ins.iter().any(|p| p.name == o.name) {
                            Sink::Carry(o.name.clone())
                        } else {
                            surfaced_ports.push(o.clone());
                            Sink::Surface(surfaced_ports.len() - 1)
                        }
                    })
                    .collect(),
            );
            prev_outputs = stage_ports(&mut resolve_stage, stage, SectionKind::Outputs)?;
        }

        // Pipeline outputs: the final stage's outputs first (`logits` stays
        // the leading contract), then the surfaced intermediates.
        let mut output_ports = prev_outputs;
        let surfaced_count = surfaced_ports.len();
        output_ports.extend(surfaced_ports);

        let window = input_ports
            .iter()
            .find(|p| p.name == "input_ids")
            .map(|p| p.element_count)
            .ok_or_else(|| {
                anyhow!(
                    "stage 0 declares no `input_ids` input port (its ports are {:?})",
                    input_ports
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                )
            })?;
        ensure!(
            output_ports.iter().any(|p| p.name == "logits"),
            "the final stage declares no `logits` output port (its ports are {:?})",
            output_ports
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
        );

        Ok(Self {
            resolve_stage,
            store,
            paged: None,
            stage_count,
            feeds,
            sinks,
            surfaced_count,
            input_ports,
            output_ports,
            window,
            stage_weight_bytes: vec![0; stage_count],
            peak_resident_weight_bytes: 0,
            on_stage: None,
            residency_budget: 0,
            resident: (0..stage_count).map(|_| None).collect(),
            resident_bytes: 0,
            resident_footprint: 0,
            residency: SharedResidency::default(),
            bound_by_footprint: false,
            last_dispatched: 0,
            last_skipped: 0,
            materialization_count: 0,
            admission_probe: None,
            expected_stage_bytes: Vec::new(),
            verified: std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashSet::new())),
        })
    }

    /// Provide the manifest-derived (raw weight bytes, element count) per
    /// stage (known before any materialization) — the largest stage's
    /// transient bound is the margin every admission must leave free.
    pub fn set_expected_stage_bytes(&mut self, bytes: Vec<(u64, u64)>) {
        self.expected_stage_bytes = bytes;
    }

    /// The admission margin: the largest stage transient bound, expected
    /// (manifest-derived) or measured, whichever is larger. Measured raw
    /// bytes carry no dtype, so their element count is bounded by bytes
    /// (1-byte elements — the widest F32 image a byte count can imply).
    fn admission_margin(&self) -> u64 {
        // The manifest-derived per-stage element counts give the ACCURATE
        // transient bound; use them when available. Only when they are absent
        // (no `set_expected_stage_bytes`) do we fall back to the raw byte
        // counts, and there the element count is bounded by the byte count
        // itself (1-byte elements — the widest F32 image a byte count can
        // imply): dividing by 4 would UNDER-count elements for a bf16/int8/int4
        // stage and shrink the margin, risking over-admission → OOM instead of
        // the intended clean windowing degrade.
        if let Some(expected) = self
            .expected_stage_bytes
            .iter()
            .map(|&(bytes, elems)| stage_transient_bound(bytes, elems))
            .max()
        {
            return expected;
        }
        self.stage_weight_bytes
            .iter()
            .map(|&bytes| stage_transient_bound(bytes, bytes))
            .max()
            .unwrap_or(0)
    }

    /// Enable weight-tier paging (row `lazy-constant-residency`): each stage
    /// loads PAGED against `budget` resident paged-weight bytes instead of
    /// materializing its weights whole, `make_resolver` producing the
    /// provider's `Send` κ-resolver per stage. A stage whose weights exceed
    /// the window then still runs — the arena is a bounded window over the
    /// κ-store. `budget == 0` pages without eviction (unbounded).
    pub fn set_weight_paging(&mut self, budget: usize, make_resolver: ResolverFactory<'a>) {
        self.paged = Some(PagedWeights {
            budget,
            make_resolver,
        });
    }

    /// Adopt a shared session verified-κ set (a growable session forwards
    /// one across window regrows — same session, same trust scope).
    pub fn share_verified_set(
        &mut self,
        set: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,
    ) {
        self.verified = set;
    }

    /// Install an environment headroom probe consulted at every admission —
    /// a resident set can only grow while the environment measurably has
    /// room (see [`AdmissionProbe`]).
    pub fn set_admission_probe(&mut self, p: AdmissionProbe<'a>) {
        self.admission_probe = Some(p);
    }

    /// Set the residency budget: materialized stage sessions stay resident
    /// across forward passes while their cumulative weight bytes fit, so the
    /// κ-store bandwidth of a stage is paid once per window instead of once
    /// per token (row `stage-residency-cache`). `0` is strict windowing —
    /// peak weight residency is exactly one stage, the original contract.
    /// The budget is an environment measurement (heap ceiling minus working
    /// margin), never a preference: a model whose stages fit runs at
    /// resident-session speed, one that doesn't falls back to the window.
    pub fn set_residency_budget(&mut self, bytes: u64) {
        self.residency_budget = bytes;
        if bytes == 0 {
            for slot in self.resident.iter_mut() {
                *slot = None;
            }
            self.resident_bytes = 0;
            // Return this runner's whole share to the shared ledger.
            self.residency.borrow_mut().footprint -= self.resident_footprint;
            self.resident_footprint = 0;
        }
    }

    /// Share an address-space residency ledger with this runner's session
    /// siblings (step / seeder / verify plans of one decode turn). Admission is
    /// then charged against the COMBINED footprint of every sharer, so no
    /// concurrent set of runners over-commits the ceiling. Called when a
    /// [`GrowableStagedSession`] wires a fresh runner; a standalone runner keeps
    /// its own ledger.
    pub(crate) fn share_residency_ledger(&mut self, ledger: SharedResidency) {
        // Move any footprint this runner already holds onto the shared ledger.
        ledger.borrow_mut().footprint += self.resident_footprint;
        self.residency = ledger;
    }

    /// Drop every resident stage session — returning its share to the shared
    /// ledger — WITHOUT changing the budget, so the stages re-materialize on the
    /// next pass. Reclaims an idle auxiliary runner's residency for a hot
    /// sibling (the prefill seeder yielding to the step runner under memory
    /// pressure). A no-op when nothing is resident.
    pub fn evict_resident(&mut self) {
        for slot in self.resident.iter_mut() {
            *slot = None;
        }
        self.resident_bytes = 0;
        self.residency.borrow_mut().footprint -= self.resident_footprint;
        self.resident_footprint = 0;
    }

    /// This runner's own resident footprint (weights + retained transients).
    pub fn resident_footprint(&self) -> u64 {
        self.resident_footprint
    }

    /// The residency budget (address ceiling under `bound_by_footprint`).
    pub fn residency_budget(&self) -> u64 {
        self.residency_budget
    }

    /// Treat the residency budget as a HARD address-space ceiling (the wasm32
    /// tab), gating admission on each session's true runtime footprint plus a
    /// largest-walk reserve rather than its packed weight bytes. Set by a
    /// 32-bit host where a resident session's retained transients (a float
    /// LM-head chunk's F32 image) count against the same ceiling as its
    /// weights. Off by default: a 64-bit host has no such ceiling and residency
    /// is a pure κ-store-bandwidth cache denominated in weight bytes.
    pub fn set_bound_by_footprint(&mut self, bounded: bool) {
        self.bound_by_footprint = bounded;
    }

    /// Stage materializations performed so far — κ-store bandwidth in units
    /// of stage loads (resident-session hits don't rematerialize).
    pub fn materialization_count(&self) -> u64 {
        self.materialization_count
    }

    /// Kernels dispatched across all stages of the most recent forward pass
    /// (class CE — content-addressed elision; see `HoloRunner`).
    pub fn last_dispatched(&self) -> u64 {
        self.last_dispatched
    }

    /// Kernels elided (memo hits on the unchanged prefix cone) across all
    /// stages of the most recent forward pass.
    pub fn last_skipped(&self) -> u64 {
        self.last_skipped
    }

    /// Install a per-stage observer: called after each stage materializes
    /// (before it executes) with `(stage, stage_count, weight_bytes)`.
    pub fn set_stage_observer(&mut self, f: StageObserver<'a>) {
        self.on_stage = Some(f);
    }

    /// Convenience over an in-memory list of stage archives (the native path:
    /// `compile_stages` output plus a [`DirKappaStore`](crate::materialize::DirKappaStore)).
    pub fn from_archives(stages: Vec<Vec<u8>>, store: Box<dyn KappaStore + 'a>) -> Result<Self> {
        let stage_count = stages.len();
        let resolve = Box::new(move |i: usize| {
            stages.get(i).cloned().ok_or_else(|| {
                anyhow!("stage {i} is out of range (the pipeline has {stage_count} stages)")
            })
        });
        Self::new(stage_count, resolve, store)
    }

    /// Number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stage_count
    }

    /// The compiled token window (stage 0's `input_ids` element count).
    pub fn window(&self) -> usize {
        self.window
    }

    /// Weight bytes materialized per stage, indexed by stage. A stage that
    /// has not executed yet reports 0; after one forward pass every entry is
    /// the stage's real κ-resolved weight residency.
    pub fn stage_weight_bytes(&self) -> &[u64] {
        &self.stage_weight_bytes
    }

    /// The largest single-stage weight residency observed — the measured
    /// peak, which the windowed design bounds by the largest stage (sessions
    /// are dropped between stages, so stages never coexist).
    pub fn peak_resident_weight_bytes(&self) -> u64 {
        self.peak_resident_weight_bytes
    }

    /// One windowed forward pass: stages in order, previous outputs feeding
    /// the next stage's inputs, one materialized session resident at a time.
    fn execute_window(&mut self, inputs: &[&[u8]]) -> Result<Vec<OutputBuffer>> {
        let mut carried: HashMap<String, Vec<u8>> = HashMap::new();
        let mut surfaced: Vec<Option<Vec<u8>>> = (0..self.surfaced_count).map(|_| None).collect();
        self.last_dispatched = 0;
        self.last_skipped = 0;
        for stage in 0..self.stage_count {
            // Resident hit: the session's weights are already materialized —
            // no κ-store traffic for this stage. Taking the slot removes its
            // bytes from the held tally until (re-)admission below.
            let taken = self.resident[stage].take();
            if let Some((_, weight, footprint)) = &taken {
                self.resident_bytes -= weight;
                self.resident_footprint -= footprint;
                self.residency.borrow_mut().footprint -= footprint;
            }
            let mut runner = if let Some((runner, _, _)) = taken {
                runner
            } else {
                let archive = (self.resolve_stage)(stage)
                    .with_context(|| format!("resolving the stage {stage} archive"))?;
                let verified = std::rc::Rc::clone(&self.verified);
                let loaded = if let Some(paged) = self.paged.as_ref() {
                    // Weight-tier paging: build the paged archive (ranged
                    // sub-tensor bindings resolve inline; whole-κ weights
                    // become provider references) and load against the budget.
                    // The residency accounting for the resident-stage cache is
                    // the pager's budget — a paged stage session holds at most
                    // that many weight bytes, not the whole stage.
                    let (paged_archive, tbl) = crate::materialize::paged_archive_with(
                        &archive,
                        self.store.as_mut(),
                        &mut verified.borrow_mut(),
                    )
                    .with_context(|| format!("building the paged stage {stage} archive"))?;
                    self.stage_weight_bytes[stage] = if paged.budget > 0 {
                        (paged.budget as u64).min(tbl.total_bytes())
                    } else {
                        tbl.total_bytes()
                    };
                    let provider = std::sync::Arc::new(crate::runner::KappaWeightProvider::new(
                        tbl,
                        (paged.make_resolver)(),
                    ));
                    self.materialization_count += 1;
                    if let Some(f) = self.on_stage.as_mut() {
                        f(stage, self.stage_count, self.stage_weight_bytes[stage]);
                    }
                    HoloRunner::from_paged(paged_archive, provider, paged.budget)
                        .with_context(|| format!("loading paged stage {stage}"))?
                } else {
                    let mut counting = CountingStore {
                        inner: self.store.as_mut(),
                        bytes: 0,
                    };
                    let material = materialize_archive_with(
                        &archive,
                        &mut counting,
                        &mut verified.borrow_mut(),
                    )
                    .with_context(|| format!("materializing stage {stage}"))?;
                    self.stage_weight_bytes[stage] = counting.bytes;
                    self.materialization_count += 1;
                    if let Some(f) = self.on_stage.as_mut() {
                        f(stage, self.stage_count, counting.bytes);
                    }
                    HoloRunner::from_bytes(material)
                        .with_context(|| format!("loading stage {stage}"))?
                };
                drop(archive);
                loaded
            };

            // Each input port draws from its resolved source: a pipeline
            // input by index, or the previous stage's same-named output.
            let refs: Vec<&[u8]> = self.feeds[stage]
                .iter()
                .map(|feed| match feed {
                    Feed::Pipeline(i) => inputs.get(*i).copied().with_context(|| {
                        format!(
                            "stage {stage} needs pipeline input `{}` but only {} inputs were passed",
                            self.input_ports[*i].name,
                            inputs.len()
                        )
                    }),
                    Feed::Carried(name) => {
                        carried.get(name).map(Vec::as_slice).with_context(|| {
                            format!("stage {stage} expects `{name}` from the previous stage")
                        })
                    }
                })
                .collect::<Result<_>>()?;
            let outputs = runner
                .execute(&refs)
                .with_context(|| format!("executing stage {stage}"))?;
            self.last_dispatched += runner.last_dispatched() as u64;
            self.last_skipped += runner.last_skipped() as u64;

            // Keep the session resident while it fits the budget; otherwise it
            // drops here, before the next stage materializes — the residency
            // window. `resident[stage]` was `take`n above, so each tally counts
            // it at most once.
            let weight = self.stage_weight_bytes[stage];
            // The session's TRUE runtime footprint: its weights PLUS the
            // intermediates the substrate buffer pool retains from the last walk
            // (for a float LM-head chunk, a whole-panel F32 Cast+Transpose image
            // several times its packed weight). Residency costs the address
            // space this much — NOT the packed weight bytes — so admission is
            // gated on it. `.max(weight)` floors it at the weight (κ-dedup can
            // under-report a session that shares content with another).
            let footprint = (runner.resident_bytes() as u64).max(weight);

            // Two budget regimes. `bound_by_footprint` (the browser, wasm): the
            // budget is a HARD address-space ceiling, so residency must account
            // for the true footprint AND reserve room for the largest single
            // walk — otherwise resident float-head sessions' retained F32 scratch
            // accumulates into an allocation abort. Crucially the footprint is
            // charged against the SHARED ledger, so the step / seeder / verify
            // runners of one decode turn never over-commit the ceiling between
            // them. Default (native, and the κ-store-bandwidth residency
            // witnesses): the budget is a bandwidth cache limit denominated in
            // WEIGHT bytes — residency saves stage re-materialization, and a
            // 64-bit host has no address ceiling to respect. Both are
            // parametric: every quantity is measured from the model's own
            // stages, no size is special-cased.
            let within_budget = if self.bound_by_footprint {
                let mut ledger = self.residency.borrow_mut();
                // Any stage of any sibling runner could be the next to walk;
                // keep the shared worst case so the combined resident set always
                // leaves room for the largest single walk on top of it.
                ledger.max_walk = ledger.max_walk.max(footprint);
                ledger.footprint + footprint + ledger.max_walk <= self.residency_budget
            } else {
                self.resident_bytes + weight <= self.residency_budget
            };
            let admissible = weight > 0
                && within_budget
                && self
                    .admission_probe
                    .as_ref()
                    .is_none_or(|p| p(self.admission_margin()));
            if admissible {
                self.resident_bytes += weight;
                self.resident_footprint += footprint;
                let mut ledger = self.residency.borrow_mut();
                ledger.footprint += footprint;
                ledger.peak = ledger.peak.max(ledger.footprint);
                drop(ledger);
                self.resident[stage] = Some((runner, weight, footprint));
            } else if weight > 0 && self.resident_bytes > 0 {
                tracing::debug!(
                    stage,
                    footprint,
                    "residency full — the stage streams per pass (projection, not refusal)"
                );
            }
            // Peak WEIGHT residency — the cache metric the witnesses assert —
            // tracks the packed weight bytes, not the execution transient.
            self.peak_resident_weight_bytes = self
                .peak_resident_weight_bytes
                .max(self.resident_bytes.max(weight));

            if stage + 1 == self.stage_count {
                // Pipeline outputs: the final stage's own outputs, then the
                // surfaced intermediates (every stage ran, so all are filled).
                let final_n = self.output_ports.len() - self.surfaced_count;
                let mut result = outputs;
                for (k, slot) in surfaced.into_iter().enumerate() {
                    let bytes = slot.with_context(|| {
                        format!(
                            "surfaced output `{}` was never produced",
                            self.output_ports[final_n + k].name
                        )
                    })?;
                    result.push(OutputBuffer { bytes });
                }
                return Ok(result);
            }
            for (port, out) in outputs.into_iter().enumerate() {
                match &self.sinks[stage][port] {
                    Sink::Carry(name) => {
                        carried.insert(name.clone(), out.bytes);
                    }
                    Sink::Surface(k) => surfaced[*k] = Some(out.bytes),
                }
            }
        }
        bail!("the staged pipeline executed no stages")
    }
}

impl Drop for StagedRunner<'_> {
    fn drop(&mut self) {
        // Return this runner's whole share to the shared ledger so a dropped
        // sibling (e.g. the prefill seeder retired on window growth) frees its
        // resident footprint for the survivors. `saturating_sub` guards a
        // ledger already reset out from under us.
        if self.resident_footprint > 0 {
            let mut ledger = self.residency.borrow_mut();
            ledger.footprint = ledger.footprint.saturating_sub(self.resident_footprint);
        }
    }
}

impl LmSession for StagedRunner<'_> {
    fn input_port_info(&self) -> Vec<PortInfo> {
        self.input_ports.clone()
    }

    fn output_port_info(&self) -> Vec<PortInfo> {
        self.output_ports.clone()
    }

    fn input_index_by_name(&self, name: &str) -> Option<usize> {
        self.input_ports.iter().position(|p| p.name == name)
    }

    fn output_index_by_name(&self, name: &str) -> Option<usize> {
        self.output_ports.iter().position(|p| p.name == name)
    }

    fn execute(&mut self, inputs: &[&[u8]]) -> Result<Vec<OutputBuffer>> {
        self.execute_window(inputs)
    }

    fn pass_dispatched(&self) -> u64 {
        self.last_dispatched
    }

    fn pass_skipped(&self) -> u64 {
        self.last_skipped
    }

    fn residency_pressure(&self) -> (u64, u64) {
        (self.resident_footprint, self.residency_budget)
    }

    fn evict_resident(&mut self) {
        StagedRunner::evict_resident(self)
    }
}

impl SessionProvider for StagedRunner<'_> {
    fn session_for(&mut self, want: usize) -> Result<&mut dyn LmSession> {
        if want > self.window {
            bail!(
                "the sequence needs a window of {want} tokens but the staged pipeline was \
                 compiled at a fixed window of {}; recompile the stages with a larger \
                 context_length",
                self.window
            );
        }
        Ok(self)
    }

    fn max_window(&self) -> usize {
        self.window
    }
}

/// Decode the named ports of a `.holo` archive section (`Inputs`/`Outputs`)
/// without loading a session — k-form archives carry their port contract
/// independent of the (placeholder) weights.
fn archive_ports(archive: &[u8], kind: SectionKind) -> Result<Vec<PortInfo>> {
    let loader =
        HoloLoader::from_bytes(archive).map_err(|e| anyhow!("loading stage archive: {e:?}"))?;
    let plan = loader
        .into_plan()
        .map_err(|e| anyhow!("decoding stage archive sections: {e:?}"))?;
    let bytes = plan
        .section(kind)
        .map_err(|e| anyhow!("stage archive has no {kind:?} section: {e:?}"))?;
    let ports = decode_ports(bytes).map_err(|e| anyhow!("decoding {kind:?} ports: {e:?}"))?;
    Ok(ports
        .into_iter()
        .map(|p| PortInfo {
            name: p.name,
            dtype: p.dtype,
            element_count: p.element_count as usize,
            shape: p.shape.iter().map(|&d| d as usize).collect(),
        })
        .collect())
}

/// A shared-store adapter so a [`GrowableStagedSession`] can hand each
/// regrown [`StagedRunner`] the same underlying κ-store without moving it.
struct SharedStore(std::rc::Rc<std::cell::RefCell<Box<dyn KappaStore>>>);

impl KappaStore for SharedStore {
    fn resolve(&mut self, kappa: &str) -> Result<Vec<u8>> {
        self.0.borrow_mut().resolve(kappa)
    }

    fn invalidate(&mut self, kappa: &str) {
        self.0.borrow_mut().invalidate(kappa);
    }

    fn resolve_range(&mut self, kappa: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.0.borrow_mut().resolve_range(kappa, offset, len)
    }

    fn content_size(&mut self, kappa: &str) -> Result<u64> {
        self.0.borrow_mut().content_size(kappa)
    }
}

/// A length-adaptive staged provider: the window follows the SEQUENCE, never
/// the model (journey S4 / dictionary row `staged-window-growth`).
///
/// [`StagedRunner`] alone serves one fixed window — the window its stage
/// archives were compiled at. Compiling stages at the model's own context and
/// executing every token against that full window makes the first token of a
/// short prompt cost a full-context forward pass (O(context²) attention per
/// layer): a 10-token chat message against a 32k-context model is a
/// months-long "hang" in a browser tab. But stage archives are weightless
/// k-forms — recompiling them at a smaller window moves no weights and costs
/// well under a second — so the window can track the sequence the way the
/// monolithic [`GrowableSession`](crate::engine::GrowableSession) already
/// does: geometric buckets from the shared
/// [`geometric_window`](crate::engine::geometric_window) policy, capped at
/// the model's own context. Peak weight residency stays one stage; per-token
/// compute scales with the actual sequence, not the model.
pub struct GrowableStagedSession {
    config_json: String,
    keys: Vec<String>,
    kappas: Vec<String>,
    shapes: Vec<Vec<u64>>,
    dtypes: Vec<DType>,
    layers_per_stage: NonZeroU64,
    /// The window ceiling — the model's own context (or the validated
    /// download-time choice).
    max_window: usize,
    store: std::rc::Rc<std::cell::RefCell<Box<dyn KappaStore>>>,
    on_stage: Option<SharedStageObserver>,
    on_window: Option<Box<dyn FnMut(usize, bool)>>,
    residency_budget: u64,
    bound_by_footprint: bool,
    /// The one address-space residency ledger every runner this session wires
    /// (step / seeder / verify) shares, so their COMBINED footprint — not each
    /// runner's alone — is what admission is charged against.
    residency: SharedResidency,
    admission_probe: Option<std::rc::Rc<dyn Fn(u64) -> bool>>,
    verified: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,
    derived_store: Option<Box<dyn DerivedStore>>,
    derived_hits: u64,
    /// The quantized derived-artifact tier (row `quantized-transit`): window
    /// compiles rewrite projection weights onto these artifacts, and the
    /// derivation key carries the map — a quantized window is a different
    /// derivation, never a reinterpretation.
    quant: Option<hologram_ai_common::lower::QuantMap>,
    /// Weight-tier paging (row `lazy-constant-residency`): when set, every
    /// regrown window's stages load PAGED against the budget, the factory
    /// producing the provider's `Send` κ-resolver (independent of this
    /// session's store). A model whose stage weights exceed the window then
    /// runs — the arena is a bounded window over the κ-store.
    weight_paging: Option<(usize, std::rc::Rc<dyn Fn() -> crate::runner::PagedStore>)>,
    current: Option<(usize, StagedRunner<'static>)>,
}

impl GrowableStagedSession {
    /// Build from the streamed-download manifest (the same inputs as
    /// [`compile_stages`]) plus the κ-store the stages materialize against.
    /// `max_window` follows the monolithic rule: `Some(n)` is the validated
    /// download-time context, `None` means the model's own trained context
    /// (read from the config by the stage compiler on first growth).
    #[allow(clippy::too_many_arguments)] // the streamed-download manifest is parallel slices
    pub fn new(
        config_json: String,
        keys: Vec<String>,
        kappas: Vec<String>,
        shapes: Vec<Vec<u64>>,
        dtypes: Vec<DType>,
        context_length: Option<u64>,
        layers_per_stage: NonZeroU64,
        store: Box<dyn KappaStore>,
    ) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_str(&config_json).context("parsing config.json")?;
        // The model's own trained context is the window ceiling — required, not
        // fabricated: an absent `max_position_embeddings` cannot silently
        // become `u64::MAX` (which would let the window grow until OOM). The
        // parametric builder requires this key, so a graph reaches here only
        // with it present.
        let model_context = config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .context(
                "config.json is missing `max_position_embeddings` — the model's trained context \
                 is the window ceiling and cannot be fabricated",
            )?;
        let max_window = context_length.unwrap_or(model_context).min(model_context) as usize;
        ensure!(max_window >= 1, "the model declares no usable context");
        Ok(Self {
            config_json,
            keys,
            kappas,
            shapes,
            dtypes,
            layers_per_stage,
            max_window,
            store: std::rc::Rc::new(std::cell::RefCell::new(store)),
            on_stage: None,
            on_window: None,
            residency_budget: 0,
            bound_by_footprint: false,
            residency: SharedResidency::default(),
            admission_probe: None,
            verified: std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashSet::new())),
            derived_store: None,
            derived_hits: 0,
            quant: None,
            weight_paging: None,
            current: None,
        })
    }

    /// Install an environment headroom probe forwarded to every regrown
    /// runner (see [`AdmissionProbe`]).
    pub fn set_admission_probe(&mut self, p: std::rc::Rc<dyn Fn(u64) -> bool>) {
        self.admission_probe = Some(p);
    }

    /// (Raw weight bytes, element count) each stage of `archives` will
    /// materialize — summed per constant from its κ-map entries' manifest
    /// sizes, known BEFORE any byte moves.
    fn expected_stage_bytes(&self, archives: &[Vec<u8>]) -> Vec<(u64, u64)> {
        let size_of: std::collections::HashMap<&str, (u64, u64)> = self
            .kappas
            .iter()
            .zip(self.shapes.iter().zip(&self.dtypes))
            .map(|(kappa, (shape, dtype))| {
                let elems: u64 = shape.iter().product();
                (
                    kappa.as_str(),
                    (elems * dtype.byte_size().unwrap_or(1) as u64, elems),
                )
            })
            .collect();
        archives
            .iter()
            .map(|archive| {
                crate::materialize::kappa_requirements(archive)
                    .map(|reqs| {
                        reqs.iter()
                            .map(|r| {
                                let (bytes, elems) =
                                    size_of.get(r.kappa.as_str()).copied().unwrap_or((0, 0));
                                match r.range {
                                    // A ranged binding materializes only its
                                    // slice — the chunk, not the tensor.
                                    Some((_, len)) => {
                                        let elem_size = bytes.checked_div(elems).unwrap_or(1);
                                        (len, len / elem_size.max(1))
                                    }
                                    None => (bytes, elems),
                                }
                            })
                            .fold((0u64, 0u64), |(b, e), (rb, re)| (b + rb, e + re))
                    })
                    .unwrap_or((0, 0))
            })
            .collect()
    }

    /// Forward a residency budget (bytes) to every regrown runner: stages
    /// whose materialized sessions fit stay resident across tokens (row
    /// `stage-residency-cache`), so κ-store bandwidth is paid per window
    /// instead of per token. `0` (the default) is strict one-stage windowing.
    pub fn set_residency_budget(&mut self, bytes: u64) {
        self.residency_budget = bytes;
        if let Some((_, runner)) = self.current.as_mut() {
            runner.set_residency_budget(bytes);
        }
    }

    /// Treat the residency budget as a hard address-space ceiling (the wasm32
    /// tab), forwarded into every regrown runner: admission is then gated on
    /// each stage session's true runtime footprint plus a largest-walk reserve,
    /// so resident float-head chunks' retained F32 transients cannot accumulate
    /// into an allocation abort. Off by default (a 64-bit host, weight-cache
    /// residency). See [`StagedRunner::set_bound_by_footprint`].
    pub fn set_bound_by_footprint(&mut self, bounded: bool) {
        self.bound_by_footprint = bounded;
        if let Some((_, runner)) = self.current.as_mut() {
            runner.set_bound_by_footprint(bounded);
        }
    }

    /// High-water COMBINED resident footprint across every runner this session
    /// has wired (step + seeder + verify) — the peak address space its resident
    /// stage sessions have held at once. A witness asserts this never exceeds
    /// the ceiling; only meaningful under [`Self::set_bound_by_footprint`].
    pub fn peak_resident_footprint(&self) -> u64 {
        self.residency.borrow().peak
    }

    /// The CURRENT combined resident footprint on the shared ledger (not the
    /// high-water). Unlike [`Self::peak_resident_footprint`] this falls when a
    /// runner is evicted/dropped, so a witness can observe that the outgoing
    /// runner's residency is freed BEFORE a bucket regrows (row
    /// `lazy-constant-residency`): growth must not hold the old resident set
    /// while it compiles the wider bucket.
    pub fn resident_footprint(&self) -> u64 {
        self.residency.borrow().footprint
    }

    /// Share this session's address-space residency ledger with `other`, so a
    /// paired second model — a speculative DRAFT model (row
    /// `speculative-draft-pairing`) — and this target charge admission against
    /// ONE combined footprint. Without this, two `bound_by_footprint` sessions
    /// each gate residency against their own budget and, together, over-commit
    /// the wasm 4 GiB ceiling (the `RuntimeError: unreachable` allocation
    /// abort). This extends the one-ledger law that already binds a decode
    /// turn's step/seeder/verify runners across the model PAIR: whichever runner
    /// of either model admits a stage checks the COMBINED footprint, so their sum
    /// never exceeds the ceiling.
    ///
    /// Called before either session wires a runner (both build lazily); any
    /// footprint `other` already holds migrates onto the shared ledger, so it is
    /// safe to call after growth too.
    pub fn share_residency_with(&mut self, other: &mut GrowableStagedSession) {
        let ledger = std::rc::Rc::clone(&self.residency);
        // Re-point `other`'s current runner (if it already wired one) so its
        // held footprint moves onto the shared ledger rather than being
        // double-counted or stranded.
        if let Some((_, runner)) = other.current.as_mut() {
            runner.share_residency_ledger(std::rc::Rc::clone(&ledger));
        }
        other.residency = ledger;
    }

    /// Install a per-stage observer forwarded into every regrown runner:
    /// `(stage, stage_count, weight_bytes)` after each stage materializes.
    pub fn set_stage_observer(&mut self, f: Box<dyn FnMut(usize, usize, u64)>) {
        self.on_stage = Some(std::rc::Rc::from(std::cell::RefCell::new(f)));
    }

    /// Install a window observer: called with the bucket size when a window
    /// (re)builds, and whether it RESOLVED from the derived store (true) or
    /// compiled fresh (false) — the narration must not say "compiling" for a
    /// resolution.
    pub fn set_window_observer(&mut self, f: Box<dyn FnMut(usize, bool)>) {
        self.on_window = Some(f);
    }

    /// Pre-derive the next geometric window bucket's stage archives into the
    /// derived store, OFF the per-token path (row `idle-derivation`): no
    /// weights move (stage k-forms are weightless), the resident window is
    /// untouched, and a later crossing resolves the bucket instead of
    /// compiling it on the critical path. Returns the pre-derived bucket, or
    /// `None` at the ceiling. Abandoned speculation is ordinary derived
    /// content — evaporable by the same lifecycle that admitted it.
    pub fn prederive_next_window(&mut self) -> Result<Option<usize>> {
        let next = match &self.current {
            Some((current, _)) if *current >= self.max_window => return Ok(None),
            Some((current, _)) => {
                crate::engine::geometric_window(current.saturating_mul(2), self.max_window)
            }
            None => crate::engine::geometric_window(1, self.max_window),
        };
        if matches!(&self.current, Some((current, _)) if *current >= next) {
            return Ok(None);
        }
        // Derivation persists via the derived store; the result is dropped —
        // this call moves no weights and swaps no runner.
        let _ = self.stages_for_window(next)?;
        Ok(Some(next))
    }

    /// The stage count of the currently-resident window (0 before the first).
    pub fn stage_count(&self) -> usize {
        self.current.as_ref().map_or(0, |(_, r)| r.stage_count())
    }

    /// The decode-plan twin of [`Self::prederive_next_window`]: derive the
    /// next geometric decode BUCKET's archives into the derived store, off
    /// the per-token path. `current` is the caller's resident bucket (the
    /// decode runner lives with its [`crate::decode::DecodeSession`], not
    /// here). Returns the pre-derived bucket, or `None` at the ceiling.
    pub fn prederive_next_decode_bucket(&mut self, current: usize) -> Result<Option<usize>> {
        if current >= self.max_window {
            return Ok(None);
        }
        let next =
            crate::engine::geometric_window(current.saturating_mul(2).max(1), self.max_window);
        if next <= current {
            return Ok(None);
        }
        let _ = self.decode_stages_for_bucket(next, 1)?;
        Ok(Some(next))
    }
}

impl GrowableStagedSession {
    /// Load `archives` into a runner wired with the session's store,
    /// observers, budget, admission probe, and verified-κ set — the shared
    /// back half of the whole-window and decode-plan providers.
    fn wire_runner(&mut self, archives: Vec<Vec<u8>>) -> Result<StagedRunner<'static>> {
        let expected = self.expected_stage_bytes(&archives);
        let mut runner = StagedRunner::from_archives(
            archives,
            Box::new(SharedStore(std::rc::Rc::clone(&self.store))),
        )
        .context("loading the staged pipeline")?;
        if let Some(hook) = &self.on_stage {
            let hook = std::rc::Rc::clone(hook);
            runner.set_stage_observer(Box::new(move |s, n, b| {
                (hook.borrow_mut())(s, n, b);
            }));
        }
        runner.set_residency_budget(self.residency_budget);
        runner.set_bound_by_footprint(self.bound_by_footprint);
        // Share the ONE address-space ledger: the step, seeder, and verify
        // runners of a decode turn are charged against their combined footprint.
        runner.share_residency_ledger(std::rc::Rc::clone(&self.residency));
        runner.share_verified_set(std::rc::Rc::clone(&self.verified));
        runner.set_expected_stage_bytes(expected);
        if let Some(probe) = &self.admission_probe {
            let probe = std::rc::Rc::clone(probe);
            runner.set_admission_probe(Box::new(move |margin| probe(margin)));
        }
        if let Some((budget, factory)) = &self.weight_paging {
            let factory = std::rc::Rc::clone(factory);
            runner.set_weight_paging(*budget, Box::new(move || factory()));
        }
        Ok(runner)
    }

    /// Enable weight-tier paging for every regrown window (row
    /// `lazy-constant-residency`): stages load PAGED against `budget`,
    /// `make_resolver` producing the provider's `Send` κ-resolver. Set before
    /// the first generation — paging is a property of the session's host, not
    /// a mid-session switch.
    pub fn set_weight_paging(
        &mut self,
        budget: usize,
        make_resolver: std::rc::Rc<dyn Fn() -> crate::runner::PagedStore>,
    ) {
        self.weight_paging = Some((budget, make_resolver));
    }

    /// A **decode-plan** pipeline runner over a bucket ≥ `want` (geometric,
    /// ceilinged by the model's context) — the decode twin of
    /// [`SessionProvider::session_for`]. The caller owns the runner (a
    /// [`crate::decode::DecodeSession`] holds it together with its carried
    /// K/V buffers); this session stays the archive factory across regrows,
    /// so derived-store hits, the verified-κ set, and observers all carry.
    pub fn decode_runner_for(&mut self, want: usize) -> Result<StagedRunner<'static>> {
        self.chunk_runner_for(want, 1)
    }

    /// [`Self::decode_runner_for`] parametric in the chunk (row
    /// `chunked-prefill`): the seeding pipeline processing `chunk` positions
    /// per pass over the same bucket geometry.
    pub fn chunk_runner_for(&mut self, want: usize, chunk: u64) -> Result<StagedRunner<'static>> {
        ensure!(
            want <= self.max_window,
            "the sequence needs a decode bucket of {want} rows but the model's context length \
             is {}",
            self.max_window
        );
        let bucket = crate::engine::geometric_window(want.max(1), self.max_window);
        tracing::info!(bucket, want, chunk, "building decode-plan bucket");
        let (archives, resolved) = self.decode_stages_for_bucket(bucket, chunk)?;
        if let Some(f) = self.on_window.as_mut() {
            f(bucket, resolved);
        }
        self.wire_runner(archives)
            .with_context(|| format!("loading the {bucket}-row decode pipeline (chunk {chunk})"))
    }
}

impl SessionProvider for GrowableStagedSession {
    fn session_for(&mut self, want: usize) -> Result<&mut dyn LmSession> {
        if want > self.max_window {
            bail!(
                "the sequence needs a window of {want} tokens but the model's context length \
                 is {}",
                self.max_window
            );
        }
        let fits = matches!(&self.current, Some((cur, _)) if *cur >= want);
        if !fits {
            let window = crate::engine::geometric_window(want, self.max_window);
            tracing::info!(window, want, "building staged generation window");
            // Drop the previous window first: peak residency stays one stage.
            self.current = None;
            let (archives, resolved) = self.stages_for_window(window)?;
            if let Some(f) = self.on_window.as_mut() {
                f(window, resolved);
            }
            let runner = self
                .wire_runner(archives)
                .with_context(|| format!("loading the {window}-token staged window"))?;
            self.current = Some((window, runner));
        }
        Ok(&mut self.current.as_mut().expect("window just ensured").1)
    }

    fn max_window(&self) -> usize {
        self.max_window
    }
}

// ── Derived-artifact closure (row `derived-artifact-kappa`) ──────────────────

/// A derived-artifact store: the known set closes over deterministic
/// derivation (resource model, Closure). A window's stage archives are
/// computed deterministically from κ inputs (config, manifest, window,
/// partition — `deterministic-compile` witnesses bit-identity), so they are
/// themselves content: derived once, persisted under their derivation key
/// with their recorded content-κs, and RESOLVED by later sessions instead of
/// re-derived. Soundness is inherited: content verifies against its recorded
/// κ at load (once per load — off the per-token path), a mismatch evaporates
/// the entry and recovery is derivation itself, and everything here is
/// re-derivable locally, so a wrong prior can never dead-end or execute
/// unverified content.
pub trait DerivedStore {
    /// The archives + recorded content-κs stored under `key`, if present.
    fn load(&mut self, key: &str) -> Option<(Vec<Vec<u8>>, Vec<String>)>;
    /// Persist `stages` (+ their content-κs) under `key`.
    fn store(&mut self, key: &str, stages: &[Vec<u8>], kappas: &[String]);
    /// Evaporate a corrupted or stale entry (the unpin of this tier).
    fn evaporate(&mut self, key: &str);
}

/// [`DerivedStore`] over a directory: `{key}/{i}.holo` + `{key}/kappas.json`
/// (the native mirror of the browser's `models/<dir>/derived/` layout).
pub struct DirDerivedStore {
    root: std::path::PathBuf,
}

impl DirDerivedStore {
    /// Create a derived-artifact store rooted at `root`.
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl DerivedStore for DirDerivedStore {
    fn load(&mut self, key: &str) -> Option<(Vec<Vec<u8>>, Vec<String>)> {
        let dir = self.root.join(key);
        let kappas: Vec<String> =
            serde_json::from_slice(&std::fs::read(dir.join("kappas.json")).ok()?).ok()?;
        let mut stages = Vec::with_capacity(kappas.len());
        for i in 0..kappas.len() {
            stages.push(std::fs::read(dir.join(format!("{i}.holo"))).ok()?);
        }
        Some((stages, kappas))
    }

    fn store(&mut self, key: &str, stages: &[Vec<u8>], kappas: &[String]) {
        let dir = self.root.join(key);
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(&dir)?;
            for (i, stage) in stages.iter().enumerate() {
                std::fs::write(dir.join(format!("{i}.holo")), stage)?;
            }
            std::fs::write(
                dir.join("kappas.json"),
                serde_json::to_vec(kappas).expect("κ list serializes"),
            )
        };
        // Persistence is an optimization; a failed write only costs a
        // future re-derivation.
        let _ = write();
    }

    fn evaporate(&mut self, key: &str) {
        let _ = std::fs::remove_dir_all(self.root.join(key));
    }
}

impl GrowableStagedSession {
    /// Install a derived-artifact store: window regrows resolve their stage
    /// archives from it (content-verified at load) and persist fresh
    /// derivations into it. Without one, every window compiles fresh — the
    /// same semantics, more derivation.
    pub fn set_derived_store(&mut self, store: Box<dyn DerivedStore>) {
        self.derived_store = Some(store);
    }

    /// Install the quantized derived-artifact map (row `quantized-transit`):
    /// every window compiled from here on binds projection weights to their
    /// quantized artifacts. Set before the first generation — the tier is a
    /// property of the session's model, stated once, never a mid-session
    /// mode switch.
    pub fn set_quant_map(&mut self, quant: hologram_ai_common::lower::QuantMap) {
        self.quant = Some(quant);
    }

    /// Window regrows served from the derived store instead of compiled —
    /// the derivation-reuse instrument.
    pub fn derived_hits(&self) -> u64 {
        self.derived_hits
    }

    /// Stage materializations of the resident window's runner (0 before the
    /// first window) — the cross-turn bandwidth instrument: a second
    /// generation over a warm session adds none while the resident set holds.
    pub fn materialization_count(&self) -> u64 {
        self.current
            .as_ref()
            .map_or(0, |(_, r)| r.materialization_count())
    }

    /// The derivation key of this session's stages at `window`: a κ over the
    /// exact inputs the derivation is a deterministic function of. Two
    /// sessions with identical inputs resolve each other's artifacts; any
    /// input change is a different key, never a reinterpretation.
    fn derivation_key(&self, window: usize) -> String {
        let mut ingest = format!(
            "stage-archives:v3:window={window}:layers_per_stage={}:context={}:config=",
            self.layers_per_stage, self.max_window
        )
        .into_bytes();
        ingest.extend_from_slice(self.config_json.as_bytes());
        for (key, kappa) in self.keys.iter().zip(&self.kappas) {
            ingest.extend_from_slice(key.as_bytes());
            ingest.push(b'=');
            ingest.extend_from_slice(kappa.as_bytes());
            ingest.push(b';');
        }
        // The quantized tier is derivation INPUT: a quantized window and a
        // wide window are different artifacts under different keys.
        if let Some(quant) = &self.quant {
            ingest.extend_from_slice(b":quant-int8:");
            let mut entries: Vec<_> = quant.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (wide, (artifact, out, inf)) in entries {
                ingest.extend_from_slice(format!("{wide}>{artifact}@{out}x{inf};").as_bytes());
            }
        }
        crate::materialize::kappa_of(&ingest)
    }

    /// Resolve the window's stage archives: derived store first (verified at
    /// load; a mismatch evaporates the entry — derive-as-recovery), else
    /// compile and persist the fresh derivation.
    fn stages_for_window(&mut self, window: usize) -> Result<(Vec<Vec<u8>>, bool)> {
        let key = self.derivation_key(window);
        if let Some(store) = self.derived_store.as_mut() {
            if let Some((stages, kappas)) = store.load(&key) {
                let intact = stages.len() == kappas.len()
                    && !stages.is_empty()
                    && stages
                        .iter()
                        .zip(&kappas)
                        .all(|(s, k)| crate::materialize::kappa_of(s) == *k);
                if intact {
                    self.derived_hits += 1;
                    tracing::info!(window, "staged window resolved from derived κ-store");
                    return Ok((stages, true));
                }
                // Corrupted or torn: evaporate and recover by deriving.
                store.evaporate(&key);
            }
        }
        let stages = compile_stages_with(
            &self.config_json,
            &self.keys,
            &self.kappas,
            &self.shapes,
            &self.dtypes,
            Some(window as u64),
            self.layers_per_stage,
            self.quant.as_ref(),
        )
        .with_context(|| format!("compiling a {window}-token staged window"))?;
        if let Some(store) = self.derived_store.as_mut() {
            let kappas: Vec<String> = stages
                .iter()
                .map(|s| crate::materialize::kappa_of(s))
                .collect();
            store.store(&key, &stages, &kappas);
        }
        Ok((stages, false))
    }

    /// The decode-plan twin of [`Self::stages_for_window`]: resolve or
    /// compile the `bucket`-row, `chunk`-position decode pipeline under its
    /// OWN derivation key — a decode pipeline and a whole-window pipeline
    /// are different derivations of the same inputs, never
    /// reinterpretations, and each (bucket, chunk) pair is its own artifact.
    fn decode_stages_for_bucket(
        &mut self,
        bucket: usize,
        chunk: u64,
    ) -> Result<(Vec<Vec<u8>>, bool)> {
        let key = crate::materialize::kappa_of(
            format!(
                "decode-archives:v2:bucket={bucket}:chunk={chunk}:base={}",
                self.derivation_key(0)
            )
            .as_bytes(),
        );
        if let Some(store) = self.derived_store.as_mut() {
            if let Some((stages, kappas)) = store.load(&key) {
                let intact = stages.len() == kappas.len()
                    && !stages.is_empty()
                    && stages
                        .iter()
                        .zip(&kappas)
                        .all(|(s, k)| crate::materialize::kappa_of(s) == *k);
                if intact {
                    self.derived_hits += 1;
                    tracing::info!(bucket, "decode pipeline resolved from derived κ-store");
                    return Ok((stages, true));
                }
                store.evaporate(&key);
            }
        }
        let stages = compile_chunk_stages(
            &self.config_json,
            &self.keys,
            &self.kappas,
            &self.shapes,
            &self.dtypes,
            bucket as u64,
            chunk,
            self.layers_per_stage,
            self.quant.as_ref(),
        )
        .with_context(|| format!("compiling a {bucket}-row decode pipeline (chunk {chunk})"))?;
        if let Some(store) = self.derived_store.as_mut() {
            let kappas: Vec<String> = stages
                .iter()
                .map(|s| crate::materialize::kappa_of(s))
                .collect();
            store.store(&key, &stages, &kappas);
        }
        Ok((stages, false))
    }

    /// Compile (and cache in the derived store) the staged VERIFY pipeline at
    /// `bucket` rows and `chunk` positions (row `speculative-decode`): the same
    /// κ bindings and quant tier as the decode pipeline, but the all-positions
    /// head, so a `chunk`-token draft verifies in one `M = chunk` pass.
    fn verify_stages_for_bucket(&mut self, bucket: usize, chunk: u64) -> Result<Vec<Vec<u8>>> {
        let key = crate::materialize::kappa_of(
            format!(
                "verify-archives:v1:bucket={bucket}:chunk={chunk}:base={}",
                self.derivation_key(0)
            )
            .as_bytes(),
        );
        if let Some(store) = self.derived_store.as_mut() {
            if let Some((stages, kappas)) = store.load(&key) {
                let intact = stages.len() == kappas.len()
                    && !stages.is_empty()
                    && stages
                        .iter()
                        .zip(&kappas)
                        .all(|(s, k)| crate::materialize::kappa_of(s) == *k);
                if intact {
                    self.derived_hits += 1;
                    return Ok(stages);
                }
                store.evaporate(&key);
            }
        }
        let stages = compile_verify_stages(
            &self.config_json,
            &self.keys,
            &self.kappas,
            &self.shapes,
            &self.dtypes,
            bucket as u64,
            chunk,
            self.layers_per_stage,
            self.quant.as_ref(),
        )
        .with_context(|| format!("compiling a {bucket}-row verify pipeline (chunk {chunk})"))?;
        if let Some(store) = self.derived_store.as_mut() {
            let kappas: Vec<String> = stages
                .iter()
                .map(|s| crate::materialize::kappa_of(s))
                .collect();
            store.store(&key, &stages, &kappas);
        }
        Ok(stages)
    }

    /// A staged VERIFY runner at `bucket` rows (which MUST match the decode
    /// session's bucket — they share the carried past) and draft width `chunk`
    /// (row `speculative-decode`).
    pub fn verify_runner_for(
        &mut self,
        bucket: usize,
        chunk: u64,
    ) -> Result<StagedRunner<'static>> {
        ensure!(
            bucket <= self.max_window,
            "verify bucket {bucket} exceeds the model's context length {}",
            self.max_window
        );
        let stages = self.verify_stages_for_bucket(bucket, chunk)?;
        self.wire_runner(stages)
            .with_context(|| format!("loading the {bucket}-row verify pipeline (chunk {chunk})"))
    }
}
