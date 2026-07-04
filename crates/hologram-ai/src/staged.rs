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
use crate::materialize::{materialize_archive, KappaStore};
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
    ensure!(
        keys.len() == kappas.len() && keys.len() == shapes.len() && keys.len() == dtypes.len(),
        "manifest slices disagree: {} keys, {} κs, {} shapes, {} dtypes",
        keys.len(),
        kappas.len(),
        shapes.len(),
        dtypes.len()
    );
    let config: serde_json::Value =
        serde_json::from_str(config_json).context("parsing config.json")?;
    let graphs = hologram_ai_safetensors::parametric::build_parametric_stage_graphs(
        &config,
        keys,
        dtypes,
        context_length,
        layers_per_stage,
    )?;

    let mut bound = vec![false; keys.len()];
    let mut archives = Vec::with_capacity(graphs.len());
    for (stage, mut graph) in graphs.into_iter().enumerate() {
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
            let info = TensorInfo::new(dtypes[i], shape_from_concrete(&shapes[i]));
            graph.tensor_info.insert(id, info.clone());
            graph.params.insert(
                id,
                AiParam::External {
                    kappa: kappas[i].clone(),
                    info,
                },
            );
            bound[i] = true;
        }
        let archive = ModelCompiler::default()
            .compile(ModelSource::AiGraph(graph))
            .with_context(|| format!("compiling stage {stage}"))?;
        archives.push(archive.bytes);
    }

    if let Some(i) = bound.iter().position(|b| !b) {
        bail!(
            "manifest tensor `{}` is consumed by no stage graph — the staged \
             partition must cover the model's tensors exactly",
            keys[i]
        );
    }
    Ok(archives)
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
}

/// Resolves the k-form archive bytes of stage `i` — a `Vec` of precompiled
/// archives natively, an OPFS read in the browser. Archives are weightless
/// k-forms (structure + κ-bindings), so resolving one moves no parameters.
pub type StageResolver<'a> = Box<dyn FnMut(usize) -> Result<Vec<u8>> + 'a>;

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
/// Implements [`LmSession`] + [`SessionProvider`], so
/// [`generate_stream`](crate::commands::generate::generate_stream) drives it
/// unchanged.
pub struct StagedRunner<'a> {
    resolve_stage: StageResolver<'a>,
    store: Box<dyn KappaStore + 'a>,
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
}

impl<'a> StagedRunner<'a> {
    /// Build a runner over `stage_count` stages resolved on demand through
    /// `resolve_stage`, materializing κs against `store`. Reads the LM port
    /// contract from the k-form archives' port sections (weight-free): stage
    /// 0 must declare an `input_ids` input and the final stage a `logits`
    /// output.
    pub fn new(
        stage_count: usize,
        mut resolve_stage: StageResolver<'a>,
        store: Box<dyn KappaStore + 'a>,
    ) -> Result<Self> {
        ensure!(
            stage_count >= 1,
            "a staged pipeline needs at least one stage"
        );

        let first = resolve_stage(0).context("resolving the stage 0 archive")?;
        let input_ports =
            archive_ports(&first, SectionKind::Inputs).context("reading stage 0 input ports")?;
        drop(first);
        let last = resolve_stage(stage_count - 1).context("resolving the final stage archive")?;
        let output_ports = archive_ports(&last, SectionKind::Outputs)
            .context("reading final stage output ports")?;
        drop(last);

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
            stage_count,
            input_ports,
            output_ports,
            window,
            stage_weight_bytes: vec![0; stage_count],
            peak_resident_weight_bytes: 0,
        })
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
        let mut carried: Vec<Vec<u8>> = Vec::new();
        for stage in 0..self.stage_count {
            let archive = (self.resolve_stage)(stage)
                .with_context(|| format!("resolving the stage {stage} archive"))?;
            let mut counting = CountingStore {
                inner: self.store.as_mut(),
                bytes: 0,
            };
            let material = materialize_archive(&archive, &mut counting)
                .with_context(|| format!("materializing stage {stage}"))?;
            drop(archive);
            self.stage_weight_bytes[stage] = counting.bytes;
            self.peak_resident_weight_bytes = self.peak_resident_weight_bytes.max(counting.bytes);

            let mut runner = HoloRunner::from_bytes(material)
                .with_context(|| format!("loading stage {stage}"))?;
            let refs: Vec<&[u8]> = if stage == 0 {
                inputs.to_vec()
            } else {
                carried.iter().map(Vec::as_slice).collect()
            };
            let outputs = runner
                .execute(&refs)
                .with_context(|| format!("executing stage {stage}"))?;
            if stage + 1 == self.stage_count {
                return Ok(outputs);
            }
            carried = outputs.into_iter().map(|o| o.bytes).collect();
            // `runner` — this stage's materialized weights — drops here,
            // before the next stage materializes: the residency window.
        }
        bail!("the staged pipeline executed no stages")
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
