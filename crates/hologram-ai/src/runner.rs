//! HoloRunner — shape-context-aware execution wrapper for compiled archives.

use anyhow::Context;
use hologram_ai_common::exec_context::ShapeContextGraph;
use tracing::info;

use crate::compiler::HoloArchive;

/// Rebuild a compiled archive adding an extra section.
///
/// Preserves all existing sections from the source archive so that
/// layer headers, model metadata, tokenizer data, etc. are not lost.
/// Uses a single unpack/repack cycle internally.
/// Pre-loaded archive ready for repeated shape-aware execution.
///
/// Supports both single-graph archives (non-LLM models) and pipeline archives
/// (LLM with prefill + decode sub-models). For pipeline archives, the first
/// `execute()` call runs the prefill model; subsequent calls run the decode model
/// (when KV cache is wired up — currently both use the prefill model).
///
/// Load once with [`HoloRunner::from_bytes`], then call [`HoloRunner::execute`]
/// many times with different inputs.
/// Owned archive storage — either heap-allocated Vec or memory-mapped file.
enum ArchiveStorage {
    Owned(Vec<u8>),
    Mmap(memmap2::Mmap),
}

impl AsRef<[u8]> for ArchiveStorage {
    fn as_ref(&self) -> &[u8] {
        match self {
            ArchiveStorage::Owned(v) => v,
            ArchiveStorage::Mmap(m) => m,
        }
    }
}

pub struct HoloRunner {
    /// Backing storage: mmap or heap. MUST be listed first so it's dropped last
    /// (LoadedPlan borrows from it).
    _storage: ArchiveStorage,
    /// The prefill model plan (first component, or only component for non-LLM).
    plan: hologram::LoadedPlan,
    shape_ctx: Option<ShapeContextGraph>,
    /// Pre-compiled execution tape for prefill.
    tape: hologram::hologram_exec::tape::EnumTape,
    /// Optional decode model (second component in LLM pipeline archives).
    /// When present, `execute_with_kv` switches to this after step 0.
    decode_plan: Option<hologram::LoadedPlan>,
    decode_tape: Option<hologram::hologram_exec::tape::EnumTape>,
    /// Pre-computed shape overrides for decode (seq=1). Extracted from the
    /// compiled node_shapes at load time — no walk_shape_context needed per
    /// step. Provides input_metas that Q4 LUT-GEMM kernels require.
    decode_shape_map: std::collections::HashMap<u32, Vec<usize>>,
    /// Optional verification model (third component in LLM pipeline archives).
    /// Compiled at seq=N for batch speculative decoding verification.
    verify_plan: Option<hologram::LoadedPlan>,
    verify_tape: Option<hologram::hologram_exec::tape::EnumTape>,
    /// Persistent weight cache for LUT-GEMM. Deserialized quantized weights
    /// are cached here across execution calls, avoiding per-step rkyv overhead.
    weight_cache: parking_lot::RwLock<hologram::WeightCache>,
    /// Optional patch pruning configuration (ViT models with PixelPrune).
    /// When present, `execute()` preprocesses the pixel input to produce
    /// `kept_indices` before feeding the compiled graph.
    patch_prune: Option<hologram_ai_common::PatchPruneContext>,
}

impl HoloRunner {
    /// Load a runner from raw archive bytes (heap-allocated).
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        Self::from_storage(ArchiveStorage::Owned(bytes))
    }

    /// Load a runner from a `.holo` file on disk using memory-mapping.
    ///
    /// This avoids reading the entire archive (often multi-GB) into heap.
    /// Weights are accessed on-demand via page faults, so RSS stays low.
    ///
    /// If the archive is compressed, decompresses to a cache file for instant
    /// loading on subsequent runs. Cache location is controlled by `cache_dir`:
    /// - `None` — falls back to `HologramConfig` then caches next to the archive
    /// - `Some(dir)` — cache in the given directory (e.g., `~/.hologram/cache/`)
    pub fn from_path(
        path: &std::path::Path,
        cache_dir: Option<&std::path::Path>,
        config_path: Option<&std::path::Path>,
    ) -> anyhow::Result<Self> {
        // Load config: explicit path > standard search.
        let config = match config_path {
            Some(p) => hologram::config::HologramConfig::load_file(p).unwrap_or_default(),
            None => hologram::config::HologramConfig::load(),
        };
        // CLI cache_dir > config cache.dir > default (next to archive).
        let config_cache = config.cache_dir();
        let cache_dir = cache_dir.or(config_cache.as_deref());
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening archive {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("memory-mapping archive {}", path.display()))?;

        // If compressed, decompress to a cache file for instant loading.
        if hologram::hologram_archive::is_compressed(&mmap) {
            let cache_path = match cache_dir {
                Some(dir) => {
                    std::fs::create_dir_all(dir)
                        .with_context(|| format!("creating cache dir {}", dir.display()))?;
                    let stem = path.file_name().unwrap_or_default();
                    dir.join(format!("{}.cache", stem.to_string_lossy()))
                }
                None => path.with_extension("holo.cache"),
            };

            if cache_path.exists() {
                let cache_file = std::fs::File::open(&cache_path)
                    .with_context(|| format!("opening cache {}", cache_path.display()))?;
                let cache_mmap = unsafe { memmap2::Mmap::map(&cache_file) }
                    .with_context(|| format!("mmap cache {}", cache_path.display()))?;
                return Self::from_storage(ArchiveStorage::Mmap(cache_mmap));
            }

            eprintln!("decompressing to {} (one-time)...", cache_path.display());
            if let Some(uncompressed) = hologram::hologram_archive::decompress_archive(&mmap)
                .with_context(|| "decompressing archive")?
            {
                std::fs::write(&cache_path, &uncompressed)
                    .with_context(|| format!("writing cache {}", cache_path.display()))?;
                let cache_file = std::fs::File::open(&cache_path)?;
                let cache_mmap = unsafe { memmap2::Mmap::map(&cache_file) }?;
                return Self::from_storage(ArchiveStorage::Mmap(cache_mmap));
            }
        }

        Self::from_storage(ArchiveStorage::Mmap(mmap))
    }

    fn from_storage(storage: ArchiveStorage) -> anyhow::Result<Self> {
        let bytes: &[u8] = storage.as_ref();

        // SAFETY: storage outlives all plans created here.
        let probe = unsafe { hologram::load_from_bytes_zero_copy(bytes) }
            .map_err(|e| anyhow::anyhow!("loading archive: {e}"))?;

        // Check if this is a pipeline archive (has SECTION_PIPELINE header).
        let is_pipeline = probe
            .sections()
            .entries
            .iter()
            .any(|e| e.kind == hologram::hologram_archive::section::SECTION_PIPELINE);

        if is_pipeline {
            // Pipeline archive: load the first (or only) model component.
            let weights_start = probe.header().weights_offset as usize;

            let pipeline_entry = probe
                .sections()
                .find(hologram::hologram_archive::section::SECTION_PIPELINE)
                .ok_or_else(|| anyhow::anyhow!("pipeline section missing"))?;
            let ps = pipeline_entry.offset as usize;
            let pe = ps + pipeline_entry.size as usize;
            let ph: hologram::hologram_archive::writer::pipeline_writer::PipelineHeader =
                rkyv::from_bytes::<
                    hologram::hologram_archive::writer::pipeline_writer::PipelineHeader,
                    rkyv::rancor::Error,
                >(&bytes[ps..pe])
                .map_err(|e| anyhow::anyhow!("parsing pipeline header: {e}"))?;

            // Load the first model component.
            let first = ph
                .models
                .first()
                .ok_or_else(|| anyhow::anyhow!("pipeline has no models"))?;
            let model_start = weights_start + first.offset as usize;
            let model_end = model_start + first.size as usize;
            if model_end > bytes.len() {
                anyhow::bail!("sub-archive out of bounds");
            }
            let model_slice = &bytes[model_start..model_end];

            let mut plan = unsafe { hologram::load_from_bytes_zero_copy(model_slice) }
                .map_err(|e| anyhow::anyhow!("loading model plan: {e}"))?;

            // Resolve shared weights via dedup index if available.
            let dedup_index = probe
                .sections()
                .find(hologram::hologram_archive::section::SECTION_WEIGHT_DEDUP)
                .and_then(|entry| {
                    let s = entry.offset as usize;
                    let e = s + entry.size as usize;
                    if e <= bytes.len() {
                        hologram::hologram_archive::WeightDedupIndex::from_bytes(&bytes[s..e]).ok()
                    } else {
                        None
                    }
                });

            if let Some(ref idx) = dedup_index {
                if plan.weights().is_empty() {
                    let wrapper_weights = probe.weights();
                    if let Some(entry) = idx.find_component(&first.name) {
                        let w_start = entry.offset as usize;
                        let w_end = w_start + entry.size as usize;
                        if w_end <= wrapper_weights.len() {
                            unsafe {
                                plan.set_weights_borrowed(&wrapper_weights[w_start..w_end]);
                            }
                        }
                    }
                }
            }

            let shape_ctx = read_shape_context_from_plan(&plan, model_slice)?;
            let tape = hologram::build_tape_from_plan(&plan)
                .map_err(|e| anyhow::anyhow!("building prefill tape: {e}"))?;

            // Load decode model (second component) if present.
            let (decode_plan, decode_tape) = if ph.models.len() >= 2 {
                let second = &ph.models[1];
                let d_start = weights_start + second.offset as usize;
                let d_end = d_start + second.size as usize;
                if d_end > bytes.len() {
                    anyhow::bail!("decode sub-archive out of bounds");
                }
                let d_slice = &bytes[d_start..d_end];
                let mut d_plan = unsafe { hologram::load_from_bytes_zero_copy(d_slice) }
                    .map_err(|e| anyhow::anyhow!("loading decode plan: {e}"))?;

                // Share weights from prefill → decode. Both components were compiled
                // from the same AiGraph with identical constant ordering, so weight
                // offsets are the same. Just borrow the prefill's weight buffer.
                if d_plan.weights().is_empty() && !plan.weights().is_empty() {
                    unsafe {
                        d_plan.set_weights_borrowed(plan.weights());
                    }
                    info!(
                        decode_weights = plan.weights().len(),
                        "decode shares prefill weights"
                    );
                }

                let d_tape = hologram::build_tape_from_plan(&d_plan)
                    .map_err(|e| anyhow::anyhow!("building decode tape: {e}"))?;
                info!("loaded decode model (seq=1) for LLM pipeline");
                (Some(d_plan), Some(d_tape))
            } else {
                (None, None)
            };

            // Load verify model (third component) if present.
            let (verify_plan, verify_tape) = if ph.models.len() >= 3 {
                let third = &ph.models[2];
                let v_start = weights_start + third.offset as usize;
                let v_end = v_start + third.size as usize;
                if v_end > bytes.len() {
                    anyhow::bail!("verify sub-archive out of bounds");
                }
                let v_slice = &bytes[v_start..v_end];
                let mut v_plan = unsafe { hologram::load_from_bytes_zero_copy(v_slice) }
                    .map_err(|e| anyhow::anyhow!("loading verify plan: {e}"))?;
                if v_plan.weights().is_empty() && !plan.weights().is_empty() {
                    unsafe {
                        v_plan.set_weights_borrowed(plan.weights());
                    }
                }
                let v_tape = hologram::build_tape_from_plan(&v_plan)
                    .map_err(|e| anyhow::anyhow!("building verify tape: {e}"))?;
                info!("loaded verify model (seq=8) for speculative decoding");
                (Some(v_plan), Some(v_tape))
            } else {
                (None, None)
            };

            // Pre-compute decode shape map from the compiled graph's node_shapes.
            // The decode graph has fully concrete shapes (seq=1), so these
            // are exact — no runtime resolution needed.
            let decode_shape_map = decode_plan
                .as_ref()
                .map(|dp| {
                    dp.graph()
                        .node_shapes
                        .iter()
                        .map(|(nid, shape)| (nid.index(), shape.clone()))
                        .collect()
                })
                .unwrap_or_default();

            let patch_prune = read_patch_prune_from_plan(&plan, model_slice);

            let runner = Self {
                _storage: storage,
                plan,
                shape_ctx,
                tape,
                decode_plan,
                decode_tape,
                decode_shape_map,
                verify_plan,
                verify_tape,
                weight_cache: parking_lot::RwLock::new(hologram::WeightCache::new()),
                patch_prune,
            };
            // Pre-warm dequant cache: populate f32 expansion for all Q4/Q8
            // constants so decode steps never pay the dequant overhead.
            #[cfg(target_os = "macos")]
            {
                let sg = runner.plan.graph();
                let mut wc = runner.weight_cache.write();
                wc.prewarm_q4(&runner.tape, &sg.constants, runner.plan.weights());
                wc.prewarm_q8(&runner.tape, &sg.constants, runner.plan.weights());
                if let Some(ref dt) = runner.decode_tape {
                    // Use the decode plan's own ConstantStore — with inlined Bytes
                    // constants (Plan 077), each graph has its own constants and the
                    // ConstantIds differ between prefill and decode.
                    let decode_constants = runner
                        .decode_plan
                        .as_ref()
                        .map(|dp| &dp.graph().constants);
                    let decode_weights = runner
                        .decode_plan
                        .as_ref()
                        .map(|dp| dp.weights())
                        .unwrap_or(&[]);
                    if let Some(dc) = decode_constants {
                        wc.prewarm_q4(dt, dc, decode_weights);
                        wc.prewarm_q8(dt, dc, decode_weights);
                    }
                }
            }
            Ok(runner)
        } else {
            // Legacy single-graph archive (backward compat).
            let shape_ctx = read_shape_context_from_plan(&probe, bytes)?;
            let tape = hologram::build_tape_from_plan(&probe)
                .map_err(|e| anyhow::anyhow!("building tape: {e}"))?;

            let patch_prune = read_patch_prune_from_plan(&probe, bytes);

            let runner = Self {
                _storage: storage,
                plan: probe,
                shape_ctx,
                tape,
                decode_plan: None,
                decode_tape: None,
                decode_shape_map: std::collections::HashMap::new(),
                verify_plan: None,
                verify_tape: None,
                weight_cache: parking_lot::RwLock::new(hologram::WeightCache::new()),
                patch_prune,
            };
            #[cfg(target_os = "macos")]
            {
                let sg = runner.plan.graph();
                let mut wc = runner.weight_cache.write();
                wc.prewarm_q4(&runner.tape, &sg.constants, runner.plan.weights());
                wc.prewarm_q8(&runner.tape, &sg.constants, runner.plan.weights());
            }
            Ok(runner)
        }
    }

    /// Execute the compiled graph with the given inputs.
    ///
    /// When a `ShapeContextGraph` is available, projects runtime input shapes
    /// through the graph to produce correct per-node shapes. This enables
    /// variable-length execution (runtime seq_len != compiled seq_len).
    ///
    /// When patch pruning is configured (ViT models compiled with
    /// `PatchPruneInjection`), automatically preprocesses the pixel input
    /// to produce `kept_indices` before feeding the compiled graph.
    pub fn execute(
        &self,
        inputs: &hologram::GraphInputs,
    ) -> anyhow::Result<hologram::GraphOutputs> {
        // If patch pruning is configured, preprocess the pixel input.
        let inputs = if let Some(ref prune) = self.patch_prune {
            self.preprocess_patch_prune(inputs, prune)?
        } else {
            std::borrow::Cow::Borrowed(inputs)
        };

        if let Some(ref ctx) = self.shape_ctx {
            let shape_map = self.resolve_shapes(ctx, &self.plan, &inputs);
            hologram::execute_tape_with_shapes(&self.tape, &self.plan, &inputs, &shape_map)
                .map_err(|e| anyhow::anyhow!("{e}"))
        } else {
            hologram::execute_tape(&self.tape, &self.plan, &inputs)
                .map_err(|e| anyhow::anyhow!("{e}"))
        }
    }

    /// Run the PatchPrune kernel on the pixel input and inject `kept_indices`.
    fn preprocess_patch_prune(
        &self,
        inputs: &hologram::GraphInputs,
        prune: &hologram_ai_common::PatchPruneContext,
    ) -> anyhow::Result<std::borrow::Cow<'_, hologram::GraphInputs>> {
        let pixel_bytes = inputs.get(prune.pixel_input).ok_or_else(|| {
            anyhow::anyhow!(
                "PatchPrune: pixel input at index {} not found",
                prune.pixel_input
            )
        })?;

        // Interpret pixel bytes as f32.
        let pixels: &[f32] = bytemuck::cast_slice(pixel_bytes);

        // Infer image dimensions from pixel count and channel count.
        let channels = prune.channels as usize;
        let total_pixels = pixels.len();
        let spatial_pixels = total_pixels / channels;
        // Assume square image if no shape info available.
        let img_side = (spatial_pixels as f64).sqrt() as usize;
        let (img_h, img_w) = if let Some(shape) = inputs.shape(prune.pixel_input) {
            // Shape is [N, C, H, W] or [C, H, W].
            if shape.len() == 4 {
                (shape[2], shape[3])
            } else if shape.len() == 3 {
                (shape[1], shape[2])
            } else {
                (img_side, img_side)
            }
        } else {
            (img_side, img_side)
        };

        let params = hologram::hologram_exec::PatchPruneParams {
            channels,
            img_h,
            img_w,
            patch_h: prune.patch_h as usize,
            patch_w: prune.patch_w as usize,
            tau: 0.0, // lossless by default
            max_kept: prune.max_kept as usize,
        };

        let result = hologram::hologram_exec::patch_prune(pixels, &params);

        // Build new inputs with kept_indices injected.
        let mut new_inputs = inputs.clone();
        let indices_bytes =
            hologram::hologram_exec::patch_prune::indices_to_bytes(&result.kept_indices);
        new_inputs.set_with_shape(
            prune.kept_indices_input,
            indices_bytes,
            vec![prune.max_kept as usize],
        );

        tracing::debug!(
            n_kept = result.n_kept,
            max_kept = prune.max_kept,
            "PatchPrune preprocessor: selected {}/{} patches",
            result.n_kept,
            prune.total_patches,
        );

        Ok(std::borrow::Cow::Owned(new_inputs))
    }

    /// Access the underlying loaded plan (for layer headers, weights, etc.).
    #[must_use]
    pub fn plan(&self) -> &hologram::LoadedPlan {
        &self.plan
    }

    /// Archive bytes (the full pipeline archive).
    #[must_use]
    pub fn archive_bytes(&self) -> &[u8] {
        self._storage.as_ref()
    }

    /// Raw top-level archive bytes (same as archive_bytes for unified format).
    /// For single-graph archives, returns the effective bytes.
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        self._storage.as_ref()
    }

    /// Whether this archive has a `ShapeContextGraph` for variable seq_len support.
    #[must_use]
    pub fn has_shape_context(&self) -> bool {
        self.shape_ctx.is_some()
    }

    /// Project runtime input shapes through the `ShapeContextGraph` to produce
    /// per-node shape overrides for the executor.
    fn resolve_shapes(
        &self,
        ctx: &ShapeContextGraph,
        plan: &hologram::LoadedPlan,
        inputs: &hologram::GraphInputs,
    ) -> std::collections::HashMap<u32, Vec<usize>> {
        let mut runtime_inputs = std::collections::HashMap::new();
        let sg = plan.graph();

        // Map graph input names to their node indices and inject runtime shapes.
        for (slot, name) in sg.input_names.iter().enumerate() {
            if let Some(shape) = inputs.shape(slot as u32) {
                for node in &sg.nodes {
                    if matches!(node.op, hologram::hologram_graph::graph::GraphOp::Input)
                        && node.id.index() == slot as u32
                    {
                        runtime_inputs.insert(node.id.index(), shape.to_vec());
                        break;
                    }
                }
                runtime_inputs
                    .entry(slot as u32)
                    .or_insert_with(|| shape.to_vec());
            }
            let _ = name;
        }

        let mut shape_map = std::collections::HashMap::new();
        hologram_ai_common::walk_shape_context(
            ctx,
            &runtime_inputs,
            &std::collections::HashMap::new(),
            &mut shape_map,
        );
        shape_map
    }

    /// Execute with a mutable KV cache state for autoregressive generation.
    ///
    /// For LLM pipeline archives: uses the prefill tape for step 0 (write_pos == 0)
    /// and the decode tape for subsequent steps. The decode graph is compiled at
    /// seq=1, making each decode step ~Nx faster than running the full prefill graph.
    ///
    /// For single-graph archives: uses the same tape for all steps.
    ///
    /// Single execution path for all steps (prefill + decode).
    ///
    /// Always uses `execute_tape_with_kv_shapes_cached` with shape overrides.
    /// - **Prefill**: resolves shapes at runtime via `walk_shape_context` for
    ///   variable-length prompt support.
    /// - **Decode**: uses pre-computed `decode_shape_map` (constant at seq=1,
    ///   no walk needed). Provides input_metas that LUT-GEMM kernels require.
    pub fn execute_with_kv(
        &self,
        inputs: &hologram::GraphInputs,
        kv_state: &mut hologram::KvCacheState,
    ) -> anyhow::Result<hologram::GraphOutputs> {
        let is_decode = kv_state.write_pos() > 0;
        let (tape, plan) = if is_decode {
            if let (Some(ref dt), Some(ref dp)) = (&self.decode_tape, &self.decode_plan) {
                (dt, dp)
            } else {
                (&self.tape, &self.plan)
            }
        } else {
            (&self.tape, &self.plan)
        };

        // Prefill: walk shape context for variable-length support.
        // Decode: use pre-computed shape map (no walk overhead).
        let shape_map = if !is_decode {
            if let Some(ref ctx) = self.shape_ctx {
                self.resolve_shapes(ctx, plan, inputs)
            } else {
                std::collections::HashMap::new()
            }
        } else {
            self.decode_shape_map.clone()
        };

        hologram::execute_tape_with_kv_shapes_cached(
            tape,
            plan,
            inputs,
            kv_state,
            &shape_map,
            &self.weight_cache,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Whether this runner has a separate decode model for fast autoregressive generation.
    #[must_use]
    pub fn has_decode_model(&self) -> bool {
        self.decode_tape.is_some()
    }

    /// Whether this runner has a verification model for batch speculative decoding.
    #[must_use]
    pub fn has_verify_model(&self) -> bool {
        self.verify_tape.is_some()
    }

    /// Whether this runner has patch pruning configured (ViT models).
    #[must_use]
    pub fn has_patch_prune(&self) -> bool {
        self.patch_prune.is_some()
    }

    /// Access the patch pruning config (for diagnostics/testing).
    #[must_use]
    pub fn patch_prune_config(&self) -> Option<&hologram_ai_common::PatchPruneContext> {
        self.patch_prune.as_ref()
    }

    /// Execute a batch verification forward pass using the verify tape (seq=N).
    ///
    /// Used by speculative decoding: draft N tokens with decode tape (seq=1),
    /// then verify all N in one forward pass through the verify tape (seq=N).
    /// BLAS amortizes the weight read across N output tokens → N× throughput.
    pub fn execute_verify(
        &self,
        inputs: &hologram::GraphInputs,
        kv_state: &mut hologram::KvCacheState,
    ) -> anyhow::Result<hologram::GraphOutputs> {
        let (tape, plan) =
            if let (Some(ref vt), Some(ref vp)) = (&self.verify_tape, &self.verify_plan) {
                (vt, vp)
            } else {
                // No verify tape — fall back to prefill tape.
                (&self.tape, &self.plan)
            };
        hologram::execute_tape_with_kv_cached(tape, plan, inputs, kv_state, &self.weight_cache)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Extract a named sub-archive's raw bytes from a pipeline archive.
///
/// Uses `LoadedPipeline` to parse the pipeline header, then extracts the
/// Read the [`ShapeContextGraph`] from an already-loaded plan + raw archive bytes.
///
/// Avoids re-deserializing the archive — uses the plan's section table to
/// find the shape context section, then reads it from the raw bytes.
/// Read a [`PatchPruneContext`] from an already-loaded plan + raw archive bytes.
///
/// Returns `None` if the archive has no patch pruning section (non-ViT models).
fn read_patch_prune_from_plan(
    plan: &hologram::LoadedPlan,
    archive_bytes: &[u8],
) -> Option<hologram_ai_common::PatchPruneContext> {
    use hologram_ai_common::exec_context::{ExecContext, SECTION_PATCH_PRUNE};
    let entry = plan.sections().find(SECTION_PATCH_PRUNE)?;
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    if end > archive_bytes.len() {
        tracing::warn!("PatchPruneContext section out of bounds, ignoring");
        return None;
    }
    match hologram_ai_common::PatchPruneContext::from_context_bytes(&archive_bytes[start..end]) {
        Ok(ctx) => {
            tracing::info!(
                max_kept = ctx.max_kept,
                total_patches = ctx.total_patches,
                "loaded PatchPruneContext from archive"
            );
            Some(ctx)
        }
        Err(e) => {
            tracing::warn!("failed to deserialize PatchPruneContext: {e}");
            None
        }
    }
}

fn read_shape_context_from_plan(
    plan: &hologram::LoadedPlan,
    archive_bytes: &[u8],
) -> anyhow::Result<Option<ShapeContextGraph>> {
    use hologram_ai_common::exec_context::{ExecContext, SECTION_SHAPE_CONTEXT};
    let entry = match plan.sections().find(SECTION_SHAPE_CONTEXT) {
        Some(e) => e,
        None => return Ok(None),
    };
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    if end > archive_bytes.len() {
        anyhow::bail!(
            "ShapeContextGraph section out of bounds: offset={} size={} archive_len={}",
            start,
            entry.size,
            archive_bytes.len()
        );
    }
    let ctx = ShapeContextGraph::from_context_bytes(&archive_bytes[start..end])?;
    Ok(Some(ctx))
}

/// Read the [`ShapeContextGraph`] embedded in a compiled `.holo` archive.
///
/// Returns `None` if the archive was compiled without a shape context section
/// (older archives or models compiled with shape context disabled).
pub fn read_shape_context_from_archive(
    archive_bytes: &[u8],
) -> anyhow::Result<Option<ShapeContextGraph>> {
    // SAFETY: plan is dropped at the end of this function; archive_bytes outlives it.
    let plan = unsafe { hologram::load_from_bytes_zero_copy(archive_bytes) }?;
    read_shape_context_from_plan(&plan, archive_bytes)
}

/// Execute a compiled archive with variable-length input support.
///
/// Builds a one-shot tape and executes via the EnumTape path.
/// Dynamic sizes are resolved at execution time via `resolve_size()`
/// and `infer_matmul_k()` in the tape executor.
///
/// If the archive was compiled from a model with attention layers
/// (`n_layers > 0`), a fresh `KvCacheState` is initialised automatically
/// so that KvWrite/KvRead ops succeed during the forward pass.
///
/// For repeated execution, prefer [`HoloRunner`] which builds the tape once.
pub fn run_with_shape_context(
    archive: &HoloArchive,
    inputs: &hologram::GraphInputs,
) -> anyhow::Result<hologram::GraphOutputs> {
    let runner = HoloRunner::from_bytes(archive.bytes.clone())?;
    let m = &archive.metadata;

    if m.n_layers > 0 {
        let mut kv = hologram::KvCacheState::new(
            m.n_layers,
            m.n_kv_heads,
            m.head_dim,
            m.context_len as usize,
        );
        runner.execute_with_kv(inputs, &mut kv)
    } else {
        runner.execute(inputs)
    }
}

pub fn rebuild_archive_with_section(
    archive: &[u8],
    section: &dyn hologram::hologram_archive::section::EmbeddableSection,
) -> anyhow::Result<Vec<u8>> {
    let unpacked = crate::compiler::unpack_archive(archive)?;

    // Filter out the section kind we're replacing.
    let new_kind = section.section_kind();
    let mut writer = hologram::HoloWriter::new()
        .set_graph_bytes_uncompressed(unpacked.graph_bytes)
        .set_weights(unpacked.weight_bytes);

    for (kind, bytes) in unpacked.sections {
        if kind == new_kind {
            continue;
        }
        writer = writer.add_raw_section(kind, bytes);
    }

    writer = writer.add_section(section);

    writer
        .build()
        .map_err(|e| anyhow::anyhow!("rebuilding archive with section: {e}"))
}
