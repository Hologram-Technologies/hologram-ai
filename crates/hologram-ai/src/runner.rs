//! Inference runner — a thin wrapper over hologram's `InferenceSession`.
//!
//! In the UOR-native model a `.holo` archive is loaded once into an
//! `InferenceSession`, which owns the content-addressed buffer pool and elides
//! repeated computation by κ-label (architecture §5.3, §7). There is no tape
//! builder, no KV-cache, and no runtime shape projection: the compiled archive
//! already carries concrete shapes and a schedule. Autoregressive reuse across
//! decode steps is structural (content-addressed elision), so each step simply
//! re-executes the graph with the next input.

use anyhow::Context;
use hologram_archive::ContentLabel;
use hologram_backend::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer, OutputBuffer};
use std::path::Path;

/// A loaded model ready for inference.
pub struct HoloRunner {
    /// The archive bytes (kept so callers can re-address / inspect the model).
    archive: Vec<u8>,
    /// The execution session (owns its decoded plan + buffer pool).
    session: InferenceSession<CpuBackend<BufferArena>>,
}

impl HoloRunner {
    /// Load a runner from in-memory `.holo` archive bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        let backend = CpuBackend::new();
        let session = InferenceSession::load(&bytes, backend)
            .map_err(|e| anyhow::anyhow!("loading .holo archive: {e:?}"))?;
        Ok(Self {
            archive: bytes,
            session,
        })
    }

    /// Load a runner from a `.holo` file. (`_config` is accepted for CLI
    /// compatibility; the UOR-native runtime needs no host config.)
    pub fn from_path(path: &Path, _config: Option<&Path>) -> anyhow::Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading .holo archive {path:?}"))?;
        Self::from_bytes(bytes)
    }

    /// Number of graph inputs the model expects.
    pub fn input_count(&self) -> usize {
        self.session.input_count()
    }

    /// Number of graph outputs the model produces.
    pub fn output_count(&self) -> usize {
        self.session.output_count()
    }

    /// The raw archive bytes.
    pub fn archive_bytes(&self) -> &[u8] {
        &self.archive
    }

    /// Byte size of each graph input (element count × dtype width), in
    /// graph-input order. Lets callers allocate correctly-sized input buffers.
    pub fn input_byte_sizes(&self) -> Vec<usize> {
        self.session
            .input_ports()
            .iter()
            .map(|p| p.element_count as usize * dtype_byte_width(p.dtype))
            .collect()
    }

    /// Execute one forward pass. `inputs[i]` is the little-endian byte image of
    /// graph input `i`. Returns the output buffers in graph-output order.
    ///
    /// This is the byte-level boundary: inputs are addressed (hashed once) on
    /// entry and outputs are materialized to bytes on exit. To compose calls
    /// without that round-trip, use the κ-label surface below.
    pub fn execute(&mut self, inputs: &[&[u8]]) -> anyhow::Result<Vec<OutputBuffer>> {
        let bufs: Vec<InputBuffer> = inputs.iter().map(|&bytes| InputBuffer { bytes }).collect();
        self.session
            .execute(&bufs)
            .map_err(|e| anyhow::anyhow!("inference execute failed: {e:?}"))
    }

    // ── Content-addressed execution ──────────────────────────────────────────
    //
    // hologram executes over uor-addr κ-labels, not raw values: a value flows
    // by its 71-byte content address and is never rehashed or copied once
    // addressed. The methods below expose that surface so a pipeline composes
    // *on addresses* — feed one call's output labels straight into the next.
    // Because a node's output κ-label is a function of its op + operand labels,
    // an unchanged sub-graph (e.g. the decode prefix) is recognized by label
    // and elided rather than recomputed — the content-addressed reuse that
    // replaces the legacy KV-cache (architecture §5.3, class CE).

    /// Intern raw input bytes into a content address (κ-label). The bytes are
    /// hashed **once**, here at the byte→address boundary; thereafter the value
    /// is referred to by its label. Feed the label to [`Self::execute_addressed`].
    pub fn intern_input(&mut self, bytes: &[u8]) -> ContentLabel {
        self.session.intern_input(bytes)
    }

    /// Execute on content addresses: `input_labels` and the returned labels are
    /// κ-labels, so an already-addressed value (a prior call's output, an
    /// interned prompt) flows with **no byte copy and nothing rehashed**. On a
    /// whole-graph memo hit the cached output labels return immediately.
    pub fn execute_addressed(
        &mut self,
        input_labels: &[ContentLabel],
    ) -> anyhow::Result<Vec<ContentLabel>> {
        self.session
            .execute_addressed(input_labels)
            .map_err(|e| anyhow::anyhow!("addressed execute failed: {e:?}"))
    }

    /// Resolve an output κ-label back to its bytes — the address→byte boundary
    /// for reading a result produced by [`Self::execute_addressed`].
    pub fn resolve(&self, label: &ContentLabel) -> Option<&[u8]> {
        self.session.resolve(label)
    }

    /// Resident bytes in the content-addressed pool, **deduplicated by κ-label**
    /// — the runtime memory footprint of all interned values (weights supplied
    /// as inputs, intermediate results). Values that share a content address
    /// occupy one buffer, so this is the size of the *distinct* set. Lets a
    /// caller measure how much space weights actually require at runtime under
    /// canonicalization.
    pub fn resident_bytes(&self) -> usize {
        self.session.resident_bytes()
    }

    /// Number of distinct resident values in the pool (deduped by κ-label).
    pub fn resident_count(&self) -> usize {
        self.session.resident_count()
    }
}

/// Byte width of a canonical dtype tag (`hologram_backend::cpu::dtype` encoding).
fn dtype_byte_width(tag: u8) -> usize {
    match tag {
        0 | 1 | 2 | 10 => 1, // Bool, U8, I8, I4 (packed → byte-addressed)
        6 | 7 => 2,          // F16, BF16
        4 => 4,              // I32
        8 => 4,              // F32
        3 | 5 | 9 => 8,      // U64, I64, F64
        _ => 4,
    }
}
