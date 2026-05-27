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

/// Shape/dtype facts about one graph port: its semantic name (e.g.
/// `"input_ids"`; empty if unnamed), the backend dtype tag
/// (`hologram_backend::cpu::dtype` encoding), the logical element count, and the
/// full row-major shape (empty if the rank wasn't registered).
#[derive(Debug, Clone)]
pub struct PortInfo {
    /// Semantic port name, or empty string if the port is unnamed.
    pub name: String,
    /// Backend dtype tag (e.g. `5` = I64, `8` = F32; see [`port_byte_size`]).
    pub dtype: u8,
    /// Logical element count (product of the port's concrete dims).
    pub element_count: usize,
    /// Full row-major shape; empty when the rank wasn't registered.
    pub shape: Vec<usize>,
}

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
            .map(|p| port_byte_size(p.element_count as usize, p.dtype))
            .collect()
    }

    /// Per-input [`PortInfo`] (name, dtype, element count, shape), in graph-input
    /// order. Compiled archives now carry port **names**, so a caller can find a
    /// role by name (e.g. `"input_ids"`) via [`Self::input_index_by_name`]
    /// instead of relying on position.
    pub fn input_port_info(&self) -> Vec<PortInfo> {
        self.session.input_ports().iter().map(port_info).collect()
    }

    /// Per-output [`PortInfo`], in graph-output order (e.g. `"logits"`).
    pub fn output_port_info(&self) -> Vec<PortInfo> {
        self.session.output_ports().iter().map(port_info).collect()
    }

    /// Index of the input port named `name` (e.g. `"input_ids"`), or `None`.
    pub fn input_index_by_name(&self, name: &str) -> Option<usize> {
        self.session.input_port_by_name(name).map(|(i, _)| i)
    }

    /// Index of the output port named `name` (e.g. `"logits"`), or `None`.
    pub fn output_index_by_name(&self, name: &str) -> Option<usize> {
        self.session.output_port_by_name(name).map(|(i, _)| i)
    }

    /// Open producer metadata stored in the archive under `key` (an extension
    /// section): tokenizer, generation config, … `None` if absent.
    pub fn extension(&self, key: &str) -> Option<&[u8]> {
        self.session.extension(key)
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

    /// Number of `dequantize → matmul` pairs hologram fused into
    /// `MatMulDequant` — the quantized weight read in-register, with the dense
    /// f32 weight never materialized. Non-zero means a quantized model keeps its
    /// weights packed at runtime (architecture §6, class QZ).
    pub fn dequant_matmul_fused_count(&self) -> usize {
        self.session.dequant_fused_count()
    }
}

/// Build a [`PortInfo`] from an archive [`PortDescriptor`] (name + dtype +
/// element count + shape).
fn port_info(p: &hologram_archive::PortDescriptor) -> PortInfo {
    PortInfo {
        name: p.name.clone(),
        dtype: p.dtype,
        element_count: p.element_count as usize,
        shape: p.shape.iter().map(|&d| d as usize).collect(),
    }
}

/// Byte size of a port holding `element_count` elements of the given dtype
/// tag, honoring sub-byte packing (I4 is two nibbles per byte) — mirrors the
/// backend's `div_ceil(n, 2)` sizing so an i4 input/weight is not over-reported.
fn port_byte_size(element_count: usize, tag: u8) -> usize {
    match tag {
        10 => element_count.div_ceil(2), // I4: 2 nibbles/byte (packed)
        _ => element_count * dtype_byte_width(tag),
    }
}

/// Byte width of a canonical (whole-byte) dtype tag
/// (`hologram_backend::cpu::dtype` encoding). Sub-byte dtypes (I4) are handled
/// by [`port_byte_size`], not here.
fn dtype_byte_width(tag: u8) -> usize {
    match tag {
        0..=2 => 1, // Bool, U8, I8
        6 | 7 => 2, // F16, BF16
        4 => 4,         // I32
        8 => 4,         // F32
        3 | 5 | 9 => 8, // U64, I64, F64
        _ => 4,
    }
}
