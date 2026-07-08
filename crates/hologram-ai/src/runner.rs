//! Inference runner — a thin wrapper over hologram's `InferenceSession`.
//!
//! In the UOR-native model a `.holo` archive is loaded once into an
//! `InferenceSession`, which owns the content-addressed buffer pool and elides
//! repeated computation by κ-label (architecture §5.3, §7). There is no tape
//! builder, no KV-cache, and no runtime shape projection: the compiled archive
//! already carries concrete shapes and a schedule. Autoregressive reuse across
//! decode steps is structural (content-addressed elision), so each step simply
//! re-executes the graph with the next input.

use anyhow::{ensure, Context, Result};
use hologram_archive::{ContentLabel, WeightFingerprint, WeightProvider};
use hologram_backend::CpuBackend;
use hologram_exec::{BufferArena, InferenceSession, InputBuffer, OutputBuffer};
use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::materialize::{kappa_of, WeightBindingTable};

/// Shape/dtype facts about one graph port: its semantic name (e.g.
/// `"input_ids"`; empty if unnamed), the backend dtype tag
/// (`hologram_backend::cpu::dtype` encoding), the logical element count, and the
/// full row-major shape (empty if the rank wasn't registered).
#[derive(Debug, Clone)]
pub struct PortInfo {
    /// Semantic port name, or empty string if the port is unnamed.
    pub name: String,
    /// Backend dtype tag (e.g. `5` = I64, `8` = F32; see `port_byte_size`).
    pub dtype: u8,
    /// Logical element count (product of the port's concrete dims).
    pub element_count: usize,
    /// Full row-major shape; empty when the rank wasn't registered.
    pub shape: Vec<usize>,
}

/// A loaded model ready for inference.
pub struct HoloRunner {
    /// The execution session (owns its decoded plan + buffer pool).
    session: InferenceSession<CpuBackend<BufferArena>>,
}

impl HoloRunner {
    /// Load a runner from in-memory `.holo` archive bytes.
    ///
    /// The session decodes its plan into owned storage, so the archive bytes are
    /// dropped here rather than retained — for a multi-hundred-MB model that
    /// halves resident memory (the session interns weights into its own pool; a
    /// second copy of the archive would just sit idle), which is what lets the
    /// length-adaptive engine hold the prepared model and a live window at once.
    pub fn from_bytes(bytes: Vec<u8>) -> anyhow::Result<Self> {
        let backend = CpuBackend::new();
        let session = InferenceSession::load(&bytes, backend)
            .map_err(|e| anyhow::anyhow!("loading .holo archive: {e:?}"))?;
        drop(bytes);
        Ok(Self { session })
    }

    /// Load a runner from a `.holo` file. (`_config` is accepted for CLI
    /// compatibility; the UOR-native runtime needs no host config.)
    pub fn from_path(path: &Path, _config: Option<&Path>) -> anyhow::Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading .holo archive {path:?}"))?;
        Self::from_bytes(bytes)
    }

    /// Load a **paged** runner (row `lazy-constant-residency`): the archive's
    /// whole-κ weight constants are `by_reference` fingerprints the session
    /// pages from `provider` on first use, holding their bytes resident only
    /// within `budget` (LRU-evicted, `budget == 0` = unbounded). The arena is a
    /// bounded **window** over the provider rather than a full copy of the
    /// weight set — the one structural change that lets a model whose weights
    /// exceed the window run at all. Build the paged archive + provider with
    /// [`Self::from_kform_paged`], or supply your own.
    pub fn from_paged(
        bytes: Vec<u8>,
        provider: Arc<dyn WeightProvider + Send + Sync>,
        budget: usize,
    ) -> anyhow::Result<Self> {
        let backend = CpuBackend::new();
        let session = InferenceSession::load_paged(&bytes, backend, provider, budget)
            .map_err(|e| anyhow::anyhow!("loading paged .holo archive: {e:?}"))?;
        drop(bytes);
        Ok(Self { session })
    }

    /// Build a paged runner directly from a k-form archive and its κ-store: turn
    /// the whole-κ constants into paged references ([`crate::materialize::paged_archive`]),
    /// wrap the store as a [`KappaWeightProvider`], and load against `budget`.
    /// The store backs the provider (page-in with verify-once and
    /// invalidate-and-recover); ranged (sub-tensor) bindings are materialized
    /// inline at build (verified once), the dominant whole-tensor weights page
    /// on demand.
    pub fn from_kform_paged<S>(kform: &[u8], mut store: S, budget: usize) -> anyhow::Result<Self>
    where
        S: crate::materialize::KappaStore + Send + 'static,
    {
        let (paged, table) = crate::materialize::paged_archive(kform, &mut store)?;
        let provider = Arc::new(KappaWeightProvider::new(table, Box::new(store)));
        Self::from_paged(paged, provider, budget)
    }

    /// Resident **paged-weight** bytes — the lazy tier bounded by the residency
    /// budget of a [`Self::from_paged`] load (`0` for a fully-resident load).
    /// Its peak across a decode is the pager's witness (row
    /// `lazy-constant-residency`): under budget while output is unchanged.
    pub fn lazy_resident_bytes(&self) -> usize {
        self.session.paged_weight_bytes()
    }

    /// Number of graph inputs the model expects.
    pub fn input_count(&self) -> usize {
        self.session.input_count()
    }

    /// Number of graph outputs the model produces.
    pub fn output_count(&self) -> usize {
        self.session.output_count()
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

    /// Byte size of each graph output, in graph-output order.
    pub fn output_byte_sizes(&self) -> Vec<usize> {
        self.session
            .output_ports()
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

    /// Kernels dispatched in the most recent compute walk (class **CE** —
    /// content-addressed elision). The contract:
    ///
    /// - A whole-graph memo hit doesn't walk at all — the counter retains its
    ///   previous value (use [`Self::resolve`] + cached output labels to check).
    /// - A walk: every node whose reuse key is already resident is **elided**
    ///   (counted by [`Self::last_skipped`]); the rest are dispatched. So
    ///   `last_dispatched + last_skipped == kernel_count` on a walked call.
    ///
    /// Re-executing on inputs that share a prefix with a prior walk drops this
    /// below [`Self::kernel_count`] — the sub-graph elision that replaces a
    /// mutable KV-cache in autoregressive decode.
    pub fn last_dispatched(&self) -> usize {
        self.session.last_dispatched()
    }

    /// Kernels elided in the most recent walk because their output κ-label was
    /// already resident — the count of reused sub-graph nodes (class **CE**).
    pub fn last_skipped(&self) -> usize {
        self.session.last_skipped()
    }

    /// Total kernels in the loaded schedule (denominator for the elision ratio).
    pub fn kernel_count(&self) -> usize {
        self.session.kernel_count()
    }
}

/// hologram's [`WeightProvider`] backed by a κ-store (row
/// `lazy-constant-residency`): the inversion of the fully-resident load, where
/// the weight bodies live in the host's κ-store (a directory, OPFS) and the
/// session pages ranges from here instead of copying every body resident.
///
/// A paged constant carries the fingerprint of its whole κ content (built by
/// [`crate::materialize::paged_archive`]); this provider maps that fingerprint
/// back to the κ and serves its bytes. Verification is placed at the
/// trust-boundary crossing exactly once per κ per session (row
/// `session-verified-kappa`): the first page-in of a κ resolves its whole
/// content and checks it re-hashes to the κ, and every later page-in (after an
/// eviction) is read-only I/O — corrupted content fails loud, never executes.
pub struct KappaWeightProvider {
    table: WeightBindingTable,
    inner: Mutex<PagedStoreState>,
}

/// A κ-store the paged provider backs onto — the SAME
/// [`KappaStore`](crate::materialize::KappaStore) trait the resident path
/// uses, so the provider inherits `invalidate` (the unpin hook) and can
/// recover a corrupted κ exactly as `patch_constants` does, rather than
/// dead-ending the paged load.
pub type PagedStore = Box<dyn crate::materialize::KappaStore + Send>;

struct PagedStoreState {
    store: PagedStore,
    verified: std::collections::HashSet<String>,
}

impl KappaWeightProvider {
    /// Build from a fingerprint→κ [`WeightBindingTable`] and the κ-store the
    /// weights page from. The provider owns verification and residency; the
    /// store owns resolution and its own cache tiers.
    pub fn new(table: WeightBindingTable, store: PagedStore) -> Self {
        Self {
            table,
            inner: Mutex::new(PagedStoreState {
                store,
                verified: std::collections::HashSet::new(),
            }),
        }
    }

    /// Total weight bytes the provider addresses — the full set the pager holds
    /// a bounded window over.
    pub fn total_bytes(&self) -> u64 {
        self.table.total_bytes()
    }
}

/// Resolve `kappa` (expected whole length `size`) with the trust-boundary law:
/// a κ verified this session is read-only I/O; a first-touch κ is verified, and
/// on a hash mismatch it UNPINS (invalidate) and re-resolves once from a deeper
/// tier — corrupted content degrades to a re-stream, never a dead-end, and
/// never executes unverified (row `saturation-residency`).
fn resolve_verified(state: &mut PagedStoreState, kappa: &str, size: u64) -> Result<Vec<u8>> {
    if state.verified.contains(kappa) {
        // Session-verified: move only the bytes (a seekable store seeks).
        return state.store.resolve_range(kappa, 0, size);
    }
    let bytes = state.store.resolve(kappa)?;
    if kappa_of(&bytes) == kappa {
        state.verified.insert(kappa.to_string());
        return Ok(bytes);
    }
    // Unpin the corrupted entry and recover once from the deeper tier.
    state.store.invalidate(kappa);
    let recovered = state.store.resolve(kappa)?;
    ensure!(
        kappa_of(&recovered) == kappa,
        "κ integrity failure for `{kappa}`: paged content does not re-hash to its label"
    );
    state.verified.insert(kappa.to_string());
    Ok(recovered)
}

impl WeightProvider for KappaWeightProvider {
    fn size(&self, fp: WeightFingerprint) -> Option<usize> {
        // A κ whose whole length exceeds this host's address space cannot be a
        // slot; return None so the load fails loud rather than truncating u64→
        // usize into a wrong slot size.
        self.table
            .resolve(&fp.0)
            .and_then(|(_, size)| usize::try_from(size).ok())
    }

    fn get_range(&self, fp: WeightFingerprint, offset: usize, len: usize) -> Option<Cow<'_, [u8]>> {
        let (kappa, size) = self.table.resolve(&fp.0)?;
        let usize_size = usize::try_from(size).ok()?;
        if offset.checked_add(len)? > usize_size {
            return None;
        }
        // Poison-tolerant: a panic in a prior resolver leaves the lock
        // recoverable rather than turning every later page-in into a panic.
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let bytes = resolve_verified(&mut state, kappa, size).ok()?;
        if bytes.len() != usize_size {
            return None; // the store's stat disagreed with the body — fail closed
        }
        Some(Cow::Owned(bytes[offset..offset + len].to_vec()))
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
/// by `port_byte_size`, not here.
fn dtype_byte_width(tag: u8) -> usize {
    match tag {
        0..=2 => 1,     // Bool, U8, I8
        6 | 7 => 2,     // F16, BF16
        4 => 4,         // I32
        8 => 4,         // F32
        3 | 5 | 9 => 8, // U64, I64, F64
        _ => 4,
    }
}

#[cfg(test)]
mod pager_tests {
    use super::*;
    use crate::materialize::{kappa_of, KappaStore, WeightBindingTable};

    /// A κ-store with a shallow cache tier over a deep tier, so corruption in
    /// the cache RECOVERS from the deep tier after an `invalidate` — the
    /// `saturation-residency` law the paged provider must uphold.
    struct TwoTier {
        good: Vec<u8>,
        cache_corrupt: bool,
        resolves: usize,
    }
    impl KappaStore for TwoTier {
        fn resolve(&mut self, _kappa: &str) -> Result<Vec<u8>> {
            self.resolves += 1;
            if self.cache_corrupt {
                Ok(b"corrupt-cache-bytes-wrong-length!".to_vec())
            } else {
                Ok(self.good.clone())
            }
        }
        fn invalidate(&mut self, _kappa: &str) {
            self.cache_corrupt = false; // the deep tier recovers
        }
    }

    fn provider_for(store: PagedStore, body: &[u8]) -> KappaWeightProvider {
        let mut table = WeightBindingTable::default();
        let fp = hologram_archive::WeightFingerprint::of(body);
        table.insert_binding(fp.0, kappa_of(body), body.len() as u64);
        KappaWeightProvider::new(table, store)
    }

    #[test]
    fn provider_recovers_a_corrupt_kappa_by_invalidate_and_re_resolve() {
        let body = b"the-real-weight-body".to_vec();
        let store = TwoTier {
            good: body.clone(),
            cache_corrupt: true,
            resolves: 0,
        };
        let provider = provider_for(Box::new(store), &body);
        let fp = hologram_archive::WeightFingerprint::of(&body);
        let got = provider
            .get_range(fp, 0, body.len())
            .expect("corruption recovers from the deep tier");
        assert_eq!(
            got.as_ref(),
            &body[..],
            "recovered content is the real body"
        );
    }

    #[test]
    fn provider_fails_closed_on_a_missing_kappa() {
        struct Empty;
        impl KappaStore for Empty {
            fn resolve(&mut self, kappa: &str) -> Result<Vec<u8>> {
                anyhow::bail!("κ `{kappa}` not present in store")
            }
        }
        let body = b"weight".to_vec();
        let provider = provider_for(Box::new(Empty), &body);
        let fp = hologram_archive::WeightFingerprint::of(&body);
        assert!(
            provider.get_range(fp, 0, body.len()).is_none(),
            "a missing κ fails closed (None), never a wrong or unverified body"
        );
    }

    #[test]
    fn provider_serves_a_zero_byte_weight() {
        struct Zero;
        impl KappaStore for Zero {
            fn resolve(&mut self, _kappa: &str) -> Result<Vec<u8>> {
                Ok(Vec::new())
            }
        }
        let body: &[u8] = &[];
        let provider = provider_for(Box::new(Zero), body);
        let fp = hologram_archive::WeightFingerprint::of(body);
        let got = provider
            .get_range(fp, 0, 0)
            .expect("a 0-byte weight resolves");
        assert!(got.as_ref().is_empty());
    }

    #[test]
    fn provider_fails_closed_when_the_stat_disagrees_with_the_body() {
        // A store whose content_size (already in the table) disagrees with the
        // resolved body length must fail closed, never serve a mismatched slot.
        struct WrongLen;
        impl KappaStore for WrongLen {
            fn resolve(&mut self, _kappa: &str) -> Result<Vec<u8>> {
                Ok(vec![0u8; 8]) // real body is 8 bytes …
            }
        }
        let mut table = WeightBindingTable::default();
        let fp = hologram_archive::WeightFingerprint::of(&[0u8; 8]);
        table.insert_binding(fp.0, kappa_of(&[0u8; 8]), 16); // … but the stat said 16
        let provider = KappaWeightProvider::new(table, Box::new(WrongLen));
        assert!(
            provider.get_range(fp, 0, 8).is_none(),
            "a stat/body length disagreement fails closed"
        );
    }
}
