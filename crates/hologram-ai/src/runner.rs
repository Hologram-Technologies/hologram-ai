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
    /// Resident-KV carry (ADR-0019 increment 3b): the κ-labels of the previous
    /// resident decode walk's updated caches, keyed by the INPUT port that
    /// consumes them next walk (`past_k_l`/`past_v_l`). Leased between walks
    /// (residency by ownership — an interleaved walk on this session cannot age
    /// them out); the lease is released at bind time so the κ120 ring write
    /// regains sole ownership and MOVES in place. Empty = no carry.
    kv_carry: std::collections::HashMap<String, ContentLabel>,
}

/// Process-global walk serialization — the substrate's implicit contract made
/// explicit and load-bearing at our boundary.
///
/// hologram's pooled kernels carry publisher/worker scratch that assumes **one
/// walk at a time per process**: two `InferenceSession`s executing concurrently
/// (e.g. parallel test threads, a parallel server) re-enter the v0.9.0 pooled
/// decode-attention scratch — a `RefCell already borrowed` panic at best,
/// silent cross-session corruption of the numbers at worst (observed: decode
/// tokens change run-to-run under concurrent sessions; see
/// `docs/notes/upstream-issue-v090-pooled-decode-scratch.md`). Production
/// drives every session sequentially already (one browser worker, one CLI
/// generation loop), so this lock is uncontended there — nanoseconds against
/// multi-millisecond walks — and it turns any future concurrent caller into a
/// correct, serialized one instead of a corrupted one.
fn walk_lock() -> std::sync::MutexGuard<'static, ()> {
    static WALK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // Poison-tolerant: a panic mid-walk in another thread must not turn every
    // later walk into a panic (consistent with the paged-store lock below).
    WALK.lock().unwrap_or_else(|e| e.into_inner())
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
        Ok(Self {
            session,
            kv_carry: Default::default(),
        })
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
        Ok(Self {
            session,
            kv_carry: Default::default(),
        })
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
        let _walk = walk_lock();
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
        let _walk = walk_lock();
        self.session
            .execute_addressed(input_labels)
            .map_err(|e| anyhow::anyhow!("addressed execute failed: {e:?}"))
    }

    /// Resolve an output κ-label back to its bytes — the address→byte boundary
    /// for reading a result produced by [`Self::execute_addressed`].
    pub fn resolve(&self, label: &ContentLabel) -> Option<&[u8]> {
        self.session.resolve(label)
    }

    /// One fused-decode walk with the carried K/V **resident in the runner**
    /// (ADR-0019 increment 3b). KV input ports (`past_k_*`/`past_v_*`) bind the
    /// κ-labels retained from this runner's previous resident walk — **no byte
    /// re-hash, no copy**, and the κ120 ring write realizes as an in-place
    /// move — while KV output ports (`k_new_*`/`v_new_*`) stay unmaterialized
    /// (retained + leased for the next walk) and return `None`. Non-KV outputs
    /// (logits, activations) are materialized as usual.
    ///
    /// `carry` declares the runner-carried KV **current**. Pass `false` on the
    /// first walk and whenever the host's KV bytes are the truth (after a
    /// seeder prefill, a speculative commit from another runner, or a bucket
    /// regrow): the passed KV bytes are then ingested once (one hash) and a
    /// fresh carry starts. Every input buffer must be supplied either way —
    /// carried KV positions are simply not read when `carry` is true. If a
    /// walk fails, the carry must be treated as broken: re-enter with
    /// `carry = false`.
    pub fn execute_kv_resident(
        &mut self,
        inputs: &[&[u8]],
        carry: bool,
    ) -> anyhow::Result<Vec<Option<OutputBuffer>>> {
        let in_ports = self.input_port_info();
        ensure!(
            inputs.len() == in_ports.len(),
            "resident walk got {} inputs for {} ports",
            inputs.len(),
            in_ports.len()
        );
        // Bind: carried KV by retained label (released at bind so the ring
        // write regains sole ownership and moves), everything else interned.
        let mut labels = Vec::with_capacity(inputs.len());
        let mut consumed: Vec<String> = Vec::new();
        for (i, port) in in_ports.iter().enumerate() {
            let is_kv = port.name.starts_with("past_k_") || port.name.starts_with("past_v_");
            if is_kv && carry {
                let label = *self.kv_carry.get(&port.name).with_context(|| {
                    format!(
                        "carry declared current but no retained cache label for `{}` — \
                         re-enter with carry = false",
                        port.name
                    )
                })?;
                self.session.release_label(&label);
                consumed.push(port.name.clone());
                labels.push(label);
            } else {
                labels.push(self.session.intern_input(inputs[i]));
            }
        }
        // Any carry entries NOT consumed this walk (a fresh ingest after a
        // seeder/commit/regrow) still hold a lease — release them, the host
        // bytes are the truth now.
        if !carry {
            for (_, stale) in self.kv_carry.drain() {
                self.session.release_label(&stale);
            }
        }

        let out_labels = {
            let _walk = walk_lock();
            self.session
                .execute_addressed(&labels)
                .map_err(|e| anyhow::anyhow!("resident decode walk failed: {e:?}"))?
        };

        // Outputs: retain + lease the updated caches under their NEXT-walk
        // input port names; materialize everything else.
        let out_ports = self.output_port_info();
        ensure!(
            out_labels.len() == out_ports.len(),
            "resident walk returned {} outputs for {} ports",
            out_labels.len(),
            out_ports.len()
        );
        let mut new_carry = std::collections::HashMap::with_capacity(4);
        let mut result = Vec::with_capacity(out_labels.len());
        for (label, port) in out_labels.iter().zip(&out_ports) {
            let next_in = port
                .name
                .strip_prefix("k_new_")
                .map(|l| format!("past_k_{l}"))
                .or_else(|| {
                    port.name
                        .strip_prefix("v_new_")
                        .map(|l| format!("past_v_{l}"))
                });
            match next_in {
                Some(name) => {
                    ensure!(
                        self.session.retain_label(label),
                        "updated cache `{}` is not resident — cannot carry",
                        port.name
                    );
                    new_carry.insert(name, *label);
                    result.push(None);
                }
                None => {
                    let bytes = self
                        .session
                        .resolve(label)
                        .with_context(|| format!("output `{}` did not resolve", port.name))?
                        .to_vec();
                    result.push(Some(OutputBuffer { bytes }));
                }
            }
        }
        self.kv_carry = new_carry;
        Ok(result)
    }

    /// Materialize the runner-resident K/V carry back to host bytes, keyed by
    /// the input port that would consume each cache (`past_k_l`/`past_v_l`).
    /// The boundary crossing for everything that needs the bytes host-side —
    /// a speculative verify on another runner, a bucket regrow — after which
    /// the caller owns the truth and must re-enter with `carry = false`.
    /// Leases are released; the carry is cleared.
    pub fn take_kv_carry(&mut self) -> anyhow::Result<Vec<(String, Vec<u8>)>> {
        let mut out = Vec::with_capacity(self.kv_carry.len());
        for (name, label) in std::mem::take(&mut self.kv_carry) {
            let bytes = self
                .session
                .resolve(&label)
                .with_context(|| format!("carried cache `{name}` did not resolve"))?
                .to_vec();
            self.session.release_label(&label);
            out.push((name, bytes));
        }
        Ok(out)
    }

    // ── κ-leases: residency by ownership, not recency (v0.9.0) ────────────────
    //
    // The transient pool keeps a value resident for a two-walk window (recency);
    // a lease keeps it resident until released (ownership). The ownership law:
    // a lease is a borrow, so a `KvCacheWrite` on a leased cache DECLINES the
    // in-place move and takes the honest copy — the leased pre-image survives
    // bit-intact. That is the substrate primitive for holding state beyond one
    // step's outputs: speculative rollback (lease the pre-state; accept ⇒
    // release and the next step moves; reject ⇒ re-step from the intact
    // pre-image) and a paired draft model's KV parked across the main's walks.

    /// Take host ownership of a resident value by κ-label so it survives every
    /// walk until [`Self::release_label`]d (refcounted). Returns `false` if the
    /// label is not resident (or is a lazily-paged weight the pager owns).
    pub fn retain_label(&mut self, label: &ContentLabel) -> bool {
        self.session.retain_label(label)
    }

    /// Release one lease taken by [`Self::retain_label`]. When the last lease
    /// drops, the value is uniquely owned again and the `KvCacheWrite` move
    /// resumes. Returns `false` if no lease is held.
    pub fn release_label(&mut self, label: &ContentLabel) -> bool {
        self.session.release_label(label)
    }

    /// Distinct host-leased values currently held ([`Self::retain_label`]).
    pub fn leased_count(&self) -> usize {
        self.session.leased_count()
    }

    /// Total allocated pool bytes — every live buffer including recycled
    /// free-list capacity. The **confinement** metric: a steady-state decode
    /// loop must hold this constant (O(1) memory per step), which is exactly the
    /// residency the 32-bit host ledger must not over-commit. Read this rather
    /// than estimating the resident footprint host-side.
    pub fn pool_allocated_bytes(&self) -> usize {
        self.session.pool_allocated_bytes()
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
