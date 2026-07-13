//! Decode-plan execution engine (dictionary rows `decode-plan`,
//! `chunked-prefill`).
//!
//! Drives compiled decode archives — `chunk` positions in, one logit row out
//! — carrying each layer's K/V rows between passes through the plan's named
//! ports. The carried K/V is derived content moving through ports, not a
//! mutable cache inside the graph: the engine owns fixed `bucket`-row
//! buffers, feeds them as `past_k_l`/`past_v_l` inputs, and splices each
//! pass's `k_new_l`/`v_new_l` outputs (head-major `[kv, chunk, head_dim]`,
//! so a per-head splice is one contiguous row-range copy) into rows
//! `pos..pos+real` afterwards. Bucket exhaustion recompiles at a
//! geometrically larger bucket and copies the realized rows — capacity is a
//! recompile, never a ceiling; the model's own trained context is the only
//! semantic bound.
//!
//! Two runners share one session's buffers: the **step** runner (`chunk = 1`,
//! generation) and an optional **seeder** (`chunk = C`, chunked prefill —
//! row `chunked-prefill`): prompt positions feed C at a time, amortizing the
//! weight stream across the chunk. A final partial chunk PADS to C — padded
//! rows land above the realized length, where the decode mask makes them
//! unreachable until real content overwrites them, so padding is sound by
//! the same law that makes a fixed bucket sound.
//!
//! Positions are runtime data: the engine synthesizes the RoPE tables at
//! each pass's absolute positions (standard `theta^(-2i/d)` tables, halves
//! duplicated to match the rotate-half kernel, pre-expanded to the plan's
//! head-major row layout) and the additive `decode_mask` carrying both
//! visibility laws — unrealized bucket rows erased, chunk position `i`
//! seeing only new columns `≤ i`.

use anyhow::{bail, ensure, Context, Result};
use hologram_ai_common::rope::RopeSpec;

use crate::engine::LmSession;

/// Geometry recovered from a decode archive's own port shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeGeometry {
    /// Decoder layers (= count of `past_k_*` input ports).
    pub layers: usize,
    /// KV heads per layer (`past_k_l` dim 0).
    pub kv_heads: usize,
    /// Query heads (`rope_cos_q` rows / chunk).
    pub heads: usize,
    /// Head dim (`past_k_l` dim 2 = rope table width).
    pub head_dim: usize,
    /// Fixed past-bucket row count (`past_k_l` dim 1).
    pub bucket: usize,
    /// Positions processed per pass (`k_new_0` dim 1; 1 = generation step).
    pub chunk: usize,
    /// Vocabulary size (`logits` element count — one gathered row per pass).
    pub vocab: usize,
    /// v0.9.0 fused resident-KV form (ADR-0019): the plan carries a `decode_pos`
    /// operand and a fused `DecodeAttention` + `KvCacheWrite`, so the carried-K/V
    /// outputs are the WHOLE updated caches (bound forward, not spliced) and the
    /// mask is `[chunk, bucket+chunk]` (one row per position, not per head-group).
    /// `false` is the legacy per-group decomposition.
    pub resident_kv: bool,
}

impl DecodeGeometry {
    /// The carried K/V bytes for ONE bucket row, across every layer:
    /// `2 (K and V) · layers · kv_heads · head_dim · 4 B`. Multiply by the bucket
    /// to get the session's whole carried K/V. Checked — a geometry whose K/V
    /// cannot be addressed fails loud naming the shape rather than wrapping.
    ///
    /// This is a MODEL quantity, derived from the model's own attention shape.
    /// Any address-space accounting that omits it is under-counting by a term
    /// that grows with the bucket: at a 32 k bucket a mid-size model's K/V is
    /// gigabytes, dwarfing any fixed reserve.
    pub fn carried_kv_bytes_per_row(&self) -> Result<u64> {
        let per_row = (self.layers as u64)
            .checked_mul(2)
            .and_then(|n| n.checked_mul(self.kv_heads as u64))
            .and_then(|n| n.checked_mul(self.head_dim as u64))
            .and_then(|n| n.checked_mul(4));
        per_row.with_context(|| {
            format!(
                "the carried K/V row (2 · {} layers · {} kv · {} head_dim · 4 B) is not addressable",
                self.layers, self.kv_heads, self.head_dim
            )
        })
    }

    /// The session's whole carried K/V at this geometry: [`Self::carried_kv_bytes_per_row`]
    /// times the bucket. Checked.
    pub fn carried_kv_bytes(&self) -> Result<u64> {
        let per_row = self.carried_kv_bytes_per_row()?;
        per_row.checked_mul(self.bucket as u64).with_context(|| {
            format!(
                "the carried K/V ({per_row} B/row · {} bucket) is not addressable",
                self.bucket
            )
        })
    }

    /// Read the geometry out of a decode plan's port shapes — the ports are
    /// the contract, so no side-channel metadata is trusted over them. Works
    /// over any [`LmSession`]: a monolithic decode archive and the staged
    /// decode pipeline expose the same ports.
    pub fn discover(session: &impl LmSession) -> Result<Self> {
        let inputs = session.input_port_info();
        let layers = inputs
            .iter()
            .filter(|p| p.name.starts_with("past_k_"))
            .count();
        ensure!(
            layers > 0,
            "archive has no past_k_* input ports — not a decode plan"
        );
        // The v0.9.0 fused form (ADR-0019) is recognized by its `decode_pos`
        // operand; it shapes the carried-K/V ports rank-4 `[1, kv, bucket, dh]`
        // and its mask one row per position. The legacy decomposition has no
        // `decode_pos`, rank-3 past ports, and a `[g·chunk, ...]` mask.
        let resident_kv = inputs.iter().any(|p| p.name == "decode_pos");

        let pk = inputs
            .iter()
            .find(|p| p.name == "past_k_0")
            .context("decode plan lacks past_k_0")?;
        let (kv_heads, bucket, head_dim) = if resident_kv {
            ensure!(
                pk.shape.len() == 4 && pk.shape[0] == 1,
                "resident-KV past_k_0 must be [1, kv, bucket, head_dim], got {:?}",
                pk.shape
            );
            (pk.shape[1], pk.shape[2], pk.shape[3])
        } else {
            ensure!(
                pk.shape.len() == 3,
                "past_k_0 must be [kv, bucket, head_dim], got {:?}",
                pk.shape
            );
            (pk.shape[0], pk.shape[1], pk.shape[2])
        };

        let outputs = session.output_port_info();
        // Chunk (positions per pass) is read from the mask's row count in the
        // fused form (`[chunk, bucket+chunk]`), whose carried-K/V outputs are the
        // whole cache; legacy reads it from the new-rows output `k_new_0`.
        let chunk = if resident_kv {
            let mask = inputs
                .iter()
                .find(|p| p.name == "decode_mask")
                .context("resident-KV plan lacks decode_mask")?;
            ensure!(
                mask.shape.len() == 2 && mask.shape[1] == bucket + mask.shape[0],
                "resident-KV decode_mask must be [chunk, bucket+chunk], got {:?} (bucket {bucket})",
                mask.shape
            );
            mask.shape[0]
        } else {
            let kn = outputs
                .iter()
                .find(|p| p.name == "k_new_0")
                .context("decode plan lacks a k_new_0 output")?;
            ensure!(
                kn.shape.len() == 3 && kn.shape[0] == kv_heads && kn.shape[2] == head_dim,
                "k_new_0 must be [kv, chunk, head_dim], got {:?}",
                kn.shape
            );
            kn.shape[1]
        };

        let cq = inputs
            .iter()
            .find(|p| p.name == "rope_cos_q")
            .context("decode plan lacks rope_cos_q")?;
        ensure!(
            cq.shape.len() == 2 && chunk > 0 && cq.shape[0].is_multiple_of(chunk),
            "rope_cos_q must be [heads·chunk, head_dim], got {:?} at chunk {chunk}",
            cq.shape
        );
        let heads = cq.shape[0] / chunk;

        // The engine trusts the geometry it reads from the ports, so validate
        // the two structural invariants the RoPE and GQA emission assume,
        // rather than dividing silently: the rotate-half pairing needs an even
        // head dim, and the per-kv-group query slicing needs an integral
        // grouping. A plan that violated either would produce wrong numbers,
        // not an error — so fail loud here.
        ensure!(
            head_dim % 2 == 0,
            "decode plan head_dim {head_dim} is odd — rotate-half RoPE pairs j ± d/2 and needs an even head dim"
        );
        ensure!(
            kv_heads > 0 && heads % kv_heads == 0,
            "decode plan head grouping {heads}/{kv_heads} is not integral"
        );

        let logits = outputs
            .iter()
            .find(|p| p.name == "logits")
            .context("decode plan lacks a logits output")?;
        Ok(Self {
            layers,
            kv_heads,
            heads,
            head_dim,
            bucket,
            chunk,
            vocab: logits.element_count,
            resident_kv,
        })
    }
}

/// Rebuild source for bucket growth: given a bucket size, compile a fresh
/// step plan. `None` pins the session to its initial bucket (exhaustion
/// then fails loud instead of silently truncating context).
pub type DecodeRebuild<S> = Box<dyn FnMut(u64) -> Result<S>>;

/// The carried state a pass reads and writes — split from the runners so a
/// pass can borrow the state and a runner disjointly.
struct DecodeState {
    /// The model's complete rotary law (base, `rope_scaling`, partial
    /// rotary) — the runtime table generator asks it for every pass's rows.
    rope: RopeSpec,
    /// Per-layer past K/V byte buffers, each `kv · bucket · head_dim` f32s.
    past_k: Vec<Vec<u8>>,
    past_v: Vec<Vec<u8>>,
    /// Realized positions (= the next token's absolute position).
    cur_len: usize,
    /// The realized token at each position — the carried K/V's provenance,
    /// so a later sequence can rewind to its common prefix instead of
    /// replaying it (cross-turn K/V retention).
    tokens: Vec<i64>,
    /// Passes executed over the session's lifetime (the retention and
    /// prefill instruments: a shared-prefix turn adds only its suffix; a
    /// chunked prefill adds ceil(suffix/chunk) instead of suffix).
    steps: u64,
    /// The per-pass input buffers, allocated once and refreshed in place —
    /// a generation step re-fills, never re-allocates (see
    /// [`DecodeState::refresh_buffers`]).
    bufs: PassBuffers,
    /// Cached rotary frequency law: `inv_freqs`/`attention_factor` are
    /// position-free for the length-independent laws, so a step reuses them;
    /// a length-dependent law (dynamic/longrope) recomputes when the realized
    /// length moves. `rope_valid_for` is the seq_len the cache was built at.
    rope_freqs: Vec<f64>,
    rope_scale: f32,
    rope_valid_for: Option<usize>,
    /// Staging rows `[chunk · head_dim]` for the rope table assembly.
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    /// The mask buffer's identity `(span, group_rows, chunk, resident)` and
    /// the realized position it encodes: a pass with the same identity flips
    /// only the Δpos columns (O(Δpos · rows), amortized O(1)/token) instead
    /// of rebuilding — and re-hashing — O(bucket) bytes every step.
    mask_key: Option<(usize, usize, usize, bool)>,
    mask_pos: usize,
}

/// The owned per-pass input buffers — everything but the carried K/V, which the
/// binder splices in by reference. Shared by `pass` (the gather-head decode /
/// generation step) and `verify_pass` (the all-positions verify pass): both
/// feed byte-identical decode inputs, differing only in which head the runner
/// carries and how their logits are read.
#[derive(Default)]
struct PassBuffers {
    ids: Vec<u8>,
    cos_q: Vec<u8>,
    sin_q: Vec<u8>,
    cos_k: Vec<u8>,
    sin_k: Vec<u8>,
    mask_b: Vec<u8>,
    lp: [u8; 8],
    /// Ring-write position (`cur_len`) for the fused resident-KV form's
    /// `decode_pos` operand; unbound (and 0) on the legacy path.
    pos: [u8; 4],
}

impl DecodeState {
    /// Refresh the per-pass input buffers IN PLACE for `tokens` at the
    /// current position (`real ≤ geom.chunk`, padded up to the chunk; pad
    /// rows are masked below the realized length until overwritten). The
    /// rope rows carry the spec's complete frequency law at the realized
    /// length; the mask update is incremental when only the position moved.
    fn refresh_buffers(&mut self, geom: DecodeGeometry, tokens: &[i64]) -> Result<()> {
        let real = tokens.len();
        ensure!(
            0 < real && real <= geom.chunk,
            "a pass takes 1..={} tokens, got {real}",
            geom.chunk
        );
        let (chunk, pos) = (geom.chunk, self.cur_len);
        let g = geom.heads / geom.kv_heads;
        let d = geom.head_dim;

        self.bufs.ids.clear();
        for i in 0..chunk {
            let t = tokens.get(i).copied().unwrap_or(0);
            self.bufs.ids.extend_from_slice(&t.to_le_bytes());
        }

        // Rope tables at absolute positions pos..pos+chunk, head-major. The
        // frequency law is cached: position-free for the length-independent
        // laws, recomputed at each new realized length for dynamic/longrope.
        let seq_len = pos + chunk;
        let stale = match self.rope_valid_for {
            None => true,
            Some(at) => self.rope.scaling.length_dependent() && at != seq_len,
        };
        if stale {
            self.rope_freqs = self.rope.inv_freqs(d, seq_len);
            self.rope_scale = self.rope.attention_factor(seq_len);
            self.rope_valid_for = Some(seq_len);
        }
        self.rope_cos.resize(chunk * d, 0.0);
        self.rope_sin.resize(chunk * d, 0.0);
        self.rope.rows_into(
            pos,
            chunk,
            d,
            &self.rope_freqs,
            self.rope_scale,
            &mut self.rope_cos,
            &mut self.rope_sin,
        );
        Self::expand_rows_into(&self.rope_cos, geom.heads, &mut self.bufs.cos_q);
        Self::expand_rows_into(&self.rope_sin, geom.heads, &mut self.bufs.sin_q);
        Self::expand_rows_into(&self.rope_cos, geom.kv_heads, &mut self.bufs.cos_k);
        Self::expand_rows_into(&self.rope_sin, geom.kv_heads, &mut self.bufs.sin_k);

        // The additive mask carries both visibility laws: a bucket col is
        // visible when realized (`col < pos`); a chunk col (the new keys) when
        // causal-within-chunk (`col - bucket ≤ i`). The fused kernel groups
        // internally, so its mask is one row per position `[chunk, span]`; the
        // legacy decomposition needs it replicated per query-head-group
        // (`[g·chunk, span]`, row `jj·chunk + i`). The fused DecodeAttention
        // kernel erases a key with a true `-∞` (its softmax maps `exp(-∞) →
        // 0.0` exactly, per the substrate contract); the legacy
        // decomposition's `Add(scores, mask)` uses a large finite `-1e9` (a
        // raw `-∞` there would make a fully-masked pad row `NaN`). Matching
        // each form's erase value exactly is load-bearing: a finite mask into
        // the fused kernel leaves a nonzero weight that flips boundary tokens.
        let span = geom.bucket + chunk;
        let erase = if geom.resident_kv {
            f32::NEG_INFINITY
        } else {
            -1e9
        };
        let group_rows = if geom.resident_kv { 1 } else { g };
        let key = (span, group_rows, chunk, geom.resident_kv);
        let write = |buf: &mut [u8], row: usize, col: usize, v: f32| {
            let at = (row * span + col) * 4;
            buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
        };
        if self.mask_key == Some(key) {
            // Same mask identity: only the realized position moved. Newly
            // realized bucket columns become visible; a rewind re-erases.
            let (from, to) = (self.mask_pos.min(pos), self.mask_pos.max(pos));
            let value = if pos > self.mask_pos { 0.0 } else { erase };
            if from != to {
                for row in 0..group_rows * chunk {
                    for col in from..to {
                        write(&mut self.bufs.mask_b, row, col, value);
                    }
                }
            }
        } else {
            let visible = |col: usize, i: usize| {
                if col < geom.bucket {
                    col < pos
                } else {
                    col - geom.bucket <= i
                }
            };
            self.bufs.mask_b.clear();
            self.bufs.mask_b.resize(group_rows * chunk * span * 4, 0);
            for jj in 0..group_rows {
                for i in 0..chunk {
                    for col in 0..span {
                        if !visible(col, i) {
                            write(&mut self.bufs.mask_b, jj * chunk + i, col, erase);
                        }
                    }
                }
            }
            self.mask_key = Some(key);
        }
        self.mask_pos = pos;

        // The gather head takes the LAST REAL position's row; the verify head
        // has no `last_pos` port, so this is simply unbound there.
        self.bufs.lp = ((real - 1) as i64).to_le_bytes();
        // The fused KvCacheWrite ring position is the current realized length.
        self.bufs.pos = (pos as u32).to_le_bytes();
        Ok(())
    }

    /// Tile per-position rope rows `[chunk, d]` into the head-major layout
    /// `[rows · chunk, d]` the plan's exact-shape `Mul` consumes — in place.
    fn expand_rows_into(table: &[f32], rows: usize, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(rows * table.len() * 4);
        for _ in 0..rows {
            for v in table {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    /// Bind the decode input ports by NAME — buffers by value, carried K/V by
    /// reference. The port set is the plan's contract; the same binder serves
    /// the gather and verify passes (verify's graph omits `last_pos`).
    fn bind_inputs<'b>(
        port_info: &[crate::runner::PortInfo],
        b: &'b PassBuffers,
        past_k: &'b [Vec<u8>],
        past_v: &'b [Vec<u8>],
    ) -> Result<Vec<&'b [u8]>> {
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(port_info.len());
        for port in port_info {
            let buf: &[u8] = match port.name.as_str() {
                "input_ids" => &b.ids,
                "rope_cos_q" => &b.cos_q,
                "rope_sin_q" => &b.sin_q,
                "rope_cos_k" => &b.cos_k,
                "rope_sin_k" => &b.sin_k,
                "decode_mask" => &b.mask_b,
                "last_pos" => &b.lp,
                "decode_pos" => &b.pos,
                name => {
                    let (kind, layer) = name
                        .rsplit_once('_')
                        .and_then(|(k, l)| l.parse::<usize>().ok().map(|l| (k, l)))
                        .with_context(|| format!("unexpected decode input port `{name}`"))?;
                    match kind {
                        "past_k" => &past_k[layer],
                        "past_v" => &past_v[layer],
                        _ => bail!("unexpected decode input port `{name}`"),
                    }
                }
            };
            inputs.push(buf);
        }
        Ok(inputs)
    }

    /// One pass of `real = tokens.len()` positions through `runner` (whose
    /// geometry is `geom`; `real ≤ geom.chunk`, padded up to the chunk). Reads
    /// the gather head's single logit row and splices the new K/V into the
    /// carried past, advancing the realized length.
    /// `carry` (resident-KV plans only): the runner already holds the carried
    /// caches from ITS previous resident walk — bind them by label instead of
    /// re-ingesting the host bytes. The caller owns carrier tracking: pass
    /// `false` on a runner's first walk and whenever the host bytes are the
    /// truth (post-seeder, post-commit, post-regrow).
    fn pass(
        &mut self,
        runner: &mut impl LmSession,
        geom: DecodeGeometry,
        tokens: &[i64],
        carry: bool,
    ) -> Result<Vec<f32>> {
        let real = tokens.len();
        let (chunk, pos) = (geom.chunk, self.cur_len);
        let outputs = {
            self.refresh_buffers(geom, tokens)?;
            let port_info = runner.input_port_info();
            let inputs = Self::bind_inputs(&port_info, &self.bufs, &self.past_k, &self.past_v)?;
            if geom.resident_kv {
                // The resident walk: KV rides κ-labels inside the runner (no
                // re-hash, ring write moves in place); `None` outputs mean the
                // updated cache stayed resident — the host copy goes stale
                // until the owner syncs at a truth boundary. A session type
                // without the override returns every output materialized and
                // the byte handling below keeps the host copy current.
                runner.execute_kv_resident(&inputs, carry)?
            } else {
                runner.execute(&inputs)?.into_iter().map(Some).collect()
            }
        };
        let out_ports = runner.output_port_info();
        ensure!(
            outputs.len() == out_ports.len(),
            "decode pass returned {} outputs for {} ports",
            outputs.len(),
            out_ports.len()
        );

        let mut logits: Option<Vec<f32>> = None;
        let row = geom.head_dim * 4;
        for (port, out) in out_ports.iter().zip(outputs.iter()) {
            let Some(out) = out else {
                // Updated cache retained inside the runner (resident carry).
                continue;
            };
            match port.name.as_str() {
                "logits" => {
                    logits = Some(le_f32_vec(&out.bytes));
                }
                name => {
                    let (kind, layer) = name
                        .rsplit_once('_')
                        .and_then(|(k, l)| l.parse::<usize>().ok().map(|l| (k, l)))
                        .with_context(|| format!("unexpected decode output port `{name}`"))?;
                    let target = match kind {
                        "k_new" => &mut self.past_k[layer],
                        "v_new" => &mut self.past_v[layer],
                        _ => bail!("unexpected decode output port `{name}`"),
                    };
                    if geom.resident_kv {
                        // Fused (ADR-0019): the KvCacheWrite already ring-wrote
                        // the real rows at `pos`, so its output IS the whole
                        // updated cache `[1, kv, bucket, dh]` — carry it forward
                        // verbatim (on the byte path this is an honest copy; the
                        // addressed path turns it into a κ-move).
                        ensure!(
                            out.bytes.len() == geom.kv_heads * geom.bucket * row,
                            "{name} returned {} bytes, expected the full cache {}",
                            out.bytes.len(),
                            geom.kv_heads * geom.bucket * row
                        );
                        target.clone_from(&out.bytes);
                    } else {
                        ensure!(
                            out.bytes.len() == geom.kv_heads * chunk * row,
                            "{name} returned {} bytes, expected {}",
                            out.bytes.len(),
                            geom.kv_heads * chunk * row
                        );
                        // Splice the REAL rows [kv, real, dh] into bucket rows
                        // pos..pos+real — head-major, so one contiguous copy per
                        // kv head; pad rows never leave the output buffer.
                        for j in 0..geom.kv_heads {
                            let src = j * chunk * row;
                            let dst = (j * geom.bucket + pos) * row;
                            target[dst..dst + real * row]
                                .copy_from_slice(&out.bytes[src..src + real * row]);
                        }
                    }
                }
            }
        }
        self.cur_len += real;
        self.tokens.extend_from_slice(tokens);
        self.steps += 1;
        logits.context("decode pass produced no logits output")
    }

    /// Materialize a runner's resident-KV carry back into the host buffers —
    /// the truth boundary before anything reads `past_k`/`past_v` bytes (a
    /// verify pass on another runner, a bucket regrow, a rewind). A no-op for
    /// a runner that never carried.
    fn sync_kv_carry(
        &mut self,
        runner: &mut impl LmSession,
        geom: DecodeGeometry,
    ) -> Result<usize> {
        let mut synced = 0usize;
        for (name, bytes) in runner.take_kv_carry()? {
            let (kind, layer) = name
                .rsplit_once('_')
                .and_then(|(k, l)| l.parse::<usize>().ok().map(|l| (k, l)))
                .with_context(|| format!("unexpected carried cache port `{name}`"))?;
            let target = match kind {
                "past_k" => &mut self.past_k[layer],
                "past_v" => &mut self.past_v[layer],
                _ => bail!("unexpected carried cache port `{name}`"),
            };
            ensure!(
                bytes.len() == geom.kv_heads * geom.bucket * geom.head_dim * 4,
                "carried cache `{name}` is {} bytes, expected {}",
                bytes.len(),
                geom.kv_heads * geom.bucket * geom.head_dim * 4
            );
            *target = bytes;
            synced += 1;
        }
        Ok(synced)
    }

    /// A **verify pass** (row `speculative-decode`): run `tokens` through a
    /// runner carrying the all-positions verify head and return the logits at
    /// EACH of the `real` positions (`[real][vocab]`). Identical decode inputs
    /// to [`pass`] over the same carried past, so position `i`'s row equals a
    /// single-position decode at that absolute position — but all `real` rows
    /// come from ONE `M=real` forward. The carried state is NOT advanced: a
    /// verify is a trial pass, and only the prefix the model itself would have
    /// produced is later committed (through [`pass`]/`feed`).
    fn verify_pass(
        &mut self,
        runner: &mut impl LmSession,
        geom: DecodeGeometry,
        tokens: &[i64],
    ) -> Result<Vec<Vec<f32>>> {
        let real = tokens.len();
        let outputs = {
            self.refresh_buffers(geom, tokens)?;
            let port_info = runner.input_port_info();
            let inputs = Self::bind_inputs(&port_info, &self.bufs, &self.past_k, &self.past_v)?;
            runner.execute(&inputs)?
        };
        let out_ports = runner.output_port_info();
        let logits = out_ports
            .iter()
            .zip(outputs.iter())
            .find(|(p, _)| p.name == "logits")
            .map(|(_, o)| &o.bytes)
            .context("verify pass produced no logits output")?;
        // The all-positions head emits `[1, chunk, vocab]` — vocab is derived
        // from the output (discover's element_count folds in the chunk), and the
        // logits must tile evenly into `chunk` rows.
        ensure!(
            logits.len() % (geom.chunk * 4) == 0,
            "verify logits {} bytes not a whole [chunk={}, vocab] tensor",
            logits.len(),
            geom.chunk
        );
        let vocab = logits.len() / (geom.chunk * 4);
        // Rows 0..real are the real positions (pad rows are discarded).
        Ok((0..real)
            .map(|i| le_f32_vec(&logits[i * vocab * 4..(i + 1) * vocab * 4]))
            .collect())
    }

    /// One FOLDED **speculative-decode** batch (row `speculative-decode`):
    /// commit `pending` — the model's OWN token for the current position,
    /// decided by the caller's rule from the previous row — and verify `draft`
    /// behind it, all in ONE `M = 1 + draft.len()` pass. Folding the commit
    /// into the batch is what lets the verify runner alone carry the resident
    /// truth across batches (`carry`): no per-batch step on another runner, so
    /// no per-batch sync → re-hash → commit-copy → re-ingest traversals.
    ///
    /// `next_token(logits, position)` is the caller's own token rule — the
    /// SAME rule plain decode applies, a pure function of (logits, position) —
    /// so the committed sequence is byte-identical to stepping one token at a
    /// time, greedy OR sampled. Row 0 (after `pending`) seeds acceptance: a
    /// draft token is accepted iff it equals the model's own token at that
    /// position; the divergence token is the returned `bonus` (the next
    /// batch's `pending`). The fused KvCacheWrite ring-wrote ALL batch rows;
    /// `cur_len` advances by `1 + accepted`, so rejected rows sit past the
    /// realized length, mask-erased until overwritten — carrying the whole
    /// updated cache forward IS the accepted-prefix state. Returns
    /// `(accepted, bonus, resident)` — `resident` = the updated caches stayed
    /// inside the runner as κ-labels (nothing materialized).
    fn speculate_pass(
        &mut self,
        runner: &mut impl LmSession,
        geom: DecodeGeometry,
        pending: i64,
        draft: &[i64],
        next_token: &mut dyn FnMut(&[f32], u64) -> i64,
        carry: bool,
    ) -> Result<(Vec<i64>, i64, bool)> {
        let k = draft.len();
        let pos = self.cur_len;
        let row = geom.head_dim * 4;
        let mut batch = Vec::with_capacity(1 + k);
        batch.push(pending);
        batch.extend_from_slice(draft);
        let outputs = {
            self.refresh_buffers(geom, &batch)?;
            let port_info = runner.input_port_info();
            let inputs = Self::bind_inputs(&port_info, &self.bufs, &self.past_k, &self.past_v)?;
            if geom.resident_kv {
                runner.execute_kv_resident(&inputs, carry)?
            } else {
                runner.execute(&inputs)?.into_iter().map(Some).collect()
            }
        };
        let out_ports = runner.output_port_info();
        let logits = out_ports
            .iter()
            .zip(outputs.iter())
            .find(|(p, _)| p.name == "logits")
            .and_then(|(_, o)| o.as_ref())
            .map(|o| o.bytes.as_slice())
            .context("verify pass produced no logits output")?;
        ensure!(
            logits.len() % (geom.chunk * 4) == 0,
            "verify logits {} bytes not a whole [chunk={}, vocab] tensor",
            logits.len(),
            geom.chunk
        );
        let vocab = logits.len() / (geom.chunk * 4);

        // Acceptance under the caller's rule. Row `i` is the logits AFTER
        // batch position `i` (absolute position pos + i), so row 0 — after the
        // committed `pending` — decides the model's token at pos + 1.
        let mut bonus = {
            let row0 = le_f32_vec(&logits[..vocab * 4]);
            next_token(&row0, (pos + 1) as u64)
        };
        let mut accepted = 0usize;
        while accepted < k {
            if draft[accepted] != bonus {
                break;
            }
            let next = le_f32_vec(&logits[(accepted + 1) * vocab * 4..(accepted + 2) * vocab * 4]);
            bonus = next_token(&next, (pos + accepted + 2) as u64);
            accepted += 1;
        }

        // Commit the batch's K/V. Resident (`None` outputs): the caches stayed
        // inside the runner — nothing to copy, the mask erases the rejected
        // tail. Materialized resident form: the output IS the whole updated
        // cache. Legacy form: splice rows 0..1+accepted into pos..
        let mut resident = geom.resident_kv;
        for (port, out) in out_ports.iter().zip(outputs.iter()) {
            if port.name == "logits" {
                continue;
            }
            let Some(out) = out else {
                continue;
            };
            let (kind, layer) = port
                .name
                .rsplit_once('_')
                .and_then(|(kd, l)| l.parse::<usize>().ok().map(|l| (kd, l)))
                .with_context(|| format!("unexpected decode output port `{}`", port.name))?;
            let target = match kind {
                "k_new" => &mut self.past_k[layer],
                "v_new" => &mut self.past_v[layer],
                _ => bail!("unexpected decode output port `{}`", port.name),
            };
            if geom.resident_kv {
                resident = false;
                ensure!(
                    out.bytes.len() == geom.kv_heads * geom.bucket * row,
                    "{} returned {} bytes, expected the full cache {}",
                    port.name,
                    out.bytes.len(),
                    geom.kv_heads * geom.bucket * row
                );
                target.clone_from(&out.bytes);
            } else {
                resident = false;
                let commit = 1 + accepted;
                for h in 0..geom.kv_heads {
                    let src = h * geom.chunk * row;
                    let dst = (h * geom.bucket + pos) * row;
                    target[dst..dst + commit * row]
                        .copy_from_slice(&out.bytes[src..src + commit * row]);
                }
            }
        }
        self.cur_len += 1 + accepted;
        self.tokens.push(pending);
        self.tokens.extend_from_slice(&draft[..accepted]);
        self.steps += 1;
        Ok((draft[..accepted].to_vec(), bonus, resident))
    }
}

/// Decode a little-endian f32 byte slice into a vector.
fn le_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// A generation session over the decode plan — generic over the
/// [`LmSession`] executing each pass, so the same engine drives a monolithic
/// decode archive (`HoloRunner`) or the staged decode pipeline
/// (`StagedRunner`).
pub struct DecodeSession<S: LmSession> {
    runner: S,
    geom: DecodeGeometry,
    /// Chunked-prefill seeder (row `chunked-prefill`): same buffers, `chunk`
    /// positions per pass. Dropped on bucket growth (its bucket went stale);
    /// the owner re-installs one lazily.
    seeder: Option<(S, DecodeGeometry)>,
    /// The model's trained position ceiling (`context_length` metadata) — the
    /// only semantic bound on generation length.
    context_length: u64,
    rebuild: Option<DecodeRebuild<S>>,
    state: DecodeState,
    /// Where the carried-K/V truth lives (ADR-0019 increment 3b).
    truth: KvTruth,
    /// Geometry of the caller-held verify runner while it carries the truth
    /// (`KvTruth::Verify`) — what `end_speculation` sizes the sync against.
    verify_geom: Option<DecodeGeometry>,
}

/// Where the carried-K/V truth lives. On a resident-KV plan the hot runner
/// keeps the updated caches as κ-labels between walks (no per-step re-hash,
/// no copy — the ring write moves in place); the host byte buffers then go
/// stale until a **truth boundary** syncs them back: a verify pass on another
/// runner, a seeder↔step hand-off, or a bucket regrow. `Poisoned` marks a
/// failed resident walk (the carried labels may be consumed) — the session
/// must be reset before further use, never silently continued on stale bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvTruth {
    /// The host `past_k`/`past_v` buffers are current (legacy plans always).
    Host,
    /// The step runner carries the truth as resident κ-labels.
    Step,
    /// The prefill seeder carries the truth as resident κ-labels.
    Seeder,
    /// The CALLER-HELD verify runner carries the truth as resident κ-labels
    /// (a folded speculative run, [`DecodeSession::speculate`]). The session
    /// cannot reach that runner on its own, so every other path refuses loud
    /// until [`DecodeSession::end_speculation`] hands the truth back.
    Verify,
    /// A resident walk failed mid-flight; the carried truth may be lost.
    Poisoned,
}

/// Bytes for one layer's carried K (or V) buffer: `kv_heads · bucket ·
/// head_dim · 4`, in CHECKED arithmetic — a bucket whose f32 K/V exceeds the
/// host's address space (`usize::MAX`, 4 GiB on wasm32) fails loud here naming
/// the shape, rather than wrapping into a silently-undersized buffer. The
/// bound is the target's own address space, never an arbitrary ceiling.
fn kv_buffer_bytes(geom: &DecodeGeometry) -> Result<usize> {
    geom.kv_heads
        .checked_mul(geom.bucket)
        .and_then(|n| n.checked_mul(geom.head_dim))
        .and_then(|n| n.checked_mul(4))
        .with_context(|| {
            format!(
                "the carried K/V buffer ({} kv · {} bucket · {} head_dim · 4 B) exceeds this \
                 host's address space",
                geom.kv_heads, geom.bucket, geom.head_dim
            )
        })
}

impl<S: LmSession> DecodeSession<S> {
    /// Open a session over a compiled step plan (`chunk = 1`). `rope` is the
    /// model's complete rotary law from its own config (the graph consumes
    /// rope as runtime data, so the table generator lives with the engine);
    /// `context_length` is the model's trained ceiling.
    pub fn new(runner: S, rope: RopeSpec, context_length: u64) -> Result<Self> {
        let geom = DecodeGeometry::discover(&runner)?;
        ensure!(
            geom.chunk == 1,
            "the session's main runner must be the step plan (chunk 1), got chunk {}",
            geom.chunk
        );
        rope.validate(geom.head_dim)
            .map_err(|e| anyhow::anyhow!("the session's rotary law is malformed: {e}"))?;
        let kv_bytes = kv_buffer_bytes(&geom)?;
        Ok(Self {
            runner,
            geom,
            seeder: None,
            context_length,
            rebuild: None,
            state: DecodeState {
                rope,
                past_k: vec![vec![0u8; kv_bytes]; geom.layers],
                past_v: vec![vec![0u8; kv_bytes]; geom.layers],
                cur_len: 0,
                tokens: Vec::new(),
                steps: 0,
                bufs: PassBuffers::default(),
                rope_freqs: Vec::new(),
                rope_scale: 1.0,
                rope_valid_for: None,
                rope_cos: Vec::new(),
                rope_sin: Vec::new(),
                mask_key: None,
                mask_pos: 0,
            },
            truth: KvTruth::Host,
            verify_geom: None,
        })
    }

    /// Materialize any runner-resident K/V carry back into the host buffers —
    /// the truth boundary before anything reads `past_k`/`past_v` bytes.
    /// Fails loud on a poisoned carry rather than continuing on stale bytes.
    fn sync_truth(&mut self) -> Result<()> {
        match self.truth {
            KvTruth::Host => {}
            KvTruth::Step => {
                self.state.sync_kv_carry(&mut self.runner, self.geom)?;
            }
            KvTruth::Seeder => {
                if let Some((seeder, sgeom)) = self.seeder.as_mut() {
                    self.state.sync_kv_carry(seeder, *sgeom)?;
                }
            }
            KvTruth::Verify => bail!(
                "the carried K/V truth lives in the speculation verify runner — \
                 call end_speculation(verify_runner) before any other pass"
            ),
            KvTruth::Poisoned => bail!(
                "the carried K/V was lost by a failed resident decode walk — \
                 reset the session before continuing"
            ),
        }
        self.truth = KvTruth::Host;
        Ok(())
    }

    /// Attach a rebuild source so bucket exhaustion regrows geometrically
    /// instead of failing.
    pub fn with_rebuild(mut self, rebuild: DecodeRebuild<S>) -> Self {
        self.rebuild = Some(rebuild);
        self
    }

    /// Install a chunked-prefill seeder (row `chunked-prefill`): a plan over
    /// the SAME bucket processing `chunk > 1` positions per pass.
    /// [`Self::feed`] uses it for multi-token prefill; growth drops it
    /// (stale bucket).
    pub fn set_seeder(&mut self, seeder: S) -> Result<()> {
        let geom = DecodeGeometry::discover(&seeder)?;
        ensure!(
            geom.bucket == self.geom.bucket
                && geom.layers == self.geom.layers
                && geom.kv_heads == self.geom.kv_heads
                && geom.heads == self.geom.heads
                && geom.head_dim == self.geom.head_dim,
            "seeder geometry {geom:?} does not match the session's {:?}",
            self.geom
        );
        ensure!(
            geom.chunk > 1,
            "a seeder must process more than one position per pass"
        );
        self.seeder = Some((seeder, geom));
        Ok(())
    }

    /// Whether a chunked-prefill seeder is installed (and its chunk).
    pub fn seeder_chunk(&self) -> Option<usize> {
        self.seeder.as_ref().map(|(_, g)| g.chunk)
    }

    /// Run a **verify pass** (row `speculative-decode`) over `tokens` on a
    /// runner carrying the all-positions verify head, returning the model's
    /// logits at EACH position (`[tokens.len()][vocab]`) from ONE
    /// `M = tokens.len()` forward — the batched matmul shape the substrate runs
    /// efficiently. The session state is UNCHANGED: a verify is a trial pass over
    /// the current carried past; only the prefix the model itself would have
    /// produced is committed, with `step`/`feed`. The verify runner shares this session's
    /// decode geometry (same bucket/layers/heads); its chunk is the draft
    /// length `K`, so `tokens.len() ≤ K`.
    pub fn verify(&mut self, verify_runner: &mut S, tokens: &[i64]) -> Result<Vec<Vec<f32>>> {
        // The verify runner binds the host K/V bytes — materialize any
        // resident carry first (truth boundary).
        self.sync_truth()?;
        let geom = self.verify_geometry(verify_runner, tokens.len())?;
        self.state.verify_pass(verify_runner, geom, tokens)
    }

    /// One FOLDED **speculative-decode** batch (row `speculative-decode`):
    /// commit `pending` (the model's own token for the current position,
    /// decided by `next_token` from the previous row) and verify `draft`
    /// behind it in ONE `M = 1 + draft.len()` pass on the verify runner —
    /// which then CARRIES the resident K/V truth across batches. During a
    /// speculative run the session's other passes refuse loud; the caller
    /// hands the truth back with [`end_speculation`] before stepping again.
    /// Returns `(accepted, bonus)`: the accepted draft prefix (the model's
    /// own tokens, byte for byte) and the model's token at the divergence —
    /// the next batch's `pending`. Output-identical to stepping one token at
    /// a time under the same rule, at ANY temperature.
    ///
    /// A batch never grows the bucket (growth would stale the verify runner
    /// while it holds the truth): the caller retires speculation before the
    /// bucket fills, and an overflowing batch is refused loud here.
    ///
    /// [`end_speculation`]: DecodeSession::end_speculation
    pub fn speculate(
        &mut self,
        verify_runner: &mut S,
        pending: i64,
        draft: &[i64],
        next_token: &mut dyn FnMut(&[f32], u64) -> i64,
    ) -> Result<(Vec<i64>, i64)> {
        ensure!(
            self.state.cur_len + 1 + draft.len() <= self.geom.bucket,
            "a speculative batch of {} rows at position {} would leave the bucket ({}) — \
             retire speculation and let plain decode grow it",
            1 + draft.len(),
            self.state.cur_len,
            self.geom.bucket
        );
        // Hand-off INTO speculation: whoever carries the truth materializes
        // once; the verify runner ingests current bytes on its first batch and
        // carries from there.
        if matches!(self.truth, KvTruth::Step | KvTruth::Seeder) {
            self.sync_truth()?;
        }
        ensure!(
            self.truth != KvTruth::Poisoned,
            "the carried K/V was lost by a failed resident decode walk — reset the session"
        );
        let vgeom = self.verify_geometry(verify_runner, 1 + draft.len())?;
        let carry = vgeom.resident_kv && self.truth == KvTruth::Verify;
        if vgeom.resident_kv {
            self.truth = KvTruth::Poisoned; // until the walk returns
        }
        let (accepted, bonus, resident) =
            self.state
                .speculate_pass(verify_runner, vgeom, pending, draft, next_token, carry)?;
        self.truth = if resident {
            self.verify_geom = Some(vgeom);
            KvTruth::Verify
        } else {
            self.verify_geom = None;
            KvTruth::Host
        };
        Ok((accepted, bonus))
    }

    /// Hand the carried K/V truth back from the speculation verify runner to
    /// the host buffers — the truth boundary that ends a speculative run
    /// (bucket retire, empty-draft fallback to plain steps, or turn end).
    /// A no-op when the verify runner never carried. The runner passed here
    /// must be THE one `speculate` ran on: a runner with no carry to give
    /// back fails loud rather than leaving stale host bytes as "truth".
    pub fn end_speculation(&mut self, verify_runner: &mut S) -> Result<()> {
        if self.truth != KvTruth::Verify {
            return Ok(());
        }
        let vgeom = self
            .verify_geom
            .take()
            .context("verify truth recorded without its geometry")?;
        self.truth = KvTruth::Poisoned; // until the sync lands
        let synced = self.state.sync_kv_carry(verify_runner, vgeom)?;
        ensure!(
            synced == 2 * vgeom.layers,
            "end_speculation synced {synced} carried caches, expected {} — \
             this is not the runner that carried the speculative truth",
            2 * vgeom.layers
        );
        self.truth = KvTruth::Host;
        Ok(())
    }

    /// Discover and validate a verify runner's geometry: it must share this
    /// session's bucket/layers/heads (its chunk bounds the batch width — for a
    /// folded speculative batch, the pending token plus the draft), and the
    /// batch must fit in one pass.
    fn verify_geometry(&self, verify_runner: &mut S, draft_len: usize) -> Result<DecodeGeometry> {
        let geom = DecodeGeometry::discover(verify_runner)?;
        ensure!(
            geom.bucket == self.geom.bucket
                && geom.layers == self.geom.layers
                && geom.kv_heads == self.geom.kv_heads
                && geom.heads == self.geom.heads
                && geom.head_dim == self.geom.head_dim,
            "verify runner geometry {geom:?} does not match the session's {:?}",
            self.geom
        );
        ensure!(
            draft_len > 0 && draft_len <= geom.chunk,
            "a verify pass drafts 1..={} tokens, got {}",
            geom.chunk,
            draft_len
        );
        Ok(geom)
    }

    pub fn geometry(&self) -> DecodeGeometry {
        self.geom
    }

    pub fn realized_len(&self) -> usize {
        self.state.cur_len
    }

    /// The model's trained position ceiling — the only semantic bound on
    /// generation length.
    pub fn context_len(&self) -> u64 {
        self.context_length
    }

    /// The executing session (e.g. for its residency/bandwidth instruments).
    pub fn runner(&self) -> &S {
        &self.runner
    }

    /// Rewind to position 0 for a fresh sequence. The K/V buffers keep their
    /// bytes — every unrealized row is erased inside the softmax by the
    /// decode mask, so stale content is unreachable by construction — and
    /// the runner keeps its materialized stages (a warm turn pays decode,
    /// never rematerialization).
    pub fn reset(&mut self) {
        self.rewind_to(0);
    }

    /// Rewind to position `len` (≤ the realized length): the carried K/V
    /// rows for positions `0..len` stay live, rows past `len` become
    /// unrealized (masked out until overwritten). Cross-turn retention:
    /// a new sequence sharing a realized prefix pays only its suffix.
    /// No truth sync is needed here: a resident carry holds the FULL bucket,
    /// and every row past the rewound length is erased by the decode mask
    /// until overwritten — the carried labels remain exactly as valid as the
    /// host bytes. A `Poisoned` carry heals at `len == 0`: with no realized
    /// positions, every row is masked-unreachable, so ANY bucket content is
    /// the correct empty state (a rewind INTO lost content stays poisoned and
    /// keeps failing loud).
    pub fn rewind_to(&mut self, len: usize) {
        let len = len.min(self.state.cur_len);
        self.state.cur_len = len;
        self.state.tokens.truncate(len);
        // A rewind while the verify runner holds the truth abandons rows the
        // host never received: at 0 every row is masked-unreachable (any
        // content is the correct empty state); anywhere else the host bytes
        // are stale, so the truth is POISONED — later passes fail loud
        // instead of continuing on wrong numbers.
        if self.truth == KvTruth::Verify {
            self.verify_geom = None;
            self.truth = if len == 0 {
                KvTruth::Host
            } else {
                KvTruth::Poisoned
            };
        }
        if len == 0 && self.truth == KvTruth::Poisoned {
            self.truth = KvTruth::Host;
        }
    }

    /// The realized token at each carried position, in order.
    pub fn realized_tokens(&self) -> &[i64] {
        &self.state.tokens
    }

    /// Passes executed over this session's lifetime (retention/prefill
    /// instrument).
    pub fn steps_taken(&self) -> u64 {
        self.state.steps
    }

    /// Kernel-dispatch counters for the last pass (perf attribution).
    pub fn last_dispatched(&self) -> u64 {
        self.runner.pass_dispatched()
    }

    pub fn last_skipped(&self) -> u64 {
        self.runner.pass_skipped()
    }

    /// Grow the bucket geometrically (clamped to the context ceiling) and
    /// copy every layer's realized rows into the wider buffers. Drops the
    /// seeder — its bucket went stale; the owner re-installs one lazily.
    fn grow(&mut self) -> Result<()> {
        if self.rebuild.is_none() {
            bail!(
                "decode bucket ({}) exhausted and the session has no rebuild source",
                self.geom.bucket
            );
        }
        // The widen below reads the host K/V bytes, and both carriers are
        // dropped/replaced by the rebuild — materialize any resident carry
        // first (truth boundary).
        self.sync_truth()?;
        // The next bucket comes from the SAME geometric policy the rebuild
        // closure uses (`engine::geometric_window`), so the rebuilt runner's
        // bucket matches `new_bucket` by construction — never a re-implemented
        // `* 2` that could silently drift from the window policy.
        let cap = self.context_length.min(usize::MAX as u64) as usize;
        let new_bucket = crate::engine::geometric_window(self.geom.bucket + 1, cap) as u64;
        ensure!(
            new_bucket > self.geom.bucket as u64,
            "decode bucket cannot grow past the model's context ({})",
            self.context_length
        );
        // Free BOTH auxiliary residencies BEFORE compiling and materializing the
        // wider bucket: the prefill SEEDER (dropped — its bucket goes stale on
        // growth anyway) and the OUTGOING step runner (evicted — it is replaced
        // below). Until they are freed the wasm linear memory (which never
        // shrinks) must hold the old resident set AND the new bucket's stage
        // compilation at once, and that peak is what aborts growth at scale — a
        // bare `RuntimeError: unreachable` (a `memory.grow` past the 4 GiB
        // ceiling that the residency ledger cannot foresee, because compilation /
        // module memory is not stage-weight residency). Freeing first lets the
        // allocator REUSE that space for the new bucket instead of growing past
        // the ceiling. Harmless where memory is ample (native / 64-bit: the old
        // runner is dropped moments later regardless); on a rebuild failure the
        // old runner survives and simply re-materializes on next use.
        self.seeder = None;
        self.runner.evict_resident();
        let rebuild = self
            .rebuild
            .as_mut()
            .expect("rebuild source present (checked above)");
        let runner = rebuild(new_bucket)?;
        let geom = DecodeGeometry::discover(&runner)?;
        ensure!(
            geom.bucket as u64 == new_bucket
                && geom.chunk == 1
                && geom.layers == self.geom.layers
                && geom.kv_heads == self.geom.kv_heads
                && geom.head_dim == self.geom.head_dim,
            "rebuilt decode archive geometry {:?} does not extend {:?}",
            geom,
            self.geom
        );

        // Checked before allocating: the wider bucket's f32 K/V must fit the
        // host's address space (same loud bound as `new`).
        let wide_bytes = kv_buffer_bytes(&geom)?;
        let row = self.geom.head_dim * 4;
        let (old_b, new_b) = (self.geom.bucket, geom.bucket);
        let realized = self.state.cur_len;
        let widen = |buffers: &mut Vec<Vec<u8>>, kv: usize| {
            for buf in buffers.iter_mut() {
                let mut wide = vec![0u8; wide_bytes];
                for j in 0..kv {
                    let src = j * old_b * row;
                    let dst = j * new_b * row;
                    let len = realized * row;
                    wide[dst..dst + len].copy_from_slice(&buf[src..src + len]);
                }
                *buf = wide;
            }
        };
        widen(&mut self.state.past_k, geom.kv_heads);
        widen(&mut self.state.past_v, geom.kv_heads);
        self.runner = runner;
        self.geom = geom;
        // The seeder was already dropped before the rebuild (freeing its
        // residency for the wider bucket's compilation); nothing to clear here.
        Ok(())
    }

    /// Grow until `n` more positions fit (growth drops the seeder).
    fn ensure_capacity(&mut self, n: usize) -> Result<()> {
        ensure!(
            (self.state.cur_len + n) as u64 <= self.context_length,
            "the model's trained context ({}) is exhausted",
            self.context_length
        );
        while self.state.cur_len + n > self.geom.bucket {
            self.grow()?;
        }
        Ok(())
    }

    /// One decode step: feed `token` at the next absolute position and return
    /// the logit row. On a resident-KV plan the updated caches stay inside the
    /// step runner as κ-labels between steps (no re-hash, no copy — the ring
    /// write moves in place); legacy plans splice into the host buffers.
    pub fn step(&mut self, token: i64) -> Result<Vec<f32>> {
        ensure!(
            self.truth != KvTruth::Verify,
            "the carried K/V truth lives in the speculation verify runner — \
             call end_speculation(verify_runner) before stepping"
        );
        self.ensure_capacity(1)?;
        // Hand-off: if the seeder carries the truth (a resident prefill just
        // ran), materialize once so this runner ingests current bytes.
        if self.truth == KvTruth::Seeder {
            self.sync_truth()?;
        }
        ensure!(
            self.truth != KvTruth::Poisoned,
            "the carried K/V was lost by a failed resident decode walk — reset the session"
        );
        let carry = self.geom.resident_kv && self.truth == KvTruth::Step;
        if self.geom.resident_kv {
            self.truth = KvTruth::Poisoned; // until the walk returns
        }
        let row = self
            .state
            .pass(&mut self.runner, self.geom, &[token], carry)?;
        self.truth = if self.geom.resident_kv {
            KvTruth::Step
        } else {
            KvTruth::Host
        };
        Ok(row)
    }

    /// Feed `tokens` at the next absolute positions and return the logit row
    /// at the LAST token (the sampler's row). Uses the chunked seeder when
    /// installed — ceil(n/chunk) passes, the final partial chunk padded —
    /// else one step per token.
    pub fn feed(&mut self, tokens: &[i64]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "feed needs at least one token");
        ensure!(
            self.truth != KvTruth::Verify,
            "the carried K/V truth lives in the speculation verify runner — \
             call end_speculation(verify_runner) before feeding"
        );
        let mut row: Option<Vec<f32>> = None;
        let mut rest = tokens;
        while !rest.is_empty() {
            // Capacity first: growth drops the seeder, so re-check it after.
            // A pass never exceeds the bucket: the chunk plan's own span is
            // bucket + chunk, and the splice lands within pos..pos+real.
            let take = match &self.seeder {
                Some((_, g)) if rest.len() >= 2 => rest.len().min(g.chunk),
                _ => 1,
            };
            self.ensure_capacity(take)?;
            let (now, later) = rest.split_at(take.min(rest.len()));
            let use_seeder = self.seeder.is_some() && now.len() >= 2;
            row = Some(if use_seeder {
                // Hand-off: if the STEP runner carries the truth, materialize
                // once so the seeder ingests current bytes.
                if self.truth == KvTruth::Step {
                    self.sync_truth()?;
                }
                ensure!(
                    self.truth != KvTruth::Poisoned,
                    "the carried K/V was lost by a failed resident decode walk — reset the session"
                );
                let (seeder, sgeom) = self.seeder.as_mut().expect("checked above");
                let carry = sgeom.resident_kv && self.truth == KvTruth::Seeder;
                let resident = sgeom.resident_kv;
                let sgeom = *sgeom;
                if resident {
                    self.truth = KvTruth::Poisoned;
                }
                let r = self.state.pass(seeder, sgeom, now, carry)?;
                self.truth = if resident {
                    KvTruth::Seeder
                } else {
                    KvTruth::Host
                };
                r
            } else {
                // One position at a time on the step plan.
                let mut last = None;
                for &t in now {
                    if self.truth == KvTruth::Seeder {
                        self.sync_truth()?;
                    }
                    ensure!(
                        self.truth != KvTruth::Poisoned,
                        "the carried K/V was lost by a failed resident decode walk — reset the session"
                    );
                    let carry = self.geom.resident_kv && self.truth == KvTruth::Step;
                    if self.geom.resident_kv {
                        self.truth = KvTruth::Poisoned;
                    }
                    last = Some(self.state.pass(&mut self.runner, self.geom, &[t], carry)?);
                    self.truth = if self.geom.resident_kv {
                        KvTruth::Step
                    } else {
                        KvTruth::Host
                    };
                }
                last.expect("now is non-empty")
            });
            rest = later;
        }
        // Reclaim the prefill seeder's residency for the hot step runner when
        // the two would not both fit under a hard address ceiling: the seeder is
        // idle for the rest of the turn, so windowing the step plan on every
        // generated token is far costlier than re-seeding the next prefill. `2×`
        // because the step plan is the SAME model as the seeder — keeping the
        // seeder resident denies the step plan an equal-sized resident set.
        // Where both fit (a small model, or a 64-bit host whose budget is
        // effectively unbounded), nothing is reclaimed and the warm turn stays
        // warm. Parametric: the threshold is the model's own footprint against
        // its own ceiling, never a fixed size.
        if let Some((seeder, _)) = self.seeder.as_mut() {
            let (footprint, budget) = seeder.residency_pressure();
            if budget > 0 && footprint.saturating_mul(2) > budget {
                seeder.evict_resident();
            }
        }
        row.context("feed processed no tokens")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(kv_heads: usize, bucket: usize, head_dim: usize) -> DecodeGeometry {
        DecodeGeometry {
            layers: 2,
            kv_heads,
            heads: kv_heads * 2,
            head_dim,
            bucket,
            chunk: 1,
            vocab: 32,
            resident_kv: false,
        }
    }

    #[test]
    fn kv_buffer_bytes_is_exact_for_a_normal_geometry() {
        let g = geom(2, 128, 16);
        assert_eq!(kv_buffer_bytes(&g).unwrap(), 2 * 128 * 16 * 4);
    }

    #[test]
    fn kv_buffer_bytes_fails_loud_on_address_space_overflow() {
        // A bucket whose f32 K/V exceeds usize must fail loud naming the shape,
        // never wrap into a silently-undersized buffer. Only reachable when
        // usize is 64-bit here, but the CHECK is what matters (the wasm32
        // build hits it at a realistic 4 GiB bucket).
        let g = geom(8, usize::MAX / 64, 128);
        let err = kv_buffer_bytes(&g).expect_err("must overflow");
        assert!(
            err.to_string().contains("address space"),
            "error names the address-space bound: {err}"
        );
    }
    #[test]
    fn carried_kv_bytes_is_derived_from_the_attention_shape() {
        let g = DecodeGeometry {
            layers: 28,
            kv_heads: 2,
            heads: 12,
            head_dim: 128,
            bucket: 128,
            chunk: 1,
            vocab: 151936,
            resident_kv: false,
        };
        // 2 (K and V) · 28 layers · 2 kv · 128 head_dim · 4 B
        let per_row = 2u64 * 28 * 2 * 128 * 4;
        assert_eq!(g.carried_kv_bytes_per_row().unwrap(), per_row);
        assert_eq!(g.carried_kv_bytes().unwrap(), per_row * 128);
    }

    #[test]
    fn carried_kv_bytes_scales_linearly_with_the_bucket() {
        let base = DecodeGeometry {
            layers: 28,
            kv_heads: 2,
            heads: 12,
            head_dim: 128,
            bucket: 64,
            chunk: 1,
            vocab: 32,
            resident_kv: false,
        };
        let wide = DecodeGeometry {
            bucket: 32768,
            ..base
        };
        let per_row = base.carried_kv_bytes_per_row().unwrap();
        assert_eq!(base.carried_kv_bytes().unwrap(), per_row * 64);
        assert_eq!(wide.carried_kv_bytes().unwrap(), per_row * 32768);
        // The point of deriving it: at a long context the carried K/V is
        // GIGABYTES — larger than any fixed reserve that pretends to cover it.
        assert!(
            wide.carried_kv_bytes().unwrap() > (1u64 << 30),
            "a 32k bucket on this shape carries more than a gibibyte of K/V"
        );
    }

    #[test]
    fn an_unaddressable_kv_geometry_fails_loud() {
        let g = DecodeGeometry {
            layers: usize::MAX,
            kv_heads: usize::MAX,
            heads: 1,
            head_dim: usize::MAX,
            bucket: usize::MAX,
            chunk: 1,
            vocab: 1,
            resident_kv: false,
        };
        assert!(g.carried_kv_bytes_per_row().is_err());
        assert!(g.carried_kv_bytes().is_err());
    }
}
