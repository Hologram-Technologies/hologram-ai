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
}

impl DecodeGeometry {
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
        let pk = inputs
            .iter()
            .find(|p| p.name == "past_k_0")
            .context("decode plan lacks past_k_0")?;
        ensure!(
            pk.shape.len() == 3,
            "past_k_0 must be [kv, bucket, head_dim], got {:?}",
            pk.shape
        );
        let (kv_heads, bucket, head_dim) = (pk.shape[0], pk.shape[1], pk.shape[2]);

        let outputs = session.output_port_info();
        let kn = outputs
            .iter()
            .find(|p| p.name == "k_new_0")
            .context("decode plan lacks a k_new_0 output")?;
        ensure!(
            kn.shape.len() == 3 && kn.shape[0] == kv_heads && kn.shape[2] == head_dim,
            "k_new_0 must be [kv, chunk, head_dim], got {:?}",
            kn.shape
        );
        let chunk = kn.shape[1];

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
    rope_theta: f32,
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
}

impl DecodeState {
    /// Standard non-interleaved RoPE rows for absolute positions
    /// `base..base+chunk` (`[chunk · head_dim]`, halves duplicated so
    /// `cos[j]`/`sin[j]` pair with the rotate-half partner `j ± d/2`).
    fn rope_rows(&self, base: usize, chunk: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
        let half = d / 2;
        let mut cos = vec![0.0f32; chunk * d];
        let mut sin = vec![0.0f32; chunk * d];
        for i in 0..chunk {
            for j in 0..half {
                let inv_freq = 1.0 / (self.rope_theta as f64).powf(2.0 * j as f64 / d as f64);
                let angle = (base + i) as f64 * inv_freq;
                let (s, c) = (angle.sin() as f32, angle.cos() as f32);
                cos[i * d + j] = c;
                cos[i * d + j + half] = c;
                sin[i * d + j] = s;
                sin[i * d + j + half] = s;
            }
        }
        (cos, sin)
    }

    /// Tile per-position rope rows `[chunk, d]` to the head-major layout
    /// `[rows · chunk, d]` the plan's exact-shape `Mul` consumes.
    fn expand_rows(table: &[f32], rows: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * table.len() * 4);
        for _ in 0..rows {
            out.extend(table.iter().flat_map(|v| v.to_le_bytes()));
        }
        out
    }

    /// One pass of `real = tokens.len()` positions through `runner` (whose
    /// geometry is `geom`; `real ≤ geom.chunk`, padded up to the chunk).
    fn pass(
        &mut self,
        runner: &mut impl LmSession,
        geom: DecodeGeometry,
        tokens: &[i64],
    ) -> Result<Vec<f32>> {
        let real = tokens.len();
        ensure!(
            0 < real && real <= geom.chunk,
            "a pass takes 1..={} tokens, got {real}",
            geom.chunk
        );
        let (chunk, pos) = (geom.chunk, self.cur_len);
        let g = geom.heads / geom.kv_heads;

        // ids: real tokens, padded to the chunk (pad rows are unreachable —
        // masked below the realized length until overwritten).
        let mut ids_v = vec![0i64; chunk];
        ids_v[..real].copy_from_slice(tokens);
        let ids: Vec<u8> = ids_v.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Rope tables at absolute positions pos..pos+chunk, head-major.
        let (cos, sin) = self.rope_rows(pos, chunk, geom.head_dim);
        let cos_q = Self::expand_rows(&cos, geom.heads);
        let sin_q = Self::expand_rows(&sin, geom.heads);
        let cos_k = Self::expand_rows(&cos, geom.kv_heads);
        let sin_k = Self::expand_rows(&sin, geom.kv_heads);

        // The mask [g·chunk, bucket+chunk]: row jj·chunk + i is position i's
        // row — bucket cols < pos visible, chunk cols ≤ i visible.
        let span = geom.bucket + chunk;
        let mut mask = vec![0.0f32; g * chunk * span];
        for jj in 0..g {
            for i in 0..chunk {
                let base = (jj * chunk + i) * span;
                for (col, slot) in mask[base..base + span].iter_mut().enumerate() {
                    let visible = if col < geom.bucket {
                        col < pos
                    } else {
                        col - geom.bucket <= i
                    };
                    if !visible {
                        *slot = -1e9;
                    }
                }
            }
        }
        let mask_b: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();

        // The head gathers the LAST REAL position's row (seq-C plans declare
        // `last_pos`; the seq-1 plan is already at the sampler's position).
        let lp = ((real - 1) as i64).to_le_bytes();

        // Bind by port NAME — the ports are the plan's contract.
        let port_info = runner.input_port_info();
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(port_info.len());
        for port in &port_info {
            let buf: &[u8] = match port.name.as_str() {
                "input_ids" => &ids,
                "rope_cos_q" => &cos_q,
                "rope_sin_q" => &sin_q,
                "rope_cos_k" => &cos_k,
                "rope_sin_k" => &sin_k,
                "decode_mask" => &mask_b,
                "last_pos" => &lp,
                name => {
                    let (kind, layer) = name
                        .rsplit_once('_')
                        .and_then(|(k, l)| l.parse::<usize>().ok().map(|l| (k, l)))
                        .with_context(|| format!("unexpected decode input port `{name}`"))?;
                    match kind {
                        "past_k" => &self.past_k[layer],
                        "past_v" => &self.past_v[layer],
                        _ => bail!("unexpected decode input port `{name}`"),
                    }
                }
            };
            inputs.push(buf);
        }

        let outputs = runner.execute(&inputs)?;
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
            match port.name.as_str() {
                "logits" => {
                    logits = Some(
                        out.bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                            .collect(),
                    );
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
        self.cur_len += real;
        self.tokens.extend_from_slice(tokens);
        self.steps += 1;
        logits.context("decode pass produced no logits output")
    }
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
}

impl<S: LmSession> DecodeSession<S> {
    /// Open a session over a compiled step plan (`chunk = 1`). `rope_theta`
    /// comes from the model's own config (the graph consumes rope as runtime
    /// data, so the table generator lives with the engine); `context_length`
    /// is the model's trained ceiling.
    pub fn new(runner: S, rope_theta: f32, context_length: u64) -> Result<Self> {
        let geom = DecodeGeometry::discover(&runner)?;
        ensure!(
            geom.chunk == 1,
            "the session's main runner must be the step plan (chunk 1), got chunk {}",
            geom.chunk
        );
        let kv_bytes = geom.kv_heads * geom.bucket * geom.head_dim * 4;
        Ok(Self {
            runner,
            geom,
            seeder: None,
            context_length,
            rebuild: None,
            state: DecodeState {
                rope_theta,
                past_k: vec![vec![0u8; kv_bytes]; geom.layers],
                past_v: vec![vec![0u8; kv_bytes]; geom.layers],
                cur_len: 0,
                tokens: Vec::new(),
                steps: 0,
            },
        })
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
    pub fn rewind_to(&mut self, len: usize) {
        let len = len.min(self.state.cur_len);
        self.state.cur_len = len;
        self.state.tokens.truncate(len);
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
        let Some(rebuild) = self.rebuild.as_mut() else {
            bail!(
                "decode bucket ({}) exhausted and the session has no rebuild source",
                self.geom.bucket
            );
        };
        let new_bucket = ((self.geom.bucket as u64) * 2).min(self.context_length);
        ensure!(
            new_bucket > self.geom.bucket as u64,
            "decode bucket cannot grow past the model's context ({})",
            self.context_length
        );
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

        let row = self.geom.head_dim * 4;
        let (old_b, new_b) = (self.geom.bucket, geom.bucket);
        let realized = self.state.cur_len;
        let widen = |buffers: &mut Vec<Vec<u8>>, kv: usize| {
            for buf in buffers.iter_mut() {
                let mut wide = vec![0u8; kv * new_b * row];
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
        self.seeder = None;
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

    /// One decode step: feed `token` at the next absolute position, splice
    /// the step's K/V rows into the past buffers, and return the logit row.
    pub fn step(&mut self, token: i64) -> Result<Vec<f32>> {
        self.ensure_capacity(1)?;
        self.state.pass(&mut self.runner, self.geom, &[token])
    }

    /// Feed `tokens` at the next absolute positions and return the logit row
    /// at the LAST token (the sampler's row). Uses the chunked seeder when
    /// installed — ceil(n/chunk) passes, the final partial chunk padded —
    /// else one step per token.
    pub fn feed(&mut self, tokens: &[i64]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "feed needs at least one token");
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
            row = Some(match self.seeder.as_mut() {
                Some((seeder, sgeom)) if now.len() >= 2 => self.state.pass(seeder, *sgeom, now)?,
                _ => {
                    // One position at a time on the step plan.
                    let mut last = None;
                    for &t in now {
                        last = Some(self.state.pass(&mut self.runner, self.geom, &[t])?);
                    }
                    last.expect("now is non-empty")
                }
            });
            rest = later;
        }
        row.context("feed processed no tokens")
    }
}
