//! Decode-step execution engine (dictionary row `decode-plan`).
//!
//! Drives a compiled decode-step archive — one token in, one logit row out —
//! carrying each layer's K/V rows between steps through the plan's named
//! ports. The carried K/V is derived content moving through ports, not a
//! mutable cache inside the graph: the engine owns fixed `bucket`-row buffers,
//! feeds them as `past_k_l`/`past_v_l` inputs, and splices each step's
//! `k_new_l`/`v_new_l` outputs into row `pos` afterwards. Bucket exhaustion
//! recompiles at a geometrically larger bucket and copies the realized rows —
//! capacity is a recompile, never a ceiling; the model's own trained context
//! is the only semantic bound.
//!
//! Positions are runtime data: the engine synthesizes `rope_cos`/`rope_sin`
//! at the token's absolute position (standard `theta^(-2i/d)` tables, halves
//! duplicated to match the rotate-half kernel) and the additive `decode_mask`
//! that erases unrealized bucket rows inside the softmax.

use anyhow::{bail, ensure, Context, Result};

use crate::engine::LmSession;

/// Geometry recovered from the decode archive's own port shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeGeometry {
    /// Decoder layers (= count of `past_k_*` input ports).
    pub layers: usize,
    /// KV heads per layer (`past_k_l` dim 0).
    pub kv_heads: usize,
    /// Head dim (`past_k_l` dim 2 = `rope_cos` width).
    pub head_dim: usize,
    /// Fixed past-bucket row count (`past_k_l` dim 1).
    pub bucket: usize,
    /// Vocabulary size (`logits` element count — the plan is single-position).
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
            "archive has no past_k_* input ports — not a decode-step plan"
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
        let logits = session
            .output_port_info()
            .into_iter()
            .find(|p| p.name == "logits")
            .context("decode plan lacks a logits output")?;
        Ok(Self {
            layers,
            kv_heads,
            head_dim,
            bucket,
            vocab: logits.element_count,
        })
    }
}

/// Rebuild source for bucket growth: given a bucket size, compile a fresh
/// decode plan. `None` pins the session to its initial bucket (exhaustion
/// then fails loud instead of silently truncating context).
pub type DecodeRebuild<S> = Box<dyn FnMut(u64) -> Result<S>>;

/// A generation session over the decode-step plan — generic over the
/// [`LmSession`] executing each step, so the same engine drives a monolithic
/// decode archive (`HoloRunner`) or the staged decode pipeline
/// (`StagedRunner`).
pub struct DecodeSession<S: LmSession> {
    runner: S,
    geom: DecodeGeometry,
    rope_theta: f32,
    /// The model's trained position ceiling (`context_length` metadata) — the
    /// only semantic bound on generation length.
    context_length: u64,
    rebuild: Option<DecodeRebuild<S>>,
    /// Per-layer past K/V byte buffers, each `kv · bucket · head_dim` f32s.
    past_k: Vec<Vec<u8>>,
    past_v: Vec<Vec<u8>>,
    /// Realized positions (= the next token's absolute position).
    cur_len: usize,
    /// The realized token at each position — the carried K/V's provenance,
    /// so a later sequence can rewind to its common prefix instead of
    /// replaying it (cross-turn K/V retention: the Generation axis's
    /// resident prefix labels held ACROSS turns).
    tokens: Vec<i64>,
    /// Steps executed over this session's lifetime (the retention
    /// instrument: a shared-prefix turn adds only its suffix).
    steps: u64,
}

impl<S: LmSession> DecodeSession<S> {
    /// Open a session over a compiled decode plan. `rope_theta` comes from
    /// the model's own config (the graph consumes rope as runtime data, so
    /// the table generator lives with the engine); `context_length` is the
    /// model's trained ceiling.
    pub fn new(runner: S, rope_theta: f32, context_length: u64) -> Result<Self> {
        let geom = DecodeGeometry::discover(&runner)?;
        let kv_bytes = geom.kv_heads * geom.bucket * geom.head_dim * 4;
        Ok(Self {
            past_k: vec![vec![0u8; kv_bytes]; geom.layers],
            past_v: vec![vec![0u8; kv_bytes]; geom.layers],
            runner,
            geom,
            rope_theta,
            context_length,
            rebuild: None,
            cur_len: 0,
            tokens: Vec::new(),
            steps: 0,
        })
    }

    /// Attach a rebuild source so bucket exhaustion regrows geometrically
    /// instead of failing.
    pub fn with_rebuild(mut self, rebuild: DecodeRebuild<S>) -> Self {
        self.rebuild = Some(rebuild);
        self
    }

    pub fn geometry(&self) -> DecodeGeometry {
        self.geom
    }

    pub fn realized_len(&self) -> usize {
        self.cur_len
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
        let len = len.min(self.cur_len);
        self.cur_len = len;
        self.tokens.truncate(len);
    }

    /// The realized token at each carried position, in order.
    pub fn realized_tokens(&self) -> &[i64] {
        &self.tokens
    }

    /// Steps executed over this session's lifetime (retention instrument).
    pub fn steps_taken(&self) -> u64 {
        self.steps
    }

    /// Kernel-dispatch counters for the last step (perf attribution).
    pub fn last_dispatched(&self) -> u64 {
        self.runner.pass_dispatched()
    }

    pub fn last_skipped(&self) -> u64 {
        self.runner.pass_skipped()
    }

    /// Standard non-interleaved RoPE tables at absolute position `pos`:
    /// `angle_i = pos · theta^(-2i/d)`, halves duplicated so `cos[j]`/`sin[j]`
    /// pair with the rotate-half partner `j ± d/2`.
    fn rope_tables(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let d = self.geom.head_dim;
        let half = d / 2;
        let mut cos = vec![0.0f32; d];
        let mut sin = vec![0.0f32; d];
        for i in 0..half {
            let inv_freq = 1.0 / (self.rope_theta as f64).powf(2.0 * i as f64 / d as f64);
            let angle = pos as f64 * inv_freq;
            let (s, c) = (angle.sin() as f32, angle.cos() as f32);
            cos[i] = c;
            cos[i + half] = c;
            sin[i] = s;
            sin[i + half] = s;
        }
        (cos, sin)
    }

    /// Grow the bucket geometrically (clamped to the context ceiling) and
    /// copy every layer's realized rows into the wider buffers.
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
                && geom.layers == self.geom.layers
                && geom.kv_heads == self.geom.kv_heads
                && geom.head_dim == self.geom.head_dim,
            "rebuilt decode archive geometry {:?} does not extend {:?}",
            geom,
            self.geom
        );

        let row = self.geom.head_dim * 4;
        let (old_b, new_b) = (self.geom.bucket, geom.bucket);
        let widen = |buffers: &mut Vec<Vec<u8>>, kv: usize, realized: usize| {
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
        widen(&mut self.past_k, geom.kv_heads, self.cur_len);
        widen(&mut self.past_v, geom.kv_heads, self.cur_len);
        self.runner = runner;
        self.geom = geom;
        Ok(())
    }

    /// One decode step: feed `token` at the next absolute position, splice
    /// the step's K/V rows into the past buffers, and return the logit row.
    pub fn step(&mut self, token: i64) -> Result<Vec<f32>> {
        ensure!(
            (self.cur_len as u64) < self.context_length,
            "the model's trained context ({}) is exhausted",
            self.context_length
        );
        if self.cur_len == self.geom.bucket {
            self.grow()?;
        }
        let pos = self.cur_len;
        let geom = self.geom;

        let ids = token.to_le_bytes();
        let (cos, sin) = self.rope_tables(pos);
        let cos_b: Vec<u8> = cos.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sin_b: Vec<u8> = sin.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut mask = vec![0.0f32; geom.bucket + 1];
        for slot in mask.iter_mut().take(geom.bucket).skip(pos) {
            *slot = -1e9;
        }
        let mask_b: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Bind by port NAME — the ports are the plan's contract.
        let port_info = self.runner.input_port_info();
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(port_info.len());
        for port in &port_info {
            let buf: &[u8] = match port.name.as_str() {
                "input_ids" => &ids,
                "rope_cos" => &cos_b,
                "rope_sin" => &sin_b,
                "decode_mask" => &mask_b,
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

        let outputs = self.runner.execute(&inputs)?;
        let out_ports = self.runner.output_port_info();
        ensure!(
            outputs.len() == out_ports.len(),
            "decode step returned {} outputs for {} ports",
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
                        out.bytes.len() == geom.kv_heads * row,
                        "{name} returned {} bytes, expected {}",
                        out.bytes.len(),
                        geom.kv_heads * row
                    );
                    // Splice the step's [kv, head_dim] rows into bucket row `pos`.
                    for j in 0..geom.kv_heads {
                        let dst = (j * geom.bucket + pos) * row;
                        target[dst..dst + row].copy_from_slice(&out.bytes[j * row..(j + 1) * row]);
                    }
                }
            }
        }
        self.cur_len += 1;
        self.tokens.push(token);
        self.steps += 1;
        logits.context("decode step produced no logits output")
    }
}
