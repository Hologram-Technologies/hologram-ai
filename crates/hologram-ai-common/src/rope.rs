//! The RoPE frequency law — ONE spec shared by every table generator.
//!
//! Three sites build rotary tables (the decode runtime's per-step rows, the
//! lowering builder's compile-time constants, and any oracle a witness needs);
//! before this module each restated `θ^(-2i/d)` locally, which made the
//! scaled-frequency checkpoints (`rope_scaling`: Llama-3's `llama3`,
//! long-context `yarn`, Phi-3's `longrope`, `linear`/`dynamic` NTK) and
//! partial-rotary checkpoints (`partial_rotary_factor`: Phi-2, GLM) impossible
//! to support without silent drift between sites. [`RopeSpec`] is the single
//! authority: parse once from `config.json`, thread through the
//! `GroupedQueryAttention` op and the decode session, and every table asks the
//! spec.
//!
//! The laws are the reference (HuggingFace `modeling_rope_utils.py`) formulas,
//! computed in f64 like the pre-existing plain law, emitted as f32 tables.
//! `seq_len` is the forward's total realized length (`pos + chunk`): the
//! `dynamic` and `longrope` variants select their frequency set by it, exactly
//! as the reference does per forward — previously cached K keep the rope they
//! were written with, which is the reference semantics under a KV cache.

/// Frequency-scaling law from `config.json`'s `rope_scaling`.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// No `rope_scaling` (or `"type": "default"`): `inv_freq = base^(-2i/r)`.
    None,
    /// `"linear"`: every frequency divided by `factor` (position interpolation).
    Linear { factor: f64 },
    /// `"dynamic"` NTK: beyond the pretrained length the base itself grows
    /// with the realized length: `base·((factor·L/orig) − (factor−1))^(r/(r−2))`.
    Dynamic {
        factor: f64,
        original_max_position_embeddings: u32,
    },
    /// `"llama3"` (Llama-3.1/3.2/3.3): piecewise by wavelength — high
    /// frequencies kept, low frequencies interpolated by `factor`, a smooth
    /// ramp between the `low/high_freq_factor` wavelength bounds.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: u32,
    },
    /// `"yarn"`: NTK-by-parts — per-dim ramp between interpolation and
    /// extrapolation, plus a global attention temperature on the tables
    /// (`0.1·ln(factor)+1` unless the config names one).
    Yarn {
        factor: f64,
        original_max_position_embeddings: u32,
        beta_fast: f64,
        beta_slow: f64,
        attention_factor: Option<f64>,
    },
    /// `"longrope"`/`"su"` (Phi-3 long-context): per-dim frequency divisors,
    /// the `short_factor` set within the pretrained length and `long_factor`
    /// beyond it, plus an attention temperature
    /// (`√(1+ln(factor)/ln(orig))` unless the config names one).
    LongRope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        original_max_position_embeddings: u32,
        max_position_embeddings: u32,
        attention_factor: Option<f64>,
    },
}

/// The complete rotary law for one model: base frequency, how many head dims
/// rotate, and the frequency scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    /// `rope_theta`.
    pub base: f32,
    /// Rotated dims per head — `None` = the full head_dim; `Some(r)` =
    /// `partial_rotary_factor` resolved against head_dim (Phi-2 style: dims
    /// `r..head_dim` pass through unrotated).
    pub rotary_dim: Option<u32>,
    pub scaling: RopeScaling,
}

impl RopeScaling {
    /// True for the laws whose frequencies depend on the forward's realized
    /// length (`dynamic`, `longrope`). The decode path realizes them exactly
    /// (rows are synthesized per step at the realized length); a plan that
    /// bakes compile-time tables at a PADDED length cannot, and must refuse
    /// rather than encode the wrong length.
    pub fn length_dependent(&self) -> bool {
        matches!(
            self,
            RopeScaling::Dynamic { .. } | RopeScaling::LongRope { .. }
        )
    }
}

impl RopeSpec {
    /// The unscaled full-rotary law — what every call site meant before
    /// `rope_scaling` support.
    pub fn plain(base: f32) -> Self {
        RopeSpec {
            base,
            rotary_dim: None,
            scaling: RopeScaling::None,
        }
    }

    /// Rotated dims for a concrete head_dim.
    pub fn rotary_dim(&self, head_dim: usize) -> usize {
        self.rotary_dim.map(|r| r as usize).unwrap_or(head_dim)
    }

    /// Everything `rows`/`inv_freqs` assumes, checked once at session/plan
    /// construction so the law methods can stay infallible.
    pub fn validate(&self, head_dim: usize) -> Result<(), String> {
        if !self.base.is_finite() || self.base <= 0.0 {
            return Err(format!("rope base must be positive, got {}", self.base));
        }
        let r = self.rotary_dim(head_dim);
        if r == 0 || !r.is_multiple_of(2) || r > head_dim {
            return Err(format!(
                "rotary_dim must be even and in 1..=head_dim ({head_dim}), got {r}"
            ));
        }
        match &self.scaling {
            RopeScaling::None => {}
            RopeScaling::Linear { factor } | RopeScaling::Dynamic { factor, .. } => {
                if !factor.is_finite() || *factor < 1.0 {
                    return Err(format!("rope_scaling factor must be ≥ 1, got {factor}"));
                }
            }
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                if !factor.is_finite() || *factor < 1.0 || *original_max_position_embeddings == 0 {
                    return Err(format!(
                        "llama3 rope_scaling needs factor ≥ 1 and a nonzero \
                         original_max_position_embeddings, got factor {factor}, \
                         original {original_max_position_embeddings}"
                    ));
                }
                if !(high_freq_factor.is_finite() && low_freq_factor.is_finite())
                    || high_freq_factor <= low_freq_factor
                {
                    return Err(format!(
                        "llama3 rope_scaling needs high_freq_factor > low_freq_factor, \
                         got low {low_freq_factor}, high {high_freq_factor}"
                    ));
                }
            }
            RopeScaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                ..
            } => {
                if !factor.is_finite() || *factor < 1.0 || *original_max_position_embeddings == 0 {
                    return Err(format!(
                        "yarn rope_scaling needs factor ≥ 1 and a nonzero \
                         original_max_position_embeddings, got factor {factor}, \
                         original {original_max_position_embeddings}"
                    ));
                }
                if !(beta_fast.is_finite() && beta_slow.is_finite()) || beta_fast <= beta_slow {
                    return Err(format!(
                        "yarn rope_scaling needs beta_fast > beta_slow, \
                         got fast {beta_fast}, slow {beta_slow}"
                    ));
                }
            }
            RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
                max_position_embeddings,
                ..
            } => {
                if short_factor.len() != r / 2 || long_factor.len() != r / 2 {
                    return Err(format!(
                        "longrope rope_scaling factor arrays must have rotary_dim/2 = {} \
                         entries, got short {} / long {}",
                        r / 2,
                        short_factor.len(),
                        long_factor.len()
                    ));
                }
                if *original_max_position_embeddings == 0
                    || max_position_embeddings < original_max_position_embeddings
                {
                    return Err(format!(
                        "longrope rope_scaling needs 0 < original_max_position_embeddings \
                         ≤ max_position_embeddings, got original \
                         {original_max_position_embeddings}, max {max_position_embeddings}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Per-pair inverse frequencies (length `rotary_dim/2`) for a forward
    /// whose total realized length is `seq_len`.
    pub fn inv_freqs(&self, head_dim: usize, seq_len: usize) -> Vec<f64> {
        let r = self.rotary_dim(head_dim);
        let half = r / 2;
        let base = self.base as f64;
        let plain = |b: f64| -> Vec<f64> {
            (0..half)
                .map(|i| 1.0 / b.powf(2.0 * i as f64 / r as f64))
                .collect()
        };
        match &self.scaling {
            RopeScaling::None => plain(base),
            RopeScaling::Linear { factor } => plain(base).into_iter().map(|f| f / factor).collect(),
            RopeScaling::Dynamic {
                factor,
                original_max_position_embeddings,
            } => {
                let orig = *original_max_position_embeddings as f64;
                let len = (seq_len as f64).max(orig);
                let grown = base
                    * ((factor * len / orig) - (factor - 1.0)).powf(r as f64 / (r as f64 - 2.0));
                plain(grown)
            }
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            } => {
                let orig = *original_max_position_embeddings as f64;
                let low_wavelen = orig / low_freq_factor;
                let high_wavelen = orig / high_freq_factor;
                plain(base)
                    .into_iter()
                    .map(|f| {
                        let wavelen = 2.0 * core::f64::consts::PI / f;
                        if wavelen < high_wavelen {
                            f
                        } else if wavelen > low_wavelen {
                            f / factor
                        } else {
                            let smooth = (orig / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            (1.0 - smooth) * f / factor + smooth * f
                        }
                    })
                    .collect()
            }
            RopeScaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                ..
            } => {
                let orig = *original_max_position_embeddings as f64;
                let dim = r as f64;
                // Dim below which a rotation count is fully extrapolated.
                let correction = |num_rot: f64| -> f64 {
                    dim * (orig / (num_rot * 2.0 * core::f64::consts::PI)).ln() / (2.0 * base.ln())
                };
                let low = correction(*beta_fast).floor().max(0.0);
                let mut high = correction(*beta_slow).ceil().min(dim - 1.0);
                if high <= low {
                    high = low + 0.001;
                }
                plain(base)
                    .into_iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                        let extrapolation = 1.0 - ramp;
                        (f / factor) * (1.0 - extrapolation) + f * extrapolation
                    })
                    .collect()
            }
            RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
                ..
            } => {
                let ext = if seq_len > *original_max_position_embeddings as usize {
                    long_factor
                } else {
                    short_factor
                };
                plain(base)
                    .into_iter()
                    .zip(ext.iter())
                    .map(|(f, e)| f / e)
                    .collect()
            }
        }
    }

    /// Global multiplier on the rotary cos/sin values (the yarn/longrope
    /// attention temperature); 1.0 for the other laws. Pass-through dims of a
    /// partial-rotary head are NEVER scaled — `rows` keeps their cos at
    /// exactly 1.
    pub fn attention_factor(&self, seq_len: usize) -> f32 {
        let _ = seq_len;
        match &self.scaling {
            RopeScaling::Yarn {
                factor,
                attention_factor,
                ..
            } => attention_factor.unwrap_or(0.1 * factor.ln() + 1.0) as f32,
            RopeScaling::LongRope {
                original_max_position_embeddings,
                max_position_embeddings,
                attention_factor,
                ..
            } => {
                if let Some(a) = attention_factor {
                    return *a as f32;
                }
                let factor =
                    *max_position_embeddings as f64 / *original_max_position_embeddings as f64;
                if factor <= 1.0 {
                    1.0
                } else {
                    (1.0 + factor.ln() / (*original_max_position_embeddings as f64).ln()).sqrt()
                        as f32
                }
            }
            _ => 1.0,
        }
    }

    /// Decode-layout rows for absolute positions `pos..pos+chunk`:
    /// `[chunk · head_dim]` full-width, halves duplicated so `cos[j]`/`sin[j]`
    /// pair with the rotate-half partner `j ± rotary_dim/2`, pass-through dims
    /// `rotary_dim..head_dim` held at `cos=1, sin=0` (identity under
    /// `x·cos + rotate_half(x)·sin`).
    pub fn rows(&self, pos: usize, chunk: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let d = head_dim;
        let r = self.rotary_dim(d);
        let half = r / 2;
        let seq_len = pos + chunk;
        let freqs = self.inv_freqs(d, seq_len);
        let scale = self.attention_factor(seq_len);
        let mut cos = vec![0.0f32; chunk * d];
        let mut sin = vec![0.0f32; chunk * d];
        for i in 0..chunk {
            for j in 0..half {
                let angle = (pos + i) as f64 * freqs[j];
                let (s, c) = (angle.sin() as f32 * scale, angle.cos() as f32 * scale);
                cos[i * d + j] = c;
                cos[i * d + j + half] = c;
                sin[i * d + j] = s;
                sin[i * d + j + half] = s;
            }
            for j in r..d {
                cos[i * d + j] = 1.0;
                sin[i * d + j] = 0.0;
            }
        }
        (cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 8;

    fn plain_freqs(base: f64, r: usize) -> Vec<f64> {
        (0..r / 2)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / r as f64))
            .collect()
    }

    #[test]
    fn plain_spec_reproduces_the_preexisting_law() {
        let spec = RopeSpec::plain(10_000.0);
        assert_eq!(spec.inv_freqs(D, 1), plain_freqs(10_000.0, D));
        assert_eq!(spec.attention_factor(1), 1.0);
        let (cos, sin) = spec.rows(3, 1, D);
        for j in 0..D / 2 {
            let angle = 3.0 * plain_freqs(10_000.0, D)[j];
            assert!((cos[j] - angle.cos() as f32).abs() < 1e-7);
            assert!((sin[j] - angle.sin() as f32).abs() < 1e-7);
            assert_eq!(cos[j], cos[j + D / 2], "halves duplicate");
            assert_eq!(sin[j], sin[j + D / 2], "halves duplicate");
        }
    }

    #[test]
    fn linear_scaling_divides_every_frequency() {
        let spec = RopeSpec {
            base: 10_000.0,
            rotary_dim: None,
            scaling: RopeScaling::Linear { factor: 4.0 },
        };
        let plain = plain_freqs(10_000.0, D);
        for (f, p) in spec.inv_freqs(D, 1).iter().zip(plain) {
            assert!((f - p / 4.0).abs() < 1e-15);
        }
    }

    #[test]
    fn dynamic_scaling_grows_the_base_only_beyond_the_pretrained_length() {
        let spec = RopeSpec {
            base: 10_000.0,
            rotary_dim: None,
            scaling: RopeScaling::Dynamic {
                factor: 2.0,
                original_max_position_embeddings: 128,
            },
        };
        // Within the pretrained length: exactly the plain law.
        assert_eq!(spec.inv_freqs(D, 128), plain_freqs(10_000.0, D));
        // Beyond: base grows by ((factor·L/orig) − (factor−1))^(r/(r−2)).
        let grown = 10_000.0f64 * (2.0f64 * 256.0 / 128.0 - 1.0).powf(8.0 / 6.0);
        assert_eq!(spec.inv_freqs(D, 256), plain_freqs(grown, D));
    }

    #[test]
    fn llama3_scaling_is_piecewise_by_wavelength() {
        // base 10000, r=8 → wavelens 2π/f: pick bounds so pair 0 is kept,
        // pair 3 is fully interpolated, and a middle pair is ramped.
        let orig = 8192u32;
        let spec = RopeSpec {
            base: 10_000.0,
            rotary_dim: None,
            scaling: RopeScaling::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: orig,
            },
        };
        let plain = plain_freqs(10_000.0, D);
        let scaled = spec.inv_freqs(D, 1);
        let high_wavelen = orig as f64 / 4.0;
        let low_wavelen = orig as f64 / 1.0;
        for (f, s) in plain.iter().zip(&scaled) {
            let wavelen = 2.0 * std::f64::consts::PI / f;
            if wavelen < high_wavelen {
                assert_eq!(s, f, "high-frequency pair is kept verbatim");
            } else if wavelen > low_wavelen {
                assert!(
                    (s - f / 8.0).abs() < 1e-15,
                    "low-frequency pair interpolates"
                );
            } else {
                let smooth = (orig as f64 / wavelen - 1.0) / (4.0 - 1.0);
                let expect = (1.0 - smooth) * f / 8.0 + smooth * f;
                assert!((s - expect).abs() < 1e-15, "mid pair rides the ramp");
            }
        }
        // The law must actually exercise all three pieces at these bounds.
        assert!(scaled.first() == plain.first());
        assert!(scaled.last() < plain.last());
    }

    #[test]
    fn yarn_ramps_between_interpolation_and_extrapolation_and_scales_attention() {
        let spec = RopeSpec {
            base: 10_000.0,
            rotary_dim: None,
            scaling: RopeScaling::Yarn {
                factor: 4.0,
                original_max_position_embeddings: 2048,
                beta_fast: 32.0,
                beta_slow: 1.0,
                attention_factor: None,
            },
        };
        let plain = plain_freqs(10_000.0, D);
        let scaled = spec.inv_freqs(D, 1);
        // Reference ramp, restated independently.
        let dim = D as f64;
        let base = 10_000.0f64;
        let corr =
            |n: f64| dim * (2048.0 / (n * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
        let low = corr(32.0).floor().max(0.0);
        let mut high = corr(1.0).ceil().min(dim - 1.0);
        if high <= low {
            high = low + 0.001;
        }
        for (i, (f, s)) in plain.iter().zip(&scaled).enumerate() {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let extrapolation = 1.0 - ramp;
            let expect = (f / 4.0) * (1.0 - extrapolation) + f * extrapolation;
            assert!((s - expect).abs() < 1e-15);
        }
        // Default attention temperature: 0.1·ln(4)+1.
        let expect_a = (0.1 * 4.0f64.ln() + 1.0) as f32;
        assert!((spec.attention_factor(1) - expect_a).abs() < 1e-7);
        // The temperature multiplies the emitted tables.
        let (cos, _) = spec.rows(0, 1, D);
        assert!((cos[0] - expect_a).abs() < 1e-6, "cos(0)·a = a");
    }

    #[test]
    fn longrope_switches_factor_sets_at_the_pretrained_boundary() {
        let short: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
        let long: Vec<f64> = vec![2.0, 4.0, 8.0, 16.0];
        let spec = RopeSpec {
            base: 10_000.0,
            rotary_dim: None,
            scaling: RopeScaling::LongRope {
                short_factor: short,
                long_factor: long.clone(),
                original_max_position_embeddings: 4096,
                max_position_embeddings: 131_072,
                attention_factor: None,
            },
        };
        assert_eq!(spec.inv_freqs(D, 4096), plain_freqs(10_000.0, D));
        let beyond = spec.inv_freqs(D, 4097);
        for ((f, e), s) in plain_freqs(10_000.0, D).iter().zip(&long).zip(&beyond) {
            assert!((s - f / e).abs() < 1e-15);
        }
        // Default attention temperature: √(1+ln(32)/ln(4096)).
        let factor = 131_072.0f64 / 4096.0;
        let expect = (1.0 + factor.ln() / 4096.0f64.ln()).sqrt() as f32;
        assert!((spec.attention_factor(1) - expect).abs() < 1e-7);
    }

    #[test]
    fn partial_rotary_rows_hold_passthrough_dims_at_identity() {
        let spec = RopeSpec {
            base: 10_000.0,
            rotary_dim: Some(4),
            scaling: RopeScaling::None,
        };
        let (cos, sin) = spec.rows(7, 2, D);
        for i in 0..2 {
            // Rotated pairs live at j and j+r/2 with the r-dim frequency law.
            let freqs = plain_freqs(10_000.0, 4);
            for j in 0..2 {
                let angle = (7 + i) as f64 * freqs[j];
                assert!((cos[i * D + j] - angle.cos() as f32).abs() < 1e-7);
                assert_eq!(cos[i * D + j], cos[i * D + j + 2], "pair at j ± r/2");
                assert_eq!(sin[i * D + j], sin[i * D + j + 2], "pair at j ± r/2");
            }
            // Pass-through dims r..d: exact identity, never scaled.
            for j in 4..D {
                assert_eq!(cos[i * D + j], 1.0);
                assert_eq!(sin[i * D + j], 0.0);
            }
        }
    }

    #[test]
    fn validate_refuses_the_malformed_specs() {
        let bad_rotary = RopeSpec {
            base: 10_000.0,
            rotary_dim: Some(3),
            scaling: RopeScaling::None,
        };
        assert!(bad_rotary.validate(D).unwrap_err().contains("rotary_dim"));
        let bad_longrope = RopeSpec {
            base: 10_000.0,
            rotary_dim: None,
            scaling: RopeScaling::LongRope {
                short_factor: vec![1.0; 3],
                long_factor: vec![1.0; 4],
                original_max_position_embeddings: 4096,
                max_position_embeddings: 131_072,
                attention_factor: None,
            },
        };
        assert!(bad_longrope
            .validate(D)
            .unwrap_err()
            .contains("rotary_dim/2"));
        let bad_factor = RopeSpec {
            base: 10_000.0,
            rotary_dim: None,
            scaling: RopeScaling::Linear { factor: 0.5 },
        };
        assert!(bad_factor.validate(D).unwrap_err().contains("factor"));
    }
}
