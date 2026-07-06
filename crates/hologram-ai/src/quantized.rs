//! Quantized derived artifacts — dictionary row `quantized-transit`.
//!
//! A quantized weight form is DERIVED CONTENT (`derived-artifact-kappa`
//! applied to weights): computed deterministically from a wide tensor's κ,
//! addressed by its own κ, persisted in the κ-store like any content. Once
//! it crystallizes, the wide blob goes gas-phase — stage graphs bind the
//! artifact's κ (two ranges: the i8 block and the per-channel f32 scales —
//! the artifact is a term over ranges, sub-tensor κ-resolution), so the wide
//! form never re-transits and never re-materializes. Recovery is inherited:
//! a corrupted artifact evaporates and re-derives from the wide κ through
//! recorded provenance, fail-closed at every step.
//!
//! The artifact is stored MATMUL-READY: the wide `[out, in]` projection is
//! transposed to `[in, out]` at derivation and encoded per-channel
//! ([`hologram_ai_quant::encode_int8_per_channel`], one scale per output
//! column), layout `q_i8(in·out) ‖ scales_f32_le(4·out)`. The compile-time
//! transpose node is retired with the wide binding: `Dequantize{axis:1}`
//! feeds the matmul directly — the adjacency the substrate fuses in-register
//! (architecture class QZ), the same shape the inline int8 pass emits.
//! Quantization is a semantic tier, never silent: the quantized form is a
//! DIFFERENT model whose quality is measured, and the pipeline states which
//! tier it runs.

use anyhow::{bail, Context, Result};
use hologram_ai_common::DType;

use crate::materialize::{DirKappaStore, KappaStore};

pub use hologram_ai_common::lower::QuantMap;

/// Decode wide weight bytes (`F32` or `BF16`, little-endian) to f32.
fn widen_to_f32(bytes: &[u8], dtype: DType, elems: usize) -> Result<Vec<f32>> {
    match dtype {
        DType::F32 => {
            if bytes.len() != elems * 4 {
                bail!(
                    "F32 weight is {} bytes, expected {}",
                    bytes.len(),
                    elems * 4
                );
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        DType::BF16 => {
            if bytes.len() != elems * 2 {
                bail!(
                    "BF16 weight is {} bytes, expected {}",
                    bytes.len(),
                    elems * 2
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect())
        }
        other => bail!("quantized derivation from {other:?} weights is not defined"),
    }
}

/// Derive the quantized artifact of a wide `[out, in]` weight: transpose to
/// the matmul orientation `[in, out]`, encode per-channel symmetric int8,
/// layout `q ‖ scales`. Deterministic — same wide bytes, same artifact
/// bytes, same κ, on every host.
pub fn derive_quantized_artifact(
    wide: &[u8],
    dtype: DType,
    out_features: u64,
    in_features: u64,
) -> Result<Vec<u8>> {
    let (rows, cols) = (out_features as usize, in_features as usize);
    let w = widen_to_f32(wide, dtype, rows * cols)?;
    let mut wt = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            wt[c * rows + r] = w[r * cols + c];
        }
    }
    let (q, scales) = hologram_ai_quant::encode_int8_per_channel(&wt, cols, rows);
    let mut artifact = Vec::with_capacity(q.len() + scales.len() * 4);
    artifact.extend_from_slice(bytemuck::cast_slice::<i8, u8>(&q));
    for s in scales {
        artifact.extend_from_slice(&s.to_le_bytes());
    }
    Ok(artifact)
}

/// Derive and persist the quantized artifact for `wide_kappa`, returning its
/// [`QuantMap`] entry `(artifact κ, out_features, in_features)`. The wide
/// content is resolved through the store; the artifact enters the store as
/// ordinary content (its κ is minted by the insert's hash). Evicting the
/// wide blob afterwards is the caller's saturation decision —
/// crystallization makes it gas-phase, the lifecycle evaporates it.
pub fn crystallize_quantized(
    store: &mut DirKappaStore,
    wide_kappa: &str,
    dtype: DType,
    out_features: u64,
    in_features: u64,
) -> Result<(String, u64, u64)> {
    let wide = store
        .resolve(wide_kappa)
        .with_context(|| format!("resolving wide κ `{wide_kappa}` for quantized derivation"))?;
    let artifact = derive_quantized_artifact(&wide, dtype, out_features, in_features)?;
    let kappa = store.insert(&artifact)?;
    Ok((kappa, out_features, in_features))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivation_is_deterministic_and_smaller() {
        let (out, inf) = (8u64, 16u64);
        let wide: Vec<u8> = (0..out * inf)
            .flat_map(|k| (((k % 7) as f32) * 0.3 - 1.0).to_le_bytes())
            .collect();
        let a = derive_quantized_artifact(&wide, DType::F32, out, inf).unwrap();
        let b = derive_quantized_artifact(&wide, DType::F32, out, inf).unwrap();
        assert_eq!(a, b, "derivation must be bit-deterministic");
        assert_eq!(a.len() as u64, out * inf + out * 4);
        assert!(a.len() < wide.len(), "the artifact is strictly smaller");
    }

    #[test]
    fn artifact_is_the_transposed_encoding() {
        // Wide [2, 3] with distinct values: transposition is observable.
        let w: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let wide: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a = derive_quantized_artifact(&wide, DType::F32, 2, 3).unwrap();
        // Transposed [3, 2] = [[1,4],[2,5],[3,6]]; per-column scales over
        // out=2 columns: max|col0|=3 → 3/127, max|col1|=6 → 6/127.
        let q = &a[..6];
        assert_eq!(q[0] as i8, 42); // 1.0 / (3/127) = 42.33 → 42
        assert_eq!(q[1] as i8, 85); // 4.0 / (6/127) = 84.67 → 85
        let s0 = f32::from_le_bytes([a[6], a[7], a[8], a[9]]);
        assert!((s0 - 3.0 / 127.0).abs() < 1e-7);
    }

    #[test]
    fn bf16_widens_before_encoding() {
        let (out, inf) = (2u64, 4u64);
        // bf16 is the top 16 bits of f32: 1.0 = 0x3F80.
        let wide: Vec<u8> = std::iter::repeat_n(0x3F80u16.to_le_bytes(), (out * inf) as usize)
            .flatten()
            .collect();
        let a = derive_quantized_artifact(&wide, DType::BF16, out, inf).unwrap();
        // All-ones: q = 127 everywhere.
        assert!(a[..(out * inf) as usize].iter().all(|&b| b as i8 == 127));
    }
}
