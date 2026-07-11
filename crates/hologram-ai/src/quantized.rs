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
        DType::F16 => {
            // IEEE-754 half: the correct decode (subnormals, inf/NaN) via `half`,
            // not a bespoke bit shuffle. F16 is a first-class safetensors dtype —
            // an int8 tier that only knew F32/BF16 rejected every F16 checkpoint.
            if bytes.len() != elems * 2 {
                bail!(
                    "F16 weight is {} bytes, expected {}",
                    bytes.len(),
                    elems * 2
                );
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        other => bail!("quantized derivation from {other:?} weights is not defined"),
    }
}

/// Derive the quantized artifact of a wide `[out, in]` weight: encode
/// per-channel symmetric int8, layout `q ‖ scales`. Deterministic — same wide
/// bytes, same artifact bytes, same κ, on every host.
///
/// The `q` block's byte order is decided by
/// [`hologram_ai_common::lower::omajor_w8a8_servable`] — the *same* predicate the
/// binder consults to declare `weight_layout` on the `Dequantize` node, so the
/// bytes and the declaration cannot drift. There is no third state: `[n,k]` bytes
/// read as `[k,n]` is a plausible wrong answer, not a slow one.
///
/// * **servable** ⇒ output-major `[out, in]`. That is the wide tensor's *own*
///   order, so this costs no transpose — it removes the one that used to be
///   here, along with the `out·in` f32 scratch buffer it needed. Under a 4 GiB
///   wasm ceiling, not allocating a second copy of a 300 MB projection is worth
///   more than the transpose's cycles.
/// * **not servable** ⇒ row-major `[in, out]`, the matmul orientation the scalar
///   W8A32 dequant loop reads. The transpose is paid only here.
///
/// Either way `q.len() == out·in` and the scales follow at that offset, so the
/// ranged κ bindings (`external_range(κ, .., 0, elems)` and
/// `external_range(κ, .., elems, out*4)`) are layout-independent.
///
/// The scales are identical under both orders, and the codes are transposes of
/// one another (`scales_omajor_and_rowmajor_agree_and_the_codes_transpose`): the
/// codec is the same, only the storage order differs, and the accumulation
/// `Σ aᵢ·d(cᵢ)` cannot tell them apart.
pub fn derive_quantized_artifact(
    wide: &[u8],
    dtype: DType,
    out_features: u64,
    in_features: u64,
) -> Result<Vec<u8>> {
    derive_quantized_artifact_tier(wide, dtype, QuantTier::Int8, out_features, in_features)
}

pub use hologram_ai_common::lower::QuantTier;

/// Tier-parametric artifact derivation. `target` selects int8 or int4; the layout
/// decision (`omajor_w8a8_servable` for the tier's dtype) is shared, so the bytes
/// and the binder's `weight_layout` declaration cannot drift for either tier.
/// Layout `q ‖ scales_f32_le(4·out)`, with `q` the tier-packed codes.
pub fn derive_quantized_artifact_tier(
    wide: &[u8],
    dtype: DType,
    target: QuantTier,
    out_features: u64,
    in_features: u64,
) -> Result<Vec<u8>> {
    let (rows, cols) = (out_features as usize, in_features as usize);
    let w = widen_to_f32(wide, dtype, rows * cols)?;
    // Same predicate the binder consults for `weight_layout`, per the tier's
    // dtype — so the artifact bytes and the declaration cannot drift. Servable ⇒
    // encode in the wide tensor's own output-major `[out, in]` order (no
    // transpose); otherwise transpose to row-major `[in, out]` first.
    let servable = hologram_ai_common::lower::omajor_w8a8_servable(target.dtype_tag(), cols, rows);

    // The tier-packed weight block (int8: one byte/code; int4: packed nibbles),
    // then the per-channel f32 scales appended at `q.len()`.
    let (mut artifact, scales) = match target {
        QuantTier::Int8 if servable => {
            let (q, scales) = hologram_ai_quant::encode_int8_per_channel_omajor(&w, rows, cols);
            (bytemuck::cast_slice::<i8, u8>(&q).to_vec(), scales)
        }
        QuantTier::Int8 => {
            let wt = transpose_omajor_to_rowmajor(&w, rows, cols);
            let (q, scales) = hologram_ai_quant::encode_int8_per_channel(&wt, cols, rows);
            (bytemuck::cast_slice::<i8, u8>(&q).to_vec(), scales)
        }
        QuantTier::Int4 if servable => {
            hologram_ai_quant::encode_int4_per_channel_omajor(&w, rows, cols)
        }
        QuantTier::Int4 => {
            let wt = transpose_omajor_to_rowmajor(&w, rows, cols);
            hologram_ai_quant::encode_int4_per_channel(&wt, cols, rows)
        }
    };
    append_scales(&mut artifact, &scales);
    Ok(artifact)
}

/// Transpose an output-major `[out, in]` weight to row-major `[in, out]` — the
/// orientation the non-servable scalar dequant loop reads.
fn transpose_omajor_to_rowmajor(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut wt = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            wt[c * rows + r] = w[r * cols + c];
        }
    }
    wt
}

/// Append the per-channel f32 scales (little-endian) after the `q` block.
fn append_scales(artifact: &mut Vec<u8>, scales: &[f32]) {
    artifact.reserve(scales.len() * 4);
    for &s in scales {
        artifact.extend_from_slice(&s.to_le_bytes());
    }
}

/// Derive and persist the quantized artifact for `wide_kappa` at `target` tier,
/// returning its [`QuantMap`] entry `(artifact κ, out_features, in_features,
/// tier)`. The wide content is resolved through the store; the artifact enters
/// the store as ordinary content (its κ is minted by the insert's hash). Evicting
/// the wide blob afterwards is the caller's saturation decision — crystallization
/// makes it gas-phase, the lifecycle evaporates it.
pub fn crystallize_quantized(
    store: &mut DirKappaStore,
    wide_kappa: &str,
    dtype: DType,
    out_features: u64,
    in_features: u64,
) -> Result<(String, u64, u64, QuantTier)> {
    crystallize_quantized_tier(
        store,
        wide_kappa,
        dtype,
        QuantTier::Int8,
        out_features,
        in_features,
    )
}

/// [`crystallize_quantized`] at an explicit `target` tier (int8 / int4).
pub fn crystallize_quantized_tier(
    store: &mut DirKappaStore,
    wide_kappa: &str,
    dtype: DType,
    target: QuantTier,
    out_features: u64,
    in_features: u64,
) -> Result<(String, u64, u64, QuantTier)> {
    let wide = store
        .resolve(wide_kappa)
        .with_context(|| format!("resolving wide κ `{wide_kappa}` for quantized derivation"))?;
    let artifact = derive_quantized_artifact_tier(&wide, dtype, target, out_features, in_features)?;
    let kappa = store.insert(&artifact)?;
    Ok((kappa, out_features, in_features, target))
}

/// Crystallize the artifact of a **head chunk** at `target` tier: a byte range
/// `[offset, offset+len)` of the wide LM-head tensor `wide_kappa` (a tied
/// head's is the embedding table), covering `out_features` vocab rows of
/// `in_features` hidden. Resolves only the chunk's slice through the store
/// (sub-tensor κ-resolution — the tied embedding never re-transits whole),
/// derives its matmul-ready per-channel form, and persists it. Returns
/// the artifact's [`QuantMap`] entry `(artifact κ, out_features, in_features,
/// tier)`; the caller keys it by [`hologram_ai_common::lower::quant_key`]`(wide_kappa,
/// Some((offset, len)))`. The wide κ is NOT evaporated — the embedding Gather
/// still binds it.
#[allow(clippy::too_many_arguments)]
pub fn crystallize_quantized_range(
    store: &mut DirKappaStore,
    wide_kappa: &str,
    offset: u64,
    len: u64,
    dtype: DType,
    out_features: u64,
    in_features: u64,
) -> Result<(String, u64, u64, QuantTier)> {
    crystallize_quantized_range_tier(
        store,
        wide_kappa,
        offset,
        len,
        dtype,
        QuantTier::Int8,
        out_features,
        in_features,
    )
}

/// [`crystallize_quantized_range`] at an explicit `target` tier (int8 / int4).
#[allow(clippy::too_many_arguments)]
pub fn crystallize_quantized_range_tier(
    store: &mut DirKappaStore,
    wide_kappa: &str,
    offset: u64,
    len: u64,
    dtype: DType,
    target: QuantTier,
    out_features: u64,
    in_features: u64,
) -> Result<(String, u64, u64, QuantTier)> {
    let slice = store
        .resolve_range(wide_kappa, offset, len)
        .with_context(|| {
            format!(
                "resolving head-chunk range [{offset}, {offset}+{len}) of wide κ `{wide_kappa}`"
            )
        })?;
    let artifact =
        derive_quantized_artifact_tier(&slice, dtype, target, out_features, in_features)?;
    let kappa = store.insert(&artifact)?;
    Ok((kappa, out_features, in_features, target))
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
    fn f16_widens_to_the_same_artifact_as_its_exact_f32() {
        // F16 is a first-class safetensors dtype; the int8 tier must derive from
        // it. Values exactly representable in f16 (multiples of 0.5) decode
        // losslessly, so the F16 artifact must equal the artifact derived from
        // the identical f32 image — bit for bit. Proves the widening, not just
        // that it runs.
        let (out, inf) = (3u64, 4u64);
        let vals: Vec<f32> = (0..out * inf).map(|k| (k as f32) * 0.5 - 2.0).collect();
        let f16_bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a16 = derive_quantized_artifact(&f16_bytes, DType::F16, out, inf).unwrap();
        let a32 = derive_quantized_artifact(&f32_bytes, DType::F32, out, inf).unwrap();
        assert_eq!(
            a16, a32,
            "F16 widening must reproduce the exact-f32 artifact for f16-representable values"
        );
    }

    #[test]
    fn int4_artifact_is_half_the_weight_bytes_and_deterministic() {
        // int4 packs the weight block to nibbles: q is out·in/2 bytes (vs out·in
        // for int8), + the same out·4 scale bytes. This is the bandwidth/size
        // lever — the artifact is strictly smaller than the int8 one.
        let (out, inf) = (4u64, 8u64);
        let wide: Vec<u8> = (0..out * inf)
            .flat_map(|k| (((k % 5) as f32) * 0.4 - 1.0).to_le_bytes())
            .collect();
        let a4 =
            derive_quantized_artifact_tier(&wide, DType::F32, QuantTier::Int4, out, inf).unwrap();
        let a8 =
            derive_quantized_artifact_tier(&wide, DType::F32, QuantTier::Int8, out, inf).unwrap();
        assert_eq!(
            a4.len() as u64,
            out * inf / 2 + out * 4,
            "int4 q block is nibbles"
        );
        assert_eq!(
            a8.len() as u64,
            out * inf + out * 4,
            "int8 q block is bytes"
        );
        assert!(a4.len() < a8.len(), "int4 artifact is strictly smaller");
        let b4 =
            derive_quantized_artifact_tier(&wide, DType::F32, QuantTier::Int4, out, inf).unwrap();
        assert_eq!(a4, b4, "int4 derivation must be bit-deterministic");
    }

    #[test]
    fn int4_artifact_dequantizes_close_to_the_wide_weights() {
        // The int4 artifact, decoded through the substrate's I4_VALUES grid +
        // per-channel scales, reproduces the wide weights within a quant step —
        // the property the quality gate rests on. Dims chosen so the encode is
        // output-major servable (the browser decode path).
        let (out, inf) = (6u64, 16u64);
        let w: Vec<f32> = (0..out * inf)
            .map(|k| ((k as f32) * 0.13).sin() * 2.0)
            .collect();
        let wide: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let art =
            derive_quantized_artifact_tier(&wide, DType::F32, QuantTier::Int4, out, inf).unwrap();
        let (o, i) = (out as usize, inf as usize);
        assert!(
            hologram_ai_common::lower::omajor_w8a8_servable(QuantTier::Int4.dtype_tag(), i, o),
            "test dims must be output-major servable"
        );
        // q = out·in/2 nibble bytes (omajor [out, in]); scales = out f32 after it.
        let q = &art[..o * i / 2];
        let scales: Vec<f32> = art[o * i / 2..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(scales.len(), o);
        for j in 0..o {
            for x in 0..i {
                let l = j * i + x; // omajor: channel j's k-run contiguous
                let byte = q[l >> 1];
                let nib = if l & 1 == 0 { byte & 0x0F } else { byte >> 4 };
                let code = if nib < 8 { nib as i8 } else { nib as i8 - 16 };
                let deq = code as f32 * scales[j];
                let orig = w[j * i + x];
                assert!(
                    (deq - orig).abs() <= scales[j] / 2.0 + 1e-6,
                    "({j},{x}): deq {deq} vs {orig}"
                );
            }
        }
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
