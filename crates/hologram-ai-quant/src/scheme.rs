/// Storage quantization scheme for a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    None,
    Q4_0,
    Q4_1,
    Q5_0,
    Q8_0,
    Q2K,
    Q4K,
    Q6K,
    IQ4Xs,
}

/// Dtype used to store scale factors within a quantized block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleDtype {
    F16,
    F32,
}

/// Descriptor combining quantization scheme with block geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantDescriptor {
    pub scheme: QuantScheme,
    pub block_size: u32,
    pub scale_dtype: ScaleDtype,
}

impl QuantDescriptor {
    /// Plain (unquantized) descriptor.
    pub fn none() -> Self {
        Self {
            scheme: QuantScheme::None,
            block_size: 1,
            scale_dtype: ScaleDtype::F32,
        }
    }

    /// Q4_0: 32 weights per block, f16 scale, 4-bit packed nibbles.
    pub fn q4_0() -> Self {
        Self {
            scheme: QuantScheme::Q4_0,
            block_size: 32,
            scale_dtype: ScaleDtype::F16,
        }
    }

    /// Q8_0: 32 weights per block, f16 scale, 8-bit signed integers.
    pub fn q8_0() -> Self {
        Self {
            scheme: QuantScheme::Q8_0,
            block_size: 32,
            scale_dtype: ScaleDtype::F16,
        }
    }

    /// Q6_K: 256 weights per super-block, 6-bit quants + 8-bit scales + f16 super-scale.
    pub fn q6_k() -> Self {
        Self {
            scheme: QuantScheme::Q6K,
            block_size: 256,
            scale_dtype: ScaleDtype::F16,
        }
    }
}
