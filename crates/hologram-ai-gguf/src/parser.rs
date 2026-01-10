//! GGUF file parser.

use crate::error::{GgufError, Result};
use crate::metadata::{Architecture, GgufMetadata};
use hologram_ai_common::{WeightDtype, WeightMap, WeightTensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// GGUF magic bytes.
const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];

/// GGUF file version we support.
const SUPPORTED_VERSION: u32 = 3;

/// GGUF quantization types.
/// Names intentionally match GGUF specification (Q4_K, Q8_0, etc.).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit float.
    F32 = 0,
    /// 16-bit float.
    F16 = 1,
    /// 4-bit quantization (legacy).
    Q4_0 = 2,
    /// 4-bit quantization (legacy).
    Q4_1 = 3,
    /// 8-bit quantization.
    Q8_0 = 8,
    /// 8-bit quantization.
    Q8_1 = 9,
    /// K-quant 2-bit.
    Q2_K = 10,
    /// K-quant 3-bit.
    Q3_K = 11,
    /// K-quant 4-bit (most common).
    Q4_K = 12,
    /// K-quant 5-bit.
    Q5_K = 13,
    /// K-quant 6-bit.
    Q6_K = 14,
    /// BFloat16.
    BF16 = 30,
}

impl GgmlType {
    /// Parse from u32.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Get the block size for this type.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::BF16 => 1,
            Self::Q4_0 => 32,
            Self::Q4_1 => 32,
            Self::Q8_0 => 32,
            Self::Q8_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
        }
    }

    /// Get bytes per block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::Q4_0 => 18, // 32 * 4 bits / 8 + 2 (scale)
            Self::Q4_1 => 20, // 32 * 4 bits / 8 + 4 (scale + min)
            Self::Q8_0 => 34, // 32 * 8 bits / 8 + 2 (scale)
            Self::Q8_1 => 36, // 32 * 8 bits / 8 + 4 (scale + sum)
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }
}

/// Tensor information from GGUF header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape dimensions.
    pub dims: Vec<u64>,
    /// Data type.
    pub dtype: GgmlType,
    /// Offset in file.
    pub offset: u64,
}

/// GGUF file parser.
pub struct GgufParser {
    reader: BufReader<File>,
    metadata_kv: HashMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    data_offset: u64,
}

/// Metadata value types.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// String.
    String(String),
    /// Array of values.
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    /// Get as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::U64(v) => Some(*v as u32),
            Self::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Get as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl GgufParser {
    /// Open a GGUF file for parsing.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic);
        }

        // Read version
        let version = read_u32(&mut reader)?;
        if version != SUPPORTED_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Read counts
        let n_tensors = read_u64(&mut reader)?;
        let n_kv = read_u64(&mut reader)?;

        // Read metadata key-value pairs
        let mut metadata_kv = HashMap::new();
        for _ in 0..n_kv {
            let (key, value) = read_kv(&mut reader)?;
            metadata_kv.insert(key, value);
        }

        // Read tensor infos
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let info = read_tensor_info(&mut reader)?;
            tensors.push(info);
        }

        // Calculate data offset (aligned to 32 bytes)
        let current_pos = reader.stream_position()?;
        let data_offset = (current_pos + 31) & !31;

        Ok(Self {
            reader,
            metadata_kv,
            tensors,
            data_offset,
        })
    }

    /// Get the metadata.
    pub fn metadata(&self) -> Result<GgufMetadata> {
        let arch_key = self.find_key("general.architecture")?;
        let arch_str = self
            .metadata_kv
            .get(&arch_key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| GgufError::MissingMetadata("general.architecture".to_string()))?;

        let architecture = Architecture::parse(arch_str);
        let arch_prefix = arch_str.to_lowercase();

        // Helper to get required u32 metadata
        let get_u32 = |suffix: &str| -> Result<u32> {
            let key = format!("{}.{}", arch_prefix, suffix);
            self.metadata_kv
                .get(&key)
                .and_then(|v| v.as_u32())
                .ok_or(GgufError::MissingMetadata(key))
        };

        // Helper to get optional f32 metadata
        let get_f32_opt = |suffix: &str, default: f32| -> f32 {
            let key = format!("{}.{}", arch_prefix, suffix);
            self.metadata_kv
                .get(&key)
                .and_then(|v| v.as_f32())
                .unwrap_or(default)
        };

        let block_count = get_u32("block_count")?;
        let embedding_length = get_u32("embedding_length")?;
        let attention_head_count = get_u32("attention.head_count")?;
        let attention_head_count_kv =
            get_u32("attention.head_count_kv").unwrap_or(attention_head_count);
        let feed_forward_length = get_u32("feed_forward_length")?;
        let context_length = get_u32("context_length").unwrap_or(4096);

        // Get vocab size from tokenizer metadata
        let vocab_size = self
            .metadata_kv
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                MetadataValue::Array(arr) => Some(arr.len() as u32),
                _ => None,
            })
            .unwrap_or(32000);

        let rope_freq_base = get_f32_opt("rope.freq_base", 10000.0);
        let rms_norm_eps = get_f32_opt("attention.layer_norm_rms_epsilon", 1e-6);

        Ok(GgufMetadata {
            architecture,
            block_count,
            embedding_length,
            attention_head_count,
            attention_head_count_kv,
            feed_forward_length,
            rope_freq_base,
            context_length,
            vocab_size,
            rms_norm_eps,
        })
    }

    /// Find a metadata key (handles architecture prefix).
    fn find_key(&self, key: &str) -> Result<String> {
        if self.metadata_kv.contains_key(key) {
            return Ok(key.to_string());
        }
        Err(GgufError::MissingMetadata(key.to_string()))
    }

    /// Load all weights, optionally dequantizing to F32.
    pub fn load_weights(&mut self, dequantize: bool) -> Result<WeightMap> {
        let mut weight_map = WeightMap::new();

        for tensor_info in self.tensors.clone() {
            let tensor = self.load_tensor(&tensor_info, dequantize)?;
            weight_map.insert(tensor_info.name.clone(), tensor);
        }

        Ok(weight_map)
    }

    /// Load a single tensor.
    fn load_tensor(&mut self, info: &TensorInfo, dequantize: bool) -> Result<WeightTensor> {
        // Seek to tensor data
        let tensor_offset = self.data_offset + info.offset;
        self.reader.seek(SeekFrom::Start(tensor_offset))?;

        // Calculate number of elements and bytes
        let n_elements: usize = info.dims.iter().map(|d| *d as usize).product();
        let n_blocks = (n_elements + info.dtype.block_size() - 1).div_ceil(info.dtype.block_size());
        let n_bytes = n_blocks * info.dtype.bytes_per_block();

        // Read raw data
        let mut raw_data = vec![0u8; n_bytes];
        self.reader.read_exact(&mut raw_data)?;

        // Convert shape to usize vec
        let shape: Vec<usize> = info.dims.iter().map(|d| *d as usize).collect();

        if dequantize {
            // Dequantize to F32
            let f32_data = crate::dequant::dequantize(&raw_data, info.dtype, n_elements)?;
            Ok(WeightTensor::from_f32(f32_data, shape))
        } else {
            // Return raw data with appropriate dtype
            let dtype = match info.dtype {
                GgmlType::F32 => WeightDtype::F32,
                GgmlType::F16 => WeightDtype::F16,
                GgmlType::BF16 => WeightDtype::BF16,
                _ => {
                    // For quantized types, we must dequantize
                    let f32_data = crate::dequant::dequantize(&raw_data, info.dtype, n_elements)?;
                    return Ok(WeightTensor::from_f32(f32_data, shape));
                }
            };
            Ok(WeightTensor {
                data: raw_data,
                shape,
                dtype,
            })
        }
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }
}

// Helper functions for reading GGUF binary format

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|_| GgufError::InvalidMetadata {
        key: "string".to_string(),
        message: "Invalid UTF-8".to_string(),
    })
}

fn read_kv<R: Read>(reader: &mut R) -> Result<(String, MetadataValue)> {
    let key = read_string(reader)?;
    let value_type = read_u32(reader)?;
    let value = read_value(reader, value_type)?;
    Ok((key, value))
}

fn read_value<R: Read>(reader: &mut R, value_type: u32) -> Result<MetadataValue> {
    match value_type {
        0 => Ok(MetadataValue::U8({
            let mut b = [0u8; 1];
            reader.read_exact(&mut b)?;
            b[0]
        })),
        1 => Ok(MetadataValue::I8({
            let mut b = [0u8; 1];
            reader.read_exact(&mut b)?;
            b[0] as i8
        })),
        2 => Ok(MetadataValue::U16({
            let mut b = [0u8; 2];
            reader.read_exact(&mut b)?;
            u16::from_le_bytes(b)
        })),
        3 => Ok(MetadataValue::I16({
            let mut b = [0u8; 2];
            reader.read_exact(&mut b)?;
            i16::from_le_bytes(b)
        })),
        4 => Ok(MetadataValue::U32(read_u32(reader)?)),
        5 => Ok(MetadataValue::I32(read_i32(reader)?)),
        6 => Ok(MetadataValue::F32(read_f32(reader)?)),
        7 => Ok(MetadataValue::Bool({
            let mut b = [0u8; 1];
            reader.read_exact(&mut b)?;
            b[0] != 0
        })),
        8 => Ok(MetadataValue::String(read_string(reader)?)),
        9 => {
            // Array
            let elem_type = read_u32(reader)?;
            let len = read_u64(reader)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(reader, elem_type)?);
            }
            Ok(MetadataValue::Array(arr))
        }
        10 => Ok(MetadataValue::U64(read_u64(reader)?)),
        11 => Ok(MetadataValue::I64({
            let mut b = [0u8; 8];
            reader.read_exact(&mut b)?;
            i64::from_le_bytes(b)
        })),
        12 => Ok(MetadataValue::F64({
            let mut b = [0u8; 8];
            reader.read_exact(&mut b)?;
            f64::from_le_bytes(b)
        })),
        _ => Err(GgufError::InvalidMetadata {
            key: "value_type".to_string(),
            message: format!("Unknown value type: {}", value_type),
        }),
    }
}

fn read_tensor_info<R: Read>(reader: &mut R) -> Result<TensorInfo> {
    let name = read_string(reader)?;
    let n_dims = read_u32(reader)?;

    let mut dims = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        dims.push(read_u64(reader)?);
    }

    let dtype_u32 = read_u32(reader)?;
    let dtype = GgmlType::from_u32(dtype_u32)
        .ok_or_else(|| GgufError::UnsupportedQuantization(format!("type {}", dtype_u32)))?;

    let offset = read_u64(reader)?;

    Ok(TensorInfo {
        name,
        n_dims,
        dims,
        dtype,
        offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_block_size() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::F16.block_size(), 1);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q8_0.block_size(), 32);
        assert_eq!(GgmlType::Q4_K.block_size(), 256);
    }

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GgmlType::from_u32(0), Some(GgmlType::F32));
        assert_eq!(GgmlType::from_u32(1), Some(GgmlType::F16));
        assert_eq!(GgmlType::from_u32(12), Some(GgmlType::Q4_K));
        assert_eq!(GgmlType::from_u32(999), None);
    }

    #[test]
    fn test_metadata_value_conversions() {
        let u32_val = MetadataValue::U32(42);
        assert_eq!(u32_val.as_u32(), Some(42));

        let f32_val = MetadataValue::F32(std::f32::consts::PI);
        assert!((f32_val.as_f32().unwrap() - std::f32::consts::PI).abs() < 0.01);

        let str_val = MetadataValue::String("test".to_string());
        assert_eq!(str_val.as_str(), Some("test"));
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo {
            name: "test.weight".to_string(),
            n_dims: 2,
            dims: vec![100, 200],
            dtype: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(info.name, "test.weight");
        assert_eq!(info.dims.len(), 2);
    }
}
