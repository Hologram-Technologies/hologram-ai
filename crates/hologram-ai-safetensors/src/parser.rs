//! SafeTensors file parser.
//!
//! SafeTensors is a simple binary format for storing tensors:
//! - 8 bytes: header size (u64 little-endian)
//! - N bytes: JSON header containing tensor metadata
//! - Remaining: tensor data
//!
//! The JSON header maps tensor names to their dtype, shape, and byte offsets.

use crate::error::{SafeTensorsError, Result};
use hologram_ai_common::{WeightMap, WeightTensor, WeightDtype};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// SafeTensors data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SafeTensorsDtype {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// Brain float 16.
    #[serde(alias = "BF16")]
    Bf16,
    /// 32-bit integer.
    I32,
    /// 64-bit integer.
    I64,
    /// Boolean.
    #[serde(alias = "BOOL")]
    Bool,
}

impl SafeTensorsDtype {
    /// Bytes per element.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Bf16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Bool => 1,
        }
    }
}

/// Tensor metadata from the header.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorMetadata {
    /// Data type.
    pub dtype: SafeTensorsDtype,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data offset range [start, end).
    pub data_offsets: [usize; 2],
}

/// Header metadata (may contain special "__metadata__" key).
#[derive(Debug, Clone, Deserialize)]
#[serde(transparent)]
pub struct SafeTensorsHeader {
    /// Tensor metadata by name.
    #[serde(flatten)]
    pub tensors: HashMap<String, TensorMetadata>,
}

/// SafeTensors file handle.
pub struct SafeTensorsFile {
    file: File,
    header: SafeTensorsHeader,
    data_offset: u64,
}

impl SafeTensorsFile {
    /// Open a SafeTensors file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;

        // Read header size
        let mut size_buf = [0u8; 8];
        file.read_exact(&mut size_buf)?;
        let header_size = u64::from_le_bytes(size_buf);

        // Read header JSON
        let mut header_buf = vec![0u8; header_size as usize];
        file.read_exact(&mut header_buf)?;

        // Parse header, filtering out __metadata__ if present
        let header_str = std::str::from_utf8(&header_buf)
            .map_err(|_| SafeTensorsError::InvalidHeader("Invalid UTF-8".to_string()))?;

        // Parse as generic JSON first to handle __metadata__
        let json: serde_json::Value = serde_json::from_str(header_str)?;
        let mut tensors = HashMap::new();

        if let serde_json::Value::Object(map) = json {
            for (key, value) in map {
                if key == "__metadata__" {
                    continue; // Skip metadata
                }
                let metadata: TensorMetadata = serde_json::from_value(value)?;
                tensors.insert(key, metadata);
            }
        }

        let header = SafeTensorsHeader { tensors };
        let data_offset = 8 + header_size;

        Ok(Self {
            file,
            header,
            data_offset,
        })
    }

    /// Get tensor metadata.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorMetadata> {
        self.header.tensors.get(name)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.keys().map(|s| s.as_str())
    }

    /// Load a tensor by name.
    pub fn load_tensor(&mut self, name: &str, convert_to_f32: bool) -> Result<WeightTensor> {
        let metadata = self.header.tensors.get(name)
            .ok_or_else(|| SafeTensorsError::InvalidTensorData(
                format!("Tensor not found: {}", name)
            ))?
            .clone();

        let [start, end] = metadata.data_offsets;
        let byte_len = end - start;

        // Seek and read
        self.file.seek(SeekFrom::Start(self.data_offset + start as u64))?;
        let mut data = vec![0u8; byte_len];
        self.file.read_exact(&mut data)?;

        let shape = metadata.shape;

        if convert_to_f32 && metadata.dtype != SafeTensorsDtype::F32 {
            let f32_data = self.convert_to_f32(&data, metadata.dtype)?;
            Ok(WeightTensor::from_f32(f32_data, shape))
        } else {
            let dtype = match metadata.dtype {
                SafeTensorsDtype::F32 => WeightDtype::F32,
                SafeTensorsDtype::F16 => WeightDtype::F16,
                SafeTensorsDtype::Bf16 => WeightDtype::BF16,
                _ => {
                    // For integer types, we need to convert
                    let f32_data = self.convert_to_f32(&data, metadata.dtype)?;
                    return Ok(WeightTensor::from_f32(f32_data, shape));
                }
            };
            Ok(WeightTensor {
                data,
                shape,
                dtype,
            })
        }
    }

    /// Convert tensor data to F32.
    fn convert_to_f32(&self, data: &[u8], dtype: SafeTensorsDtype) -> Result<Vec<f32>> {
        match dtype {
            SafeTensorsDtype::F32 => {
                let mut result = Vec::with_capacity(data.len() / 4);
                for chunk in data.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    result.push(f32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            SafeTensorsDtype::F16 => {
                let mut result = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    let bytes: [u8; 2] = chunk.try_into().unwrap();
                    let f16 = half::f16::from_le_bytes(bytes);
                    result.push(f16.to_f32());
                }
                Ok(result)
            }
            SafeTensorsDtype::Bf16 => {
                let mut result = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    let bytes: [u8; 2] = chunk.try_into().unwrap();
                    let bf16 = half::bf16::from_le_bytes(bytes);
                    result.push(bf16.to_f32());
                }
                Ok(result)
            }
            SafeTensorsDtype::I32 => {
                let mut result = Vec::with_capacity(data.len() / 4);
                for chunk in data.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    result.push(i32::from_le_bytes(bytes) as f32);
                }
                Ok(result)
            }
            SafeTensorsDtype::I64 => {
                let mut result = Vec::with_capacity(data.len() / 8);
                for chunk in data.chunks_exact(8) {
                    let bytes: [u8; 8] = chunk.try_into().unwrap();
                    result.push(i64::from_le_bytes(bytes) as f32);
                }
                Ok(result)
            }
            SafeTensorsDtype::Bool => {
                Ok(data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect())
            }
        }
    }
}

/// Parser for SafeTensors model directories.
pub struct SafeTensorsParser {
    files: Vec<SafeTensorsFile>,
    tensor_to_file: HashMap<String, usize>,
}

impl SafeTensorsParser {
    /// Open a model directory containing SafeTensors files.
    pub fn open_dir<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();

        // Find all .safetensors files
        let mut safetensor_paths: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()?.to_str()? == "safetensors" {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        if safetensor_paths.is_empty() {
            return Err(SafeTensorsError::MissingSafeTensors);
        }

        // Sort for consistent ordering (handles sharded models)
        safetensor_paths.sort();

        // Open all files and build tensor index
        let mut files = Vec::new();
        let mut tensor_to_file = HashMap::new();

        for (file_idx, path) in safetensor_paths.iter().enumerate() {
            let file = SafeTensorsFile::open(path)?;

            // Index tensors in this file
            for name in file.tensor_names() {
                tensor_to_file.insert(name.to_string(), file_idx);
            }

            files.push(file);
        }

        Ok(Self {
            files,
            tensor_to_file,
        })
    }

    /// Load all weights.
    pub fn load_all_weights(&mut self, convert_to_f32: bool) -> Result<WeightMap> {
        let mut weight_map = WeightMap::new();

        // Get all tensor names first (to avoid borrow issues)
        let tensor_names: Vec<String> = self.tensor_to_file.keys().cloned().collect();

        for name in tensor_names {
            let file_idx = self.tensor_to_file[&name];
            let tensor = self.files[file_idx].load_tensor(&name, convert_to_f32)?;
            weight_map.insert(name, tensor);
        }

        Ok(weight_map)
    }

    /// Load a specific tensor.
    pub fn load_tensor(&mut self, name: &str, convert_to_f32: bool) -> Result<WeightTensor> {
        let file_idx = *self.tensor_to_file.get(name)
            .ok_or_else(|| SafeTensorsError::InvalidTensorData(
                format!("Tensor not found: {}", name)
            ))?;
        self.files[file_idx].load_tensor(name, convert_to_f32)
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_to_file.keys().map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype_byte_size() {
        assert_eq!(SafeTensorsDtype::F32.byte_size(), 4);
        assert_eq!(SafeTensorsDtype::F16.byte_size(), 2);
        assert_eq!(SafeTensorsDtype::Bf16.byte_size(), 2);
        assert_eq!(SafeTensorsDtype::I32.byte_size(), 4);
        assert_eq!(SafeTensorsDtype::I64.byte_size(), 8);
        assert_eq!(SafeTensorsDtype::Bool.byte_size(), 1);
    }

    #[test]
    fn test_tensor_metadata_deserialize() {
        let json = r#"{"dtype": "F32", "shape": [100, 200], "data_offsets": [0, 80000]}"#;
        let metadata: TensorMetadata = serde_json::from_str(json).unwrap();

        assert_eq!(metadata.dtype, SafeTensorsDtype::F32);
        assert_eq!(metadata.shape, vec![100, 200]);
        assert_eq!(metadata.data_offsets, [0, 80000]);
    }

    #[test]
    fn test_tensor_metadata_f16() {
        let json = r#"{"dtype": "F16", "shape": [10], "data_offsets": [0, 20]}"#;
        let metadata: TensorMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(metadata.dtype, SafeTensorsDtype::F16);
    }

    #[test]
    fn test_tensor_metadata_bf16() {
        let json = r#"{"dtype": "BF16", "shape": [10], "data_offsets": [0, 20]}"#;
        let metadata: TensorMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(metadata.dtype, SafeTensorsDtype::Bf16);
    }
}
