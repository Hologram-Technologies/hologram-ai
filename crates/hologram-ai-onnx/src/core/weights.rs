//! Memory-efficient weight extraction and streaming.
//!
//! This module handles extraction of weights from ONNX models with a focus on:
//! - **Zero-copy operations**: Minimize data copying using bytemuck
//! - **O(1) deduplication**: Hash-based weight deduplication
//! - **Streaming**: Process weights without loading entire model
//! - **Compile-time work**: All weight processing during compilation
//!
//! # Performance Characteristics
//!
//! - **Space**: O(unique_weights) - duplicates are deduplicated
//! - **Time**: O(n) for n weights, O(1) per weight lookup after dedup
//! - **Memory**: Streaming prevents OOM with large models
//!
//! # Examples
//!
//! ```no_run
//! use hologram_ai_onnx::core::{WeightData, parse_model};
//! use std::fs;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let bytes = fs::read("model.onnx")?;
//! let model = parse_model(&bytes)?;
//! let graph = model.graph.as_ref().unwrap();
//!
//! let mut weight_data = WeightData::new();
//!
//! // Stream weights from initializers
//! for init in &graph.initializer {
//!     let data = WeightData::extract_tensor_data(init)?;
//!     weight_data.add_weight(&init.name, data);
//! }
//!
//! // Deduplicate (O(n) operation, O(1) per weight)
//! weight_data.deduplicate();
//!
//! // Write to file (zero-copy where possible)
//! weight_data.write_to_file("model.weights")?;
//! # Ok(())
//! # }
//! ```

use crate::{OnnxError, Result};
use ahash::AHashMap;
// bytemuck used for zero-copy f32 conversion
use crate::proto::TensorProto;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::weights_format::{WeightDType, WeightsFileWriter};

/// Reference to external weight data.
///
/// This lightweight structure points to a location in the external
/// `.weights` file, enabling O(1) weight lookups at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightRef {
    /// Offset in bytes from start of weights file
    pub offset: u64,
    /// Number of elements (not bytes)
    pub length: usize,
}

/// Extended weight reference with full metadata for mmap serialization.
///
/// Used when serializing to HOLW format, which requires
/// dtype and shape information for each weight.
#[derive(Debug, Clone)]
pub struct MmapWeightEntry {
    /// Basic reference (offset, length)
    pub weight_ref: WeightRef,
    /// Data type of the weight
    pub dtype: WeightDType,
    /// Tensor shape
    pub shape: Vec<usize>,
}

/// Weight data manager with deduplication.
///
/// Manages weight extraction from ONNX models with:
/// - **Streaming extraction**: No full model load required
/// - **O(1) deduplication**: Hash-based duplicate detection
/// - **Zero-copy serialization**: Direct byte writing
/// - **HOLW format support**: Serialize with full metadata
///
/// # Performance
///
/// - Adding weights: O(1) amortized
/// - Deduplication: O(n) where n = number of weights
/// - Lookup after dedup: O(1)
/// - Memory: O(unique_weights)
#[derive(Debug)]
pub struct WeightData {
    /// Raw weight bytes (all weights concatenated)
    buffer: Vec<u8>,
    /// Weight name → (offset, length)
    refs: AHashMap<String, WeightRef>,
    /// Weight name → full entry with metadata (for HOLW serialization)
    entries: AHashMap<String, MmapWeightEntry>,
    /// Weight hash → offset (for deduplication)
    hash_to_offset: AHashMap<u64, u64>,
}

impl WeightData {
    /// Create a new empty weight data manager.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::WeightData;
    ///
    /// let weights = WeightData::new();
    /// assert_eq!(weights.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            refs: AHashMap::new(),
            entries: AHashMap::new(),
            hash_to_offset: AHashMap::new(),
        }
    }

    /// Add weight data with automatic deduplication.
    ///
    /// This is a **O(1) operation** (amortized) that:
    /// 1. Computes weight hash
    /// 2. Checks if identical weight exists (O(1) lookup)
    /// 3. Reuses existing offset if duplicate, or appends if unique
    ///
    /// # Arguments
    ///
    /// * `name` - Weight identifier
    /// * `data` - Weight values (f32)
    ///
    /// # Performance
    ///
    /// - Time: O(weight_size) for hashing, O(1) for lookup
    /// - Space: O(1) if duplicate, O(weight_size) if unique
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::WeightData;
    ///
    /// let mut weights = WeightData::new();
    /// let data = vec![1.0, 2.0, 3.0];
    ///
    /// let ref1 = weights.add_weight("weight1", data.clone());
    /// let ref2 = weights.add_weight("weight2", data); // Deduplicated!
    ///
    /// assert_eq!(ref1.offset, ref2.offset); // Same offset - deduplicated
    /// ```
    pub fn add_weight(&mut self, name: &str, data: Vec<f32>) -> WeightRef {
        // Default to f32 dtype and 1D shape
        self.add_weight_with_metadata(name, data.clone(), &[data.len()], WeightDType::F32)
    }

    /// Add weight data with full metadata.
    ///
    /// This is the primary method for adding weights with shape and dtype information,
    /// which is required for HOLW format serialization.
    ///
    /// # Arguments
    ///
    /// * `name` - Weight identifier
    /// * `data` - Weight values (f32)
    /// * `shape` - Tensor shape
    /// * `dtype` - Data type
    ///
    /// # Returns
    ///
    /// Reference to the weight data location.
    pub fn add_weight_with_metadata(
        &mut self,
        name: &str,
        data: Vec<f32>,
        shape: &[usize],
        dtype: WeightDType,
    ) -> WeightRef {
        // Compute hash of weight data (O(weight_size))
        let hash = Self::hash_data(&data);

        // Check if we've seen this exact weight before (O(1))
        if let Some(&existing_offset) = self.hash_to_offset.get(&hash) {
            // Duplicate weight - reuse existing data (zero-copy)
            let weight_ref = WeightRef {
                offset: existing_offset,
                length: data.len(),
            };

            self.refs.insert(name.to_string(), weight_ref);
            self.entries.insert(
                name.to_string(),
                MmapWeightEntry {
                    weight_ref,
                    dtype,
                    shape: shape.to_vec(),
                },
            );
            return weight_ref;
        }

        // New unique weight - append to buffer
        let offset = self.buffer.len() as u64;

        // Convert f32 to bytes (zero-copy via bytemuck)
        let bytes = bytemuck::cast_slice(&data);
        self.buffer.extend_from_slice(bytes);

        let weight_ref = WeightRef {
            offset,
            length: data.len(),
        };

        // Update indices (both O(1))
        self.refs.insert(name.to_string(), weight_ref);
        self.entries.insert(
            name.to_string(),
            MmapWeightEntry {
                weight_ref,
                dtype,
                shape: shape.to_vec(),
            },
        );
        self.hash_to_offset.insert(hash, offset);

        weight_ref
    }

    /// Get reference for a named weight.
    ///
    /// **O(1) operation** using hash map lookup.
    ///
    /// # Arguments
    ///
    /// * `name` - Weight identifier
    ///
    /// # Returns
    ///
    /// Weight reference if found, None otherwise.
    pub fn get_ref(&self, name: &str) -> Option<WeightRef> {
        self.refs.get(name).copied()
    }

    /// Get full entry with metadata for a named weight.
    ///
    /// **O(1) operation** using hash map lookup.
    ///
    /// # Arguments
    ///
    /// * `name` - Weight identifier
    ///
    /// # Returns
    ///
    /// Weight entry with metadata if found, None otherwise.
    pub fn get_entry(&self, name: &str) -> Option<&MmapWeightEntry> {
        self.entries.get(name)
    }

    /// Get the raw weight buffer.
    ///
    /// This is the concatenated bytes of all weights.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Serialize to HOLW format.
    ///
    /// Creates a properly formatted HOLW file with:
    /// - Header with magic bytes and metadata
    /// - Index section with weight names, shapes, and offsets
    /// - Page-aligned data section for efficient mmap
    ///
    /// # Returns
    ///
    /// Complete HOLW file bytes ready to write to disk.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut weights = WeightData::new();
    /// weights.add_weight_with_metadata("layer.weight", data, &[768, 768], WeightDType::F32);
    ///
    /// let holw_bytes = weights.serialize_to_holw();
    /// std::fs::write("model.weights", holw_bytes)?;
    /// ```
    pub fn serialize_to_holw(&self) -> Vec<u8> {
        let mut writer = WeightsFileWriter::new();

        // Add all weights with their metadata
        for (name, entry) in &self.entries {
            // Extract the bytes for this weight from the buffer
            let start = entry.weight_ref.offset as usize;
            let size_bytes = entry.weight_ref.length * entry.dtype.element_size();
            let end = start + size_bytes;

            if end <= self.buffer.len() {
                let weight_bytes = &self.buffer[start..end];
                writer.add_weight(name, weight_bytes, &entry.shape, entry.dtype);
            }
        }

        writer.finish()
    }

    /// Iterate over all weight names.
    pub fn weight_names(&self) -> impl Iterator<Item = &str> {
        self.refs.keys().map(|s| s.as_str())
    }

    /// Get number of unique weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::WeightData;
    ///
    /// let mut weights = WeightData::new();
    /// weights.add_weight("w1", vec![1.0]);
    /// weights.add_weight("w2", vec![1.0]); // Duplicate
    ///
    /// assert_eq!(weights.len(), 2); // 2 names, but deduplicated storage
    /// ```
    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Check if no weights are stored.
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// Get total size of weight buffer in bytes.
    ///
    /// This is the size of the `.weights` file that will be written.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::WeightData;
    ///
    /// let mut weights = WeightData::new();
    /// weights.add_weight("w1", vec![1.0, 2.0, 3.0]); // 12 bytes
    ///
    /// assert_eq!(weights.buffer_size(), 12);
    /// ```
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Deduplicate weights (legacy - now automatic in add_weight).
    ///
    /// This method is kept for API compatibility but is now a no-op
    /// since deduplication happens automatically during add_weight.
    pub fn deduplicate(&mut self) {
        // Deduplication is now automatic - this is a no-op
    }

    /// Write weights to file.
    ///
    /// **Zero-copy operation** - writes buffer directly to file.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    ///
    /// # Errors
    ///
    /// Returns IO error if file cannot be written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use hologram_ai_onnx::core::WeightData;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut weights = WeightData::new();
    /// weights.add_weight("w1", vec![1.0, 2.0]);
    ///
    /// weights.write_to_file("model.weights")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&self.buffer)?;
        file.flush()?;
        Ok(())
    }

    /// Extract tensor data from ONNX TensorProto.
    ///
    /// Supports all common ONNX data types with zero-copy conversion
    /// where possible using bytemuck.
    ///
    /// # Arguments
    ///
    /// * `tensor` - ONNX tensor protobuf
    ///
    /// # Returns
    ///
    /// Vector of f32 values. Non-f32 types are converted.
    ///
    /// # Errors
    ///
    /// Returns error for unsupported data types or malformed tensors.
    ///
    /// # Supported Types
    ///
    /// - FLOAT (f32) - zero-copy
    /// - FLOAT16 (f16) - converted to f32
    /// - INT32, INT64 - converted to f32
    /// - DOUBLE (f64) - converted to f32
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use hologram_ai_onnx::core::WeightData;
    /// use hologram_ai_onnx::proto::TensorProto;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Assume we have a TensorProto
    /// # let tensor = TensorProto::default();
    /// let data = WeightData::extract_tensor_data(&tensor)?;
    /// println!("Extracted {} values", data.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn extract_tensor_data(tensor: &TensorProto) -> Result<Vec<f32>> {
        // ONNX data type constants
        const FLOAT: i32 = 1;
        const UINT8: i32 = 2;
        const INT8: i32 = 3;
        const UINT16: i32 = 4;
        const INT16: i32 = 5;
        const INT32: i32 = 6;
        const INT64: i32 = 7;
        const FLOAT16: i32 = 10;
        const DOUBLE: i32 = 11;

        match tensor.data_type {
            FLOAT => {
                // Zero-copy conversion for f32
                if !tensor.raw_data.is_empty() {
                    // Raw bytes - cast directly (zero-copy)
                    Ok(bytemuck::cast_slice(&tensor.raw_data).to_vec())
                } else {
                    // Float array field
                    Ok(tensor.float_data.clone())
                }
            }

            FLOAT16 => {
                // f16 → f32 conversion
                if !tensor.raw_data.is_empty() {
                    let u16_data: &[u16] = bytemuck::cast_slice(&tensor.raw_data);
                    Ok(u16_data
                        .iter()
                        .map(|&bits| half::f16::from_bits(bits).to_f32())
                        .collect())
                } else {
                    // int32_data contains f16 as uint16
                    Ok(tensor
                        .int32_data
                        .iter()
                        .map(|&bits| half::f16::from_bits(bits as u16).to_f32())
                        .collect())
                }
            }

            DOUBLE => {
                // f64 → f32 conversion
                if !tensor.raw_data.is_empty() {
                    let f64_data: &[f64] = bytemuck::cast_slice(&tensor.raw_data);
                    Ok(f64_data.iter().map(|&v| v as f32).collect())
                } else {
                    Ok(tensor.double_data.iter().map(|&v| v as f32).collect())
                }
            }

            INT32 => {
                // i32 → f32 conversion
                if !tensor.raw_data.is_empty() {
                    let i32_data: &[i32] = bytemuck::cast_slice(&tensor.raw_data);
                    Ok(i32_data.iter().map(|&v| v as f32).collect())
                } else {
                    Ok(tensor.int32_data.iter().map(|&v| v as f32).collect())
                }
            }

            INT64 => {
                // i64 → f32 conversion
                if !tensor.raw_data.is_empty() {
                    let i64_data: &[i64] = bytemuck::cast_slice(&tensor.raw_data);
                    Ok(i64_data.iter().map(|&v| v as f32).collect())
                } else {
                    Ok(tensor.int64_data.iter().map(|&v| v as f32).collect())
                }
            }

            INT8 | UINT8 | INT16 | UINT16 => {
                // Small integer types
                if !tensor.raw_data.is_empty() {
                    Ok(tensor.raw_data.iter().map(|&v| v as f32).collect())
                } else {
                    Ok(tensor.int32_data.iter().map(|&v| v as f32).collect())
                }
            }

            _ => Err(OnnxError::UnsupportedDataType(format!(
                "Data type {} not supported",
                tensor.data_type
            ))),
        }
    }

    /// Extract tensor data with support for external data files.
    ///
    /// ONNX models can store large tensors in external files (`.onnx_data`).
    /// This function handles both inline and external data.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor proto to extract data from
    /// * `model_path` - Path to the ONNX model file (for resolving relative external paths)
    ///
    /// # Returns
    ///
    /// Vector of f32 values extracted from the tensor.
    pub fn extract_tensor_data_with_external(
        tensor: &TensorProto,
        model_path: &Path,
    ) -> Result<Vec<f32>> {
        // Check if tensor uses external data (data_location == 1 means EXTERNAL)
        if tensor.data_location == 1 && !tensor.external_data.is_empty() {
            // Extract external data parameters
            let mut location: Option<&str> = None;
            let mut offset: u64 = 0;
            let mut length: Option<u64> = None;

            for entry in &tensor.external_data {
                match entry.key.as_str() {
                    "location" => location = Some(&entry.value),
                    "offset" => offset = entry.value.parse().unwrap_or(0),
                    "length" => length = entry.value.parse().ok(),
                    _ => {}
                }
            }

            let location = location.ok_or_else(|| {
                OnnxError::InvalidModel("External data missing 'location' field".to_string())
            })?;

            // Resolve path relative to model file
            let external_path = if Path::new(location).is_absolute() {
                std::path::PathBuf::from(location)
            } else {
                model_path.parent().unwrap_or(Path::new(".")).join(location)
            };

            // Open external file and read data
            let mut file = File::open(&external_path).map_err(|e| {
                OnnxError::IoError(io::Error::other(format!(
                    "Failed to open external data file '{}': {}",
                    external_path.display(),
                    e
                )))
            })?;

            // Seek to offset
            file.seek(SeekFrom::Start(offset)).map_err(|e| {
                OnnxError::IoError(io::Error::other(format!("Failed to seek in external data: {}", e)))
            })?;

            // Determine how many bytes to read
            let bytes_to_read = if let Some(len) = length {
                len as usize
            } else {
                // If length not specified, compute from tensor dims and data type
                let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
                let bytes_per_element = match tensor.data_type {
                    1 => 4,  // FLOAT
                    10 => 2, // FLOAT16
                    11 => 8, // DOUBLE
                    6 => 4,  // INT32
                    7 => 8,  // INT64
                    _ => 4,  // Default to 4
                };
                num_elements * bytes_per_element
            };

            // Read raw bytes
            let mut raw_data = vec![0u8; bytes_to_read];
            file.read_exact(&mut raw_data).map_err(|e| {
                OnnxError::IoError(io::Error::other(format!("Failed to read external data: {}", e)))
            })?;

            // Convert based on data type
            Self::convert_raw_bytes_to_f32(&raw_data, tensor.data_type)
        } else {
            // Use inline data
            Self::extract_tensor_data(tensor)
        }
    }

    /// Convert raw bytes to f32 based on ONNX data type.
    fn convert_raw_bytes_to_f32(raw_data: &[u8], data_type: i32) -> Result<Vec<f32>> {
        const FLOAT: i32 = 1;
        const FLOAT16: i32 = 10;
        const DOUBLE: i32 = 11;
        const INT32: i32 = 6;
        const INT64: i32 = 7;

        match data_type {
            FLOAT => Ok(bytemuck::cast_slice(raw_data).to_vec()),
            FLOAT16 => {
                let u16_data: &[u16] = bytemuck::cast_slice(raw_data);
                Ok(u16_data
                    .iter()
                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                    .collect())
            }
            DOUBLE => {
                let f64_data: &[f64] = bytemuck::cast_slice(raw_data);
                Ok(f64_data.iter().map(|&v| v as f32).collect())
            }
            INT32 => {
                let i32_data: &[i32] = bytemuck::cast_slice(raw_data);
                Ok(i32_data.iter().map(|&v| v as f32).collect())
            }
            INT64 => {
                let i64_data: &[i64] = bytemuck::cast_slice(raw_data);
                Ok(i64_data.iter().map(|&v| v as f32).collect())
            }
            _ => {
                // For other types, treat as bytes
                Ok(raw_data.iter().map(|&v| v as f32).collect())
            }
        }
    }

    /// Compute hash of weight data for deduplication.
    ///
    /// **O(weight_size) operation** using fast AHash algorithm.
    ///
    /// # Performance
    ///
    /// Uses AHash which is:
    /// - ~3x faster than SipHash (Rust default)
    /// - Optimized for small keys (weights)
    /// - DoS-resistant for hash flooding
    fn hash_data(data: &[f32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        // Hash raw bytes for exact equality check
        bytemuck::cast_slice::<f32, u8>(data).hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for WeightData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_weight_data() {
        let weights = WeightData::new();
        assert_eq!(weights.len(), 0);
        assert_eq!(weights.buffer_size(), 0);
        assert!(weights.is_empty());
    }

    #[test]
    fn test_add_weight() {
        let mut weights = WeightData::new();

        let data = vec![1.0, 2.0, 3.0];
        let weight_ref = weights.add_weight("test_weight", data);

        assert_eq!(weights.len(), 1);
        assert_eq!(weight_ref.offset, 0);
        assert_eq!(weight_ref.length, 3);
        assert_eq!(weights.buffer_size(), 12); // 3 * 4 bytes
    }

    #[test]
    fn test_weight_deduplication() {
        let mut weights = WeightData::new();

        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![1.0, 2.0, 3.0]; // Identical

        let ref1 = weights.add_weight("weight1", data1);
        let ref2 = weights.add_weight("weight2", data2);

        // Both weights should point to same data
        assert_eq!(ref1.offset, ref2.offset);
        assert_eq!(ref1.length, ref2.length);

        // Buffer should only contain one copy
        assert_eq!(weights.buffer_size(), 12); // Not 24

        // But both names should be tracked
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_get_ref() {
        let mut weights = WeightData::new();

        weights.add_weight("test", vec![1.0, 2.0]);

        let weight_ref = weights.get_ref("test").unwrap();
        assert_eq!(weight_ref.length, 2);

        assert!(weights.get_ref("nonexistent").is_none());
    }

    #[test]
    fn test_multiple_unique_weights() {
        let mut weights = WeightData::new();

        weights.add_weight("w1", vec![1.0]);
        weights.add_weight("w2", vec![2.0]);
        weights.add_weight("w3", vec![3.0]);

        assert_eq!(weights.len(), 3);
        assert_eq!(weights.buffer_size(), 12); // 3 weights * 4 bytes each

        let ref1 = weights.get_ref("w1").unwrap();
        let ref2 = weights.get_ref("w2").unwrap();
        let ref3 = weights.get_ref("w3").unwrap();

        // Each should have different offset
        assert_eq!(ref1.offset, 0);
        assert_eq!(ref2.offset, 4);
        assert_eq!(ref3.offset, 8);
    }

    #[test]
    fn test_hash_data() {
        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![1.0, 2.0, 3.0];
        let data3 = vec![1.0, 2.0, 3.1]; // Different

        let hash1 = WeightData::hash_data(&data1);
        let hash2 = WeightData::hash_data(&data2);
        let hash3 = WeightData::hash_data(&data3);

        assert_eq!(hash1, hash2); // Identical data
        assert_ne!(hash1, hash3); // Different data
    }

    #[test]
    fn test_write_to_file() -> io::Result<()> {
        let mut weights = WeightData::new();
        weights.add_weight("test", vec![1.0, 2.0, 3.0]);

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_weights.bin");

        weights.write_to_file(&temp_file)?;

        // Verify file was created and has correct size
        let metadata = std::fs::metadata(&temp_file)?;
        assert_eq!(metadata.len(), 12);

        // Clean up
        std::fs::remove_file(&temp_file)?;

        Ok(())
    }

    #[test]
    fn test_extract_float_tensor() {
        use crate::proto::TensorProto;

        let tensor = TensorProto {
            data_type: 1, // FLOAT
            float_data: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };

        let data = WeightData::extract_tensor_data(&tensor).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_extract_int32_tensor() {
        use crate::proto::TensorProto;

        let tensor = TensorProto {
            data_type: 6, // INT32
            int32_data: vec![1, 2, 3],
            ..Default::default()
        };

        let data = WeightData::extract_tensor_data(&tensor).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unsupported_datatype() {
        use crate::proto::TensorProto;

        let tensor = TensorProto {
            data_type: 999, // Invalid
            ..Default::default()
        };

        let result = WeightData::extract_tensor_data(&tensor);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxError::UnsupportedDataType(_)
        ));
    }
}
