//! HOLW (Hologram Weights) file format for memory-mapped weight storage.
//!
//! This module defines the binary format for storing model weights separately
//! from the .holo file, enabling memory-mapped access for large models.
//!
//! # File Structure
//!
//! ```text
//! ┌────────────────────┐
//! │ Header (32 bytes)  │  Magic, version, count, offsets
//! ├────────────────────┤
//! │ Index Section      │  Name → (offset, size, dtype, shape)
//! ├────────────────────┤
//! │ Data Section       │  Page-aligned weight bytes
//! └────────────────────┘
//! ```
//!
//! # Benefits
//!
//! - Memory-mapped access: OS pages in weights on demand
//! - Zero-copy: Direct pointer access without loading entire file
//! - Large model support: Handle GB-sized weight files efficiently
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_onnx::core::weights_format::{WeightsFile, WeightsHeader};
//!
//! // Write weights
//! let mut writer = WeightsFileWriter::new();
//! writer.add_weight("layer1.weight", &data, &shape, DataType::F32)?;
//! let bytes = writer.finish()?;
//! std::fs::write("model.weights", bytes)?;
//!
//! // Read weights (mmap'd)
//! let file = WeightsFile::open("model.weights")?;
//! let weight = file.get_weight("layer1.weight")?;
//! ```

use crate::core::error::{OnnxError, Result};
use std::collections::HashMap;

/// Magic bytes identifying a HOLW file.
pub const WEIGHTS_MAGIC: [u8; 4] = *b"HOLW";

/// Current format version.
pub const WEIGHTS_VERSION: u32 = 1;

/// Page size for alignment (4KB on most systems).
pub const PAGE_SIZE: usize = 4096;

/// Data types for weight tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WeightDType {
    /// 32-bit floating point
    F32 = 0,
    /// 16-bit floating point
    F16 = 1,
    /// 16-bit brain floating point
    BF16 = 2,
    /// 32-bit signed integer
    I32 = 3,
    /// 64-bit signed integer
    I64 = 4,
    /// 8-bit unsigned integer
    U8 = 5,
    /// 8-bit signed integer (quantized)
    I8 = 6,
}

impl WeightDType {
    /// Size in bytes per element for this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            WeightDType::F32 | WeightDType::I32 => 4,
            WeightDType::F16 | WeightDType::BF16 => 2,
            WeightDType::I64 => 8,
            WeightDType::U8 | WeightDType::I8 => 1,
        }
    }

    /// Convert from raw byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(WeightDType::F32),
            1 => Some(WeightDType::F16),
            2 => Some(WeightDType::BF16),
            3 => Some(WeightDType::I32),
            4 => Some(WeightDType::I64),
            5 => Some(WeightDType::U8),
            6 => Some(WeightDType::I8),
            _ => None,
        }
    }
}

/// Header for a HOLW file (32 bytes).
///
/// Layout:
/// - magic: 4 bytes
/// - version: 4 bytes (little-endian u32)
/// - flags: 4 bytes (reserved)
/// - weight_count: 4 bytes (little-endian u32)
/// - index_size: 8 bytes (little-endian u64)
/// - data_offset: 8 bytes (little-endian u64, page-aligned)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct WeightsHeader {
    /// Magic bytes: b"HOLW"
    pub magic: [u8; 4],
    /// Format version (currently 1)
    pub version: u32,
    /// Flags (reserved for future use)
    pub flags: u32,
    /// Number of weight entries in the index
    pub weight_count: u32,
    /// Size of the index section in bytes
    pub index_size: u64,
    /// Offset to the data section (page-aligned)
    pub data_offset: u64,
}

impl WeightsHeader {
    /// Header size in bytes.
    pub const SIZE: usize = 32;

    /// Create a new header.
    pub fn new(weight_count: u32, index_size: u64, data_offset: u64) -> Self {
        Self {
            magic: WEIGHTS_MAGIC,
            version: WEIGHTS_VERSION,
            flags: 0,
            weight_count,
            index_size,
            data_offset,
        }
    }

    /// Serialize header to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.weight_count.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.index_size.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.data_offset.to_le_bytes());
        bytes
    }

    /// Parse header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(OnnxError::ParseError(format!(
                "HOLW header too short: {} < {}",
                bytes.len(),
                Self::SIZE
            )));
        }

        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != WEIGHTS_MAGIC {
            return Err(OnnxError::ParseError(format!(
                "Invalid HOLW magic: {:?}",
                magic
            )));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != WEIGHTS_VERSION {
            return Err(OnnxError::ParseError(format!(
                "Unsupported HOLW version: {} (expected {})",
                version, WEIGHTS_VERSION
            )));
        }

        let flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let weight_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let index_size = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let data_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

        Ok(Self {
            magic,
            version,
            flags,
            weight_count,
            index_size,
            data_offset,
        })
    }
}

/// Index entry for a single weight tensor.
///
/// Variable-length encoding:
/// - name_len: 4 bytes (u32)
/// - name: name_len bytes (UTF-8)
/// - dtype: 1 byte
/// - rank: 1 byte
/// - padding: 2 bytes (alignment)
/// - offset: 8 bytes (u64, relative to data section)
/// - size_bytes: 8 bytes (u64)
/// - shape: rank * 8 bytes (u64 per dimension)
#[derive(Debug, Clone)]
pub struct WeightIndexEntry {
    /// Weight name (e.g., "encoder.layer.0.attention.query.weight")
    pub name: String,
    /// Data type
    pub dtype: WeightDType,
    /// Offset from start of data section
    pub offset: u64,
    /// Size in bytes
    pub size_bytes: u64,
    /// Tensor shape
    pub shape: Vec<u64>,
}

impl WeightIndexEntry {
    /// Serialize index entry to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let rank = self.shape.len() as u8;

        // Calculate size: 4 (name_len) + name + 1 (dtype) + 1 (rank) + 2 (pad) + 8 (offset) + 8 (size) + rank*8 (shape)
        let size = 4 + name_bytes.len() + 4 + 16 + (rank as usize) * 8;
        let mut bytes = Vec::with_capacity(size);

        // Name length and name
        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);

        // dtype, rank, padding
        bytes.push(self.dtype as u8);
        bytes.push(rank);
        bytes.extend_from_slice(&[0u8; 2]); // padding for alignment

        // offset and size_bytes
        bytes.extend_from_slice(&self.offset.to_le_bytes());
        bytes.extend_from_slice(&self.size_bytes.to_le_bytes());

        // shape
        for dim in &self.shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }

        bytes
    }

    /// Parse index entry from bytes, returning (entry, bytes_consumed).
    pub fn from_bytes(bytes: &[u8]) -> Result<(Self, usize)> {
        if bytes.len() < 4 {
            return Err(OnnxError::ParseError("Index entry too short".into()));
        }

        let name_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;

        if bytes.len() < pos + name_len + 20 {
            return Err(OnnxError::ParseError("Index entry truncated".into()));
        }

        let name = String::from_utf8(bytes[pos..pos + name_len].to_vec())
            .map_err(|e| OnnxError::ParseError(format!("Invalid UTF-8 in weight name: {}", e)))?;
        pos += name_len;

        let dtype = WeightDType::from_u8(bytes[pos])
            .ok_or_else(|| OnnxError::ParseError(format!("Invalid dtype: {}", bytes[pos])))?;
        let rank = bytes[pos + 1] as usize;
        pos += 4; // dtype + rank + 2 padding

        if bytes.len() < pos + 16 + rank * 8 {
            return Err(OnnxError::ParseError("Index entry shape truncated".into()));
        }

        let offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let size_bytes = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()));
            pos += 8;
        }

        Ok((
            Self {
                name,
                dtype,
                offset,
                size_bytes,
                shape,
            },
            pos,
        ))
    }
}

/// Writer for creating HOLW files.
pub struct WeightsFileWriter {
    /// Index entries
    entries: Vec<WeightIndexEntry>,
    /// Weight data (concatenated)
    data: Vec<u8>,
    /// Current offset in data section
    current_offset: u64,
}

impl WeightsFileWriter {
    /// Create a new writer.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            data: Vec::new(),
            current_offset: 0,
        }
    }

    /// Add a weight tensor.
    ///
    /// # Arguments
    /// * `name` - Weight name
    /// * `data` - Raw weight bytes
    /// * `shape` - Tensor shape
    /// * `dtype` - Data type
    pub fn add_weight(&mut self, name: &str, data: &[u8], shape: &[usize], dtype: WeightDType) {
        let entry = WeightIndexEntry {
            name: name.to_string(),
            dtype,
            offset: self.current_offset,
            size_bytes: data.len() as u64,
            shape: shape.iter().map(|&d| d as u64).collect(),
        };

        self.entries.push(entry);
        self.data.extend_from_slice(data);
        self.current_offset += data.len() as u64;
    }

    /// Add a weight tensor from f32 data.
    pub fn add_weight_f32(&mut self, name: &str, data: &[f32], shape: &[usize]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.add_weight(name, bytes, shape, WeightDType::F32);
    }

    /// Finish writing and return the complete file bytes.
    pub fn finish(self) -> Vec<u8> {
        // Serialize index entries
        let mut index_bytes = Vec::new();
        for entry in &self.entries {
            index_bytes.extend(entry.to_bytes());
        }

        // Calculate data offset (page-aligned)
        let header_and_index_size = WeightsHeader::SIZE + index_bytes.len();
        let data_offset = align_to_page(header_and_index_size);
        let padding_size = data_offset - header_and_index_size;

        // Create header
        let header = WeightsHeader::new(
            self.entries.len() as u32,
            index_bytes.len() as u64,
            data_offset as u64,
        );

        // Assemble file
        let total_size = data_offset + self.data.len();
        let mut result = Vec::with_capacity(total_size);

        result.extend_from_slice(&header.to_bytes());
        result.extend_from_slice(&index_bytes);
        result.extend(std::iter::repeat_n(0u8, padding_size));
        result.extend_from_slice(&self.data);

        result
    }

    /// Get the number of weights added.
    pub fn weight_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the total data size in bytes.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }
}

impl Default for WeightsFileWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reader for HOLW files (works with memory-mapped data).
pub struct WeightsFileReader<'a> {
    /// Header
    header: WeightsHeader,
    /// Index entries
    index: HashMap<String, WeightIndexEntry>,
    /// Raw data slice (typically memory-mapped)
    data: &'a [u8],
    /// Data section start offset
    data_offset: usize,
}

impl<'a> WeightsFileReader<'a> {
    /// Create a reader from raw bytes (typically memory-mapped).
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let header = WeightsHeader::from_bytes(bytes)?;

        // Parse index entries
        let index_start = WeightsHeader::SIZE;
        let index_end = index_start + header.index_size as usize;

        if bytes.len() < index_end {
            return Err(OnnxError::ParseError("File truncated (index)".into()));
        }

        let mut index = HashMap::new();
        let mut pos = index_start;
        for _ in 0..header.weight_count {
            let (entry, consumed) = WeightIndexEntry::from_bytes(&bytes[pos..])?;
            index.insert(entry.name.clone(), entry);
            pos += consumed;
        }

        let data_offset = header.data_offset as usize;
        if bytes.len() < data_offset {
            return Err(OnnxError::ParseError("File truncated (data)".into()));
        }

        Ok(Self {
            header,
            index,
            data: bytes,
            data_offset,
        })
    }

    /// Get a weight by name as raw bytes.
    pub fn get_weight(&self, name: &str) -> Option<&'a [u8]> {
        let entry = self.index.get(name)?;
        let start = self.data_offset + entry.offset as usize;
        let end = start + entry.size_bytes as usize;

        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Get a weight by name as f32 slice.
    pub fn get_weight_f32(&self, name: &str) -> Option<&'a [f32]> {
        let bytes = self.get_weight(name)?;
        Some(bytemuck::cast_slice(bytes))
    }

    /// Get weight metadata.
    pub fn get_entry(&self, name: &str) -> Option<&WeightIndexEntry> {
        self.index.get(name)
    }

    /// Iterate over all weight names.
    pub fn weight_names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    /// Get the number of weights.
    pub fn weight_count(&self) -> usize {
        self.header.weight_count as usize
    }

    /// Get the header.
    pub fn header(&self) -> &WeightsHeader {
        &self.header
    }
}

/// Align a size up to the next page boundary.
fn align_to_page(size: usize) -> usize {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = WeightsHeader::new(5, 256, 4096);
        let bytes = header.to_bytes();
        let parsed = WeightsHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.magic, WEIGHTS_MAGIC);
        assert_eq!(parsed.version, WEIGHTS_VERSION);
        assert_eq!(parsed.weight_count, 5);
        assert_eq!(parsed.index_size, 256);
        assert_eq!(parsed.data_offset, 4096);
    }

    #[test]
    fn test_index_entry_roundtrip() {
        let entry = WeightIndexEntry {
            name: "encoder.layer.0.weight".to_string(),
            dtype: WeightDType::F32,
            offset: 1024,
            size_bytes: 4096,
            shape: vec![768, 768],
        };

        let bytes = entry.to_bytes();
        let (parsed, consumed) = WeightIndexEntry::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.name, entry.name);
        assert_eq!(parsed.dtype, entry.dtype);
        assert_eq!(parsed.offset, entry.offset);
        assert_eq!(parsed.size_bytes, entry.size_bytes);
        assert_eq!(parsed.shape, entry.shape);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_writer_reader_roundtrip() {
        let mut writer = WeightsFileWriter::new();

        // Add some test weights
        let weight1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weight2: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        writer.add_weight_f32("layer1.weight", &weight1, &[2, 2]);
        writer.add_weight_f32("layer2.bias", &weight2, &[6]);

        let bytes = writer.finish();

        // Read back
        let reader = WeightsFileReader::new(&bytes).unwrap();

        assert_eq!(reader.weight_count(), 2);

        let read_weight1 = reader.get_weight_f32("layer1.weight").unwrap();
        assert_eq!(read_weight1, &weight1[..]);

        let read_weight2 = reader.get_weight_f32("layer2.bias").unwrap();
        assert_eq!(read_weight2, &weight2[..]);

        // Check metadata
        let entry1 = reader.get_entry("layer1.weight").unwrap();
        assert_eq!(entry1.shape, vec![2, 2]);
        assert_eq!(entry1.dtype, WeightDType::F32);

        let entry2 = reader.get_entry("layer2.bias").unwrap();
        assert_eq!(entry2.shape, vec![6]);
    }

    #[test]
    fn test_page_alignment() {
        assert_eq!(align_to_page(0), 0);
        assert_eq!(align_to_page(1), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE + 1), PAGE_SIZE * 2);
    }

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(WeightDType::F32.element_size(), 4);
        assert_eq!(WeightDType::F16.element_size(), 2);
        assert_eq!(WeightDType::I64.element_size(), 8);
        assert_eq!(WeightDType::I8.element_size(), 1);
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(b"XXXX");
        let result = WeightsHeader::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_weights_file() {
        let writer = WeightsFileWriter::new();
        let bytes = writer.finish();

        let reader = WeightsFileReader::new(&bytes).unwrap();
        assert_eq!(reader.weight_count(), 0);
    }
}
