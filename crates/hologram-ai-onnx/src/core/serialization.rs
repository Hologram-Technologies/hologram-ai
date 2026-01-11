#![allow(missing_docs)]
//! Serialization types for .holo format.
//!
//! This module contains type definitions and constants for the .holo file format.
//! Full serialization functionality requires implementation when compilation support
//! is added.

use crate::{OnnxError, Result};
use serde::{Deserialize, Serialize};

// =============================================================================
// Constants
// =============================================================================

/// Magic bytes for .holo files
pub const HOLO_MAGIC: &[u8; 4] = b"HOLO";

/// Current format version
pub const FORMAT_VERSION: u32 = 1;

/// Header size in bytes
pub const HEADER_SIZE: usize = 40;

/// Flag: has external weights file
pub const FLAG_EXTERNAL_WEIGHTS: u32 = 0x01;

/// Flag: has compressed data
#[allow(dead_code)] // Reserved for future use
pub const FLAG_COMPRESSED: u32 = 0x02;

// =============================================================================
// Bundle Format Constants (reserved for future bundle format implementation)
// =============================================================================

#[allow(dead_code)]
/// Magic bytes for bundle files (single-file with embedded weights)
pub const HOLB_MAGIC: [u8; 4] = *b"HOLB";

#[allow(dead_code)]
/// Magic bytes for hologram plan files (from hologram backend)
pub const HOLP_MAGIC: [u8; 4] = *b"HOLP";

#[allow(dead_code)]
/// Bundle format version
pub const BUNDLE_VERSION: u32 = 1;

#[allow(dead_code)]
/// Bundle header size in bytes (fixed 64 bytes for alignment)
pub const BUNDLE_HEADER_SIZE: usize = 64;

#[allow(dead_code)]
/// Page size for weight section alignment (4KB)
pub const PAGE_SIZE: usize = 4096;

// =============================================================================
// Public Types (needed for re-exports)
// =============================================================================

/// Serializable representation of a .holo file header.
#[derive(Debug, Clone)]
pub struct HoloHeader {
    pub version: u32,
    pub flags: u32,
    pub metadata_offset: u64,
    pub graph_offset: u64,
    pub weights_offset: u64,
}

impl HoloHeader {
    /// Convert header to raw bytes for writing.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(HOLO_MAGIC);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..20].copy_from_slice(&self.metadata_offset.to_le_bytes());
        buf[20..28].copy_from_slice(&self.graph_offset.to_le_bytes());
        buf[28..36].copy_from_slice(&self.weights_offset.to_le_bytes());
        buf
    }

    /// Parse header from raw bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(OnnxError::InvalidModel("Header too small".into()));
        }
        if &buf[0..4] != HOLO_MAGIC {
            return Err(OnnxError::InvalidModel("Invalid magic bytes".into()));
        }

        Ok(Self {
            version: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            flags: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            metadata_offset: u64::from_le_bytes([
                buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
            ]),
            graph_offset: u64::from_le_bytes([
                buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27],
            ]),
            weights_offset: u64::from_le_bytes([
                buf[28], buf[29], buf[30], buf[31], buf[32], buf[33], buf[34], buf[35],
            ]),
        })
    }

    pub fn has_external_weights(&self) -> bool {
        self.flags & FLAG_EXTERNAL_WEIGHTS != 0
    }
}

// =============================================================================
// Bundle Header (HOLB format)
// =============================================================================

/// Header for the unified bundle format (HOLB).
///
/// This format combines the computation graph and weights into a single file
/// while maintaining memory-mapping capability for the weights section.
///
/// # Layout
/// ```text
/// +================================+
/// |  Bundle Header (64 bytes)      |  Magic: "HOLB", offsets, checksums
/// +================================+
/// |  Graph Section (HOLP data)     |  Existing hologram format bytes
/// +--------------------------------+
/// |  Padding to 4KB boundary       |
/// +================================+
/// |  Weights Section               |  Page-aligned for mmap
/// +================================+
/// ```
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoloBundleHeader {
    /// Magic bytes: "HOLB"
    pub magic: [u8; 4],
    /// Format version (currently 1)
    pub version: u32,
    /// Flags (reserved for future use)
    pub flags: u32,
    /// Offset to graph section (always 64, after header)
    pub graph_offset: u64,
    /// Size of graph section in bytes
    pub graph_size: u64,
    /// Offset to weights section (page-aligned)
    pub weights_offset: u64,
    /// Size of weights section in bytes
    pub weights_size: u64,
    /// CRC32 checksum of graph section
    pub graph_checksum: u32,
    /// CRC32 checksum of weights section
    pub weights_checksum: u32,
    // Reserved bytes: 12 bytes to reach 64-byte header
}

impl HoloBundleHeader {
    /// Create a new bundle header with the given section sizes.
    ///
    /// Automatically calculates the page-aligned weights offset.
    pub fn new(graph_size: u64, weights_size: u64) -> Self {
        let weights_offset = Self::calculate_weights_offset(graph_size);
        Self {
            magic: HOLB_MAGIC,
            version: BUNDLE_VERSION,
            flags: 0,
            graph_offset: BUNDLE_HEADER_SIZE as u64,
            graph_size,
            weights_offset,
            weights_size,
            graph_checksum: 0,
            weights_checksum: 0,
        }
    }

    /// Calculate the page-aligned offset for the weights section.
    ///
    /// The weights section starts at the next 4KB boundary after the graph section.
    pub fn calculate_weights_offset(graph_size: u64) -> u64 {
        let graph_end = BUNDLE_HEADER_SIZE as u64 + graph_size;
        // Round up to next page boundary
        graph_end.div_ceil(PAGE_SIZE as u64) * PAGE_SIZE as u64
    }

    /// Set checksums for graph and weights sections.
    pub fn set_checksums(&mut self, graph_checksum: u32, weights_checksum: u32) {
        self.graph_checksum = graph_checksum;
        self.weights_checksum = weights_checksum;
    }

    /// Convert header to raw bytes for writing.
    pub fn to_bytes(&self) -> [u8; BUNDLE_HEADER_SIZE] {
        let mut buf = [0u8; BUNDLE_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..20].copy_from_slice(&self.graph_offset.to_le_bytes());
        buf[20..28].copy_from_slice(&self.graph_size.to_le_bytes());
        buf[28..36].copy_from_slice(&self.weights_offset.to_le_bytes());
        buf[36..44].copy_from_slice(&self.weights_size.to_le_bytes());
        buf[44..48].copy_from_slice(&self.graph_checksum.to_le_bytes());
        buf[48..52].copy_from_slice(&self.weights_checksum.to_le_bytes());
        // bytes 52-63 are reserved (zero-initialized)
        buf
    }

    /// Parse header from raw bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < BUNDLE_HEADER_SIZE {
            return Err(OnnxError::InvalidModel("Bundle header too small".into()));
        }

        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != HOLB_MAGIC {
            return Err(OnnxError::InvalidModel(format!(
                "Invalid bundle magic: expected {:?}, got {:?}",
                HOLB_MAGIC, magic
            )));
        }

        Ok(Self {
            magic,
            version: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            flags: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            graph_offset: u64::from_le_bytes([
                buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
            ]),
            graph_size: u64::from_le_bytes([
                buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27],
            ]),
            weights_offset: u64::from_le_bytes([
                buf[28], buf[29], buf[30], buf[31], buf[32], buf[33], buf[34], buf[35],
            ]),
            weights_size: u64::from_le_bytes([
                buf[36], buf[37], buf[38], buf[39], buf[40], buf[41], buf[42], buf[43],
            ]),
            graph_checksum: u32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]),
            weights_checksum: u32::from_le_bytes([buf[48], buf[49], buf[50], buf[51]]),
        })
    }

    /// Check if weights section is present and non-empty.
    pub fn has_weights(&self) -> bool {
        self.weights_size > 0
    }

    /// Validate the header fields for consistency.
    pub fn validate(&self) -> Result<()> {
        if self.magic != HOLB_MAGIC {
            return Err(OnnxError::InvalidModel("Invalid bundle magic".into()));
        }
        if self.version != BUNDLE_VERSION {
            return Err(OnnxError::InvalidModel(format!(
                "Unsupported bundle version: {}",
                self.version
            )));
        }
        if self.graph_offset != BUNDLE_HEADER_SIZE as u64 {
            return Err(OnnxError::InvalidModel(
                "Graph offset must be 64 (after header)".into(),
            ));
        }
        if self.weights_size > 0 && !self.weights_offset.is_multiple_of(PAGE_SIZE as u64) {
            return Err(OnnxError::InvalidModel(
                "Weights offset must be page-aligned".into(),
            ));
        }
        Ok(())
    }
}

// =============================================================================
// Format Detection
// =============================================================================

/// Detected file format based on magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoloFormat {
    /// Bundle format (HOLB) - single model with embedded weights
    Bundle,
    /// Pipeline format (HOLM) - multi-model with embedded weights
    Pipeline,
    /// Plan format (HOLP) - hologram backend format, may have external weights
    Plan,
    /// Legacy HOLO format
    Legacy,
    /// Unknown format
    Unknown,
}

impl HoloFormat {
    /// Detect format from magic bytes.
    pub fn detect(magic: &[u8]) -> Self {
        if magic.len() < 4 {
            return Self::Unknown;
        }
        match &magic[0..4] {
            b"HOLB" => Self::Bundle,
            b"HOLM" => Self::Pipeline,
            b"HOLP" => Self::Plan,
            b"HOLO" => Self::Legacy,
            _ => Self::Unknown,
        }
    }

    /// Check if this format is a bundle (single-file with weights).
    pub fn is_bundle(&self) -> bool {
        matches!(self, Self::Bundle)
    }

    /// Check if this format may have external weights.
    pub fn may_have_external_weights(&self) -> bool {
        matches!(self, Self::Plan | Self::Legacy)
    }

    /// Check if this format is a pipeline bundle.
    pub fn is_pipeline(&self) -> bool {
        matches!(self, Self::Pipeline)
    }
}

// =============================================================================
// Pipeline Bundle Header (HOLM format)
// =============================================================================

/// Magic bytes for pipeline bundle files (multi-model with embedded weights)
pub const HOLM_MAGIC: [u8; 4] = *b"HOLM";

/// Pipeline bundle format version
pub const PIPELINE_VERSION: u32 = 1;

/// Pipeline header size in bytes (fixed 64 bytes for alignment)
pub const PIPELINE_HEADER_SIZE: usize = 64;

/// Header for the pipeline bundle format (HOLM).
///
/// This format packages multiple HOLB bundles into a single file,
/// enabling deployment of complete ML pipelines (encoder, decoder, tokenizer)
/// as a single artifact with efficient per-model mmap access.
///
/// # Layout
/// ```text
/// +================================+
/// |  Pipeline Header (64 bytes)    |  Magic: "HOLM", model count, flags
/// +================================+
/// |  Model Index (variable)        |  Per-model: name, offset, size, checksum
/// +--------------------------------+
/// |  Padding to 4KB boundary       |
/// +================================+
/// |  Model 0 (HOLB bundle)         |  Complete HOLB with graph+weights
/// +================================+
/// |  Model 1 (HOLB bundle)         |  Complete HOLB with graph+weights
/// +================================+
/// |  ...                           |
/// +================================+
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoloPipelineHeader {
    /// Magic bytes: "HOLM"
    pub magic: [u8; 4],
    /// Format version (currently 1)
    pub version: u32,
    /// Flags (reserved for future use)
    pub flags: u32,
    /// Number of models in the pipeline
    pub model_count: u32,
    /// Offset to the model index (always 64, after header)
    pub index_offset: u64,
    /// Size of the model index section in bytes
    pub index_size: u64,
    /// Offset to first model (page-aligned)
    pub models_offset: u64,
    /// Total size of all model data
    pub models_total_size: u64,
    /// CRC32 checksum of the index section
    pub index_checksum: u32,
    // Reserved bytes: 12 bytes to reach 64-byte header
}

impl HoloPipelineHeader {
    /// Create a new pipeline header.
    ///
    /// The models_offset is calculated to be page-aligned after the index.
    pub fn new(model_count: u32, index_size: u64) -> Self {
        let index_end = PIPELINE_HEADER_SIZE as u64 + index_size;
        let models_offset = index_end.div_ceil(PAGE_SIZE as u64) * PAGE_SIZE as u64;
        Self {
            magic: HOLM_MAGIC,
            version: PIPELINE_VERSION,
            flags: 0,
            model_count,
            index_offset: PIPELINE_HEADER_SIZE as u64,
            index_size,
            models_offset,
            models_total_size: 0, // Set later when models are added
            index_checksum: 0,
        }
    }

    /// Set the total models size.
    pub fn set_models_total_size(&mut self, size: u64) {
        self.models_total_size = size;
    }

    /// Set the index checksum.
    pub fn set_index_checksum(&mut self, checksum: u32) {
        self.index_checksum = checksum;
    }

    /// Calculate the page-aligned offset for models section.
    pub fn calculate_models_offset(index_size: u64) -> u64 {
        let index_end = PIPELINE_HEADER_SIZE as u64 + index_size;
        index_end.div_ceil(PAGE_SIZE as u64) * PAGE_SIZE as u64
    }

    /// Convert header to raw bytes for writing.
    pub fn to_bytes(&self) -> [u8; PIPELINE_HEADER_SIZE] {
        let mut buf = [0u8; PIPELINE_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..16].copy_from_slice(&self.model_count.to_le_bytes());
        buf[16..24].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.index_size.to_le_bytes());
        buf[32..40].copy_from_slice(&self.models_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&self.models_total_size.to_le_bytes());
        buf[48..52].copy_from_slice(&self.index_checksum.to_le_bytes());
        // bytes 52-63 are reserved (zero-initialized)
        buf
    }

    /// Parse header from raw bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < PIPELINE_HEADER_SIZE {
            return Err(OnnxError::InvalidModel("Pipeline header too small".into()));
        }

        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != HOLM_MAGIC {
            return Err(OnnxError::InvalidModel(format!(
                "Invalid pipeline magic: expected {:?}, got {:?}",
                HOLM_MAGIC, magic
            )));
        }

        Ok(Self {
            magic,
            version: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            flags: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            model_count: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            index_offset: u64::from_le_bytes([
                buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
            ]),
            index_size: u64::from_le_bytes([
                buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
            ]),
            models_offset: u64::from_le_bytes([
                buf[32], buf[33], buf[34], buf[35], buf[36], buf[37], buf[38], buf[39],
            ]),
            models_total_size: u64::from_le_bytes([
                buf[40], buf[41], buf[42], buf[43], buf[44], buf[45], buf[46], buf[47],
            ]),
            index_checksum: u32::from_le_bytes([buf[48], buf[49], buf[50], buf[51]]),
        })
    }

    /// Validate the header fields for consistency.
    pub fn validate(&self) -> Result<()> {
        if self.magic != HOLM_MAGIC {
            return Err(OnnxError::InvalidModel("Invalid pipeline magic".into()));
        }
        if self.version != PIPELINE_VERSION {
            return Err(OnnxError::InvalidModel(format!(
                "Unsupported pipeline version: {}",
                self.version
            )));
        }
        if self.index_offset != PIPELINE_HEADER_SIZE as u64 {
            return Err(OnnxError::InvalidModel(
                "Index offset must be 64 (after header)".into(),
            ));
        }
        if self.model_count > 0 && !self.models_offset.is_multiple_of(PAGE_SIZE as u64) {
            return Err(OnnxError::InvalidModel(
                "Models offset must be page-aligned".into(),
            ));
        }
        Ok(())
    }
}

/// Index entry for a model within a pipeline bundle.
///
/// Each entry describes one HOLB bundle embedded in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineModelEntry {
    /// Model name (e.g., "encoder", "decoder", "tokenizer")
    pub name: String,
    /// Offset to the HOLB data within the pipeline file (page-aligned)
    pub offset: u64,
    /// Size of the HOLB bundle in bytes
    pub size: u64,
    /// CRC32 checksum of the HOLB bundle
    pub checksum: u32,
}

impl PipelineModelEntry {
    /// Create a new model entry.
    pub fn new(name: String, offset: u64, size: u64, checksum: u32) -> Self {
        Self {
            name,
            offset,
            size,
            checksum,
        }
    }

    /// Serialize entry to bytes.
    ///
    /// Format: name_len (4) | name (var) | offset (8) | size (8) | checksum (4)
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.name.as_bytes();
        let mut buf = Vec::with_capacity(4 + name_bytes.len() + 8 + 8 + 4);
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&self.offset.to_le_bytes());
        buf.extend_from_slice(&self.size.to_le_bytes());
        buf.extend_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    /// Parse entry from bytes, returning entry and bytes consumed.
    pub fn from_bytes(buf: &[u8]) -> Result<(Self, usize)> {
        if buf.len() < 4 {
            return Err(OnnxError::InvalidModel("Model entry too small".into()));
        }

        let name_len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let min_size = 4 + name_len + 8 + 8 + 4;
        if buf.len() < min_size {
            return Err(OnnxError::InvalidModel(format!(
                "Model entry truncated: need {} bytes, have {}",
                min_size,
                buf.len()
            )));
        }

        let name = String::from_utf8(buf[4..4 + name_len].to_vec())
            .map_err(|e| OnnxError::InvalidModel(format!("Invalid model name: {}", e)))?;

        let offset_start = 4 + name_len;
        let offset = u64::from_le_bytes([
            buf[offset_start],
            buf[offset_start + 1],
            buf[offset_start + 2],
            buf[offset_start + 3],
            buf[offset_start + 4],
            buf[offset_start + 5],
            buf[offset_start + 6],
            buf[offset_start + 7],
        ]);

        let size_start = offset_start + 8;
        let size = u64::from_le_bytes([
            buf[size_start],
            buf[size_start + 1],
            buf[size_start + 2],
            buf[size_start + 3],
            buf[size_start + 4],
            buf[size_start + 5],
            buf[size_start + 6],
            buf[size_start + 7],
        ]);

        let checksum_start = size_start + 8;
        let checksum = u32::from_le_bytes([
            buf[checksum_start],
            buf[checksum_start + 1],
            buf[checksum_start + 2],
            buf[checksum_start + 3],
        ]);

        Ok((
            Self {
                name,
                offset,
                size,
                checksum,
            },
            min_size,
        ))
    }

    /// Calculate serialized size.
    pub fn serialized_size(&self) -> usize {
        4 + self.name.len() + 8 + 8 + 4
    }
}

/// Serializable metadata for the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoloMetadata {
    pub name: String,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<OutputSpec>,
    pub embedded_weight_size: u64,
    pub external_weight_size: u64,
    pub node_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<DimSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    pub node_id: usize,
    pub dtype: String,
    pub shape: Vec<DimSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DimSpec {
    Concrete(usize),
    Symbolic(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEntry {
    pub id: usize,
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: u64,
    pub size: usize,
    pub external: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackedWeightKind {
    MatMulRhs,
    Conv2dIm2Col,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedWeightEntry {
    pub id: usize,
    pub source_weight_id: usize,
    pub kind: PackedWeightKind,
    pub layout: String,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size: usize,
    pub external: bool,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_header_new() {
        let header = HoloBundleHeader::new(1000, 5000);

        assert_eq!(header.magic, HOLB_MAGIC);
        assert_eq!(header.version, BUNDLE_VERSION);
        assert_eq!(header.graph_offset, 64);
        assert_eq!(header.graph_size, 1000);
        assert_eq!(header.weights_size, 5000);
        // Weights offset should be page-aligned
        assert_eq!(header.weights_offset % PAGE_SIZE as u64, 0);
    }

    #[test]
    fn test_bundle_header_weights_offset_calculation() {
        // Graph ends at 64 + 1000 = 1064, next 4KB boundary is 4096
        assert_eq!(HoloBundleHeader::calculate_weights_offset(1000), 4096);

        // Graph ends at 64 + 4032 = 4096, already at boundary
        assert_eq!(HoloBundleHeader::calculate_weights_offset(4032), 4096);

        // Graph ends at 64 + 4033 = 4097, next boundary is 8192
        assert_eq!(HoloBundleHeader::calculate_weights_offset(4033), 8192);

        // Graph ends at 64 + 0 = 64, next boundary is 4096
        assert_eq!(HoloBundleHeader::calculate_weights_offset(0), 4096);

        // Large graph: ends at 64 + 100000 = 100064, next boundary is 102400
        assert_eq!(HoloBundleHeader::calculate_weights_offset(100000), 102400);
    }

    #[test]
    fn test_bundle_header_roundtrip() {
        let mut header = HoloBundleHeader::new(12345, 67890);
        header.set_checksums(0xDEADBEEF, 0xCAFEBABE);

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), BUNDLE_HEADER_SIZE);

        let parsed = HoloBundleHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn test_bundle_header_validation() {
        let header = HoloBundleHeader::new(1000, 5000);
        assert!(header.validate().is_ok());

        // Invalid magic
        let mut bad_header = header.clone();
        bad_header.magic = *b"BAAD";
        assert!(bad_header.validate().is_err());

        // Invalid version
        let mut bad_header = header.clone();
        bad_header.version = 99;
        assert!(bad_header.validate().is_err());

        // Invalid graph offset
        let mut bad_header = header.clone();
        bad_header.graph_offset = 128;
        assert!(bad_header.validate().is_err());

        // Non-page-aligned weights offset (with non-zero weights)
        let mut bad_header = header.clone();
        bad_header.weights_offset = 4097; // Not page-aligned
        assert!(bad_header.validate().is_err());
    }

    #[test]
    fn test_bundle_header_has_weights() {
        let header_with = HoloBundleHeader::new(1000, 5000);
        assert!(header_with.has_weights());

        let header_without = HoloBundleHeader::new(1000, 0);
        assert!(!header_without.has_weights());
    }

    #[test]
    fn test_bundle_header_from_bytes_too_small() {
        let small_buf = [0u8; 32];
        assert!(HoloBundleHeader::from_bytes(&small_buf).is_err());
    }

    #[test]
    fn test_bundle_header_from_bytes_wrong_magic() {
        let mut buf = [0u8; BUNDLE_HEADER_SIZE];
        buf[0..4].copy_from_slice(b"XXXX");
        assert!(HoloBundleHeader::from_bytes(&buf).is_err());
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(HoloFormat::detect(b"HOLB"), HoloFormat::Bundle);
        assert_eq!(HoloFormat::detect(b"HOLM"), HoloFormat::Pipeline);
        assert_eq!(HoloFormat::detect(b"HOLP"), HoloFormat::Plan);
        assert_eq!(HoloFormat::detect(b"HOLO"), HoloFormat::Legacy);
        assert_eq!(HoloFormat::detect(b"XXXX"), HoloFormat::Unknown);
        assert_eq!(HoloFormat::detect(b"HO"), HoloFormat::Unknown); // Too short
    }

    #[test]
    fn test_format_is_bundle() {
        assert!(HoloFormat::Bundle.is_bundle());
        assert!(!HoloFormat::Pipeline.is_bundle());
        assert!(!HoloFormat::Plan.is_bundle());
        assert!(!HoloFormat::Legacy.is_bundle());
        assert!(!HoloFormat::Unknown.is_bundle());
    }

    #[test]
    fn test_format_is_pipeline() {
        assert!(!HoloFormat::Bundle.is_pipeline());
        assert!(HoloFormat::Pipeline.is_pipeline());
        assert!(!HoloFormat::Plan.is_pipeline());
        assert!(!HoloFormat::Legacy.is_pipeline());
        assert!(!HoloFormat::Unknown.is_pipeline());
    }

    #[test]
    fn test_format_may_have_external_weights() {
        assert!(!HoloFormat::Bundle.may_have_external_weights());
        assert!(!HoloFormat::Pipeline.may_have_external_weights());
        assert!(HoloFormat::Plan.may_have_external_weights());
        assert!(HoloFormat::Legacy.may_have_external_weights());
        assert!(!HoloFormat::Unknown.may_have_external_weights());
    }

    // Pipeline header tests
    #[test]
    fn test_pipeline_header_new() {
        let header = HoloPipelineHeader::new(3, 100);

        assert_eq!(header.magic, HOLM_MAGIC);
        assert_eq!(header.version, PIPELINE_VERSION);
        assert_eq!(header.model_count, 3);
        assert_eq!(header.index_offset, 64);
        assert_eq!(header.index_size, 100);
        // Models offset should be page-aligned
        assert_eq!(header.models_offset % PAGE_SIZE as u64, 0);
    }

    #[test]
    fn test_pipeline_header_models_offset_calculation() {
        // Index ends at 64 + 100 = 164, next 4KB boundary is 4096
        assert_eq!(HoloPipelineHeader::calculate_models_offset(100), 4096);

        // Index ends at 64 + 4032 = 4096, already at boundary
        assert_eq!(HoloPipelineHeader::calculate_models_offset(4032), 4096);

        // Index ends at 64 + 4033 = 4097, next boundary is 8192
        assert_eq!(HoloPipelineHeader::calculate_models_offset(4033), 8192);
    }

    #[test]
    fn test_pipeline_header_roundtrip() {
        let mut header = HoloPipelineHeader::new(3, 256);
        header.set_models_total_size(300_000_000);
        header.set_index_checksum(0xDEADBEEF);

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), PIPELINE_HEADER_SIZE);

        let parsed = HoloPipelineHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn test_pipeline_header_validation() {
        let header = HoloPipelineHeader::new(3, 100);
        assert!(header.validate().is_ok());

        // Invalid magic
        let mut bad_header = header.clone();
        bad_header.magic = *b"BAAD";
        assert!(bad_header.validate().is_err());

        // Invalid version
        let mut bad_header = header.clone();
        bad_header.version = 99;
        assert!(bad_header.validate().is_err());

        // Invalid index offset
        let mut bad_header = header.clone();
        bad_header.index_offset = 128;
        assert!(bad_header.validate().is_err());

        // Non-page-aligned models offset
        let mut bad_header = header.clone();
        bad_header.models_offset = 4097;
        assert!(bad_header.validate().is_err());
    }

    #[test]
    fn test_pipeline_header_from_bytes_too_small() {
        let small_buf = [0u8; 32];
        assert!(HoloPipelineHeader::from_bytes(&small_buf).is_err());
    }

    #[test]
    fn test_pipeline_header_from_bytes_wrong_magic() {
        let mut buf = [0u8; PIPELINE_HEADER_SIZE];
        buf[0..4].copy_from_slice(b"XXXX");
        assert!(HoloPipelineHeader::from_bytes(&buf).is_err());
    }

    // Pipeline model entry tests
    #[test]
    fn test_pipeline_model_entry_roundtrip() {
        let entry = PipelineModelEntry::new("encoder".to_string(), 4096, 141_000_000, 0xCAFEBABE);

        let bytes = entry.to_bytes();
        let (parsed, consumed) = PipelineModelEntry::from_bytes(&bytes).unwrap();

        assert_eq!(parsed, entry);
        assert_eq!(consumed, bytes.len());
        assert_eq!(consumed, entry.serialized_size());
    }

    #[test]
    fn test_pipeline_model_entry_multiple() {
        let entries = vec![
            PipelineModelEntry::new("encoder".to_string(), 4096, 141_000_000, 0xAABBCCDD),
            PipelineModelEntry::new("decoder".to_string(), 145_000_000, 166_000_000, 0x11223344),
            PipelineModelEntry::new("tokenizer".to_string(), 311_000_000, 500_000, 0x55667788),
        ];

        // Serialize all entries
        let mut buf = Vec::new();
        for entry in &entries {
            buf.extend_from_slice(&entry.to_bytes());
        }

        // Parse them back
        let mut offset = 0;
        for expected in &entries {
            let (parsed, consumed) = PipelineModelEntry::from_bytes(&buf[offset..]).unwrap();
            assert_eq!(&parsed, expected);
            offset += consumed;
        }
        assert_eq!(offset, buf.len());
    }

    #[test]
    fn test_pipeline_model_entry_serialized_size() {
        let entry = PipelineModelEntry::new("encoder".to_string(), 0, 0, 0);
        // 4 (name_len) + 7 (name) + 8 (offset) + 8 (size) + 4 (checksum) = 31
        assert_eq!(entry.serialized_size(), 31);

        let long_name = PipelineModelEntry::new("very_long_model_name".to_string(), 0, 0, 0);
        // 4 + 20 + 8 + 8 + 4 = 44
        assert_eq!(long_name.serialized_size(), 44);
    }

    #[test]
    fn test_pipeline_model_entry_from_bytes_too_small() {
        let small_buf = [0u8; 2];
        assert!(PipelineModelEntry::from_bytes(&small_buf).is_err());
    }

    #[test]
    fn test_pipeline_model_entry_truncated() {
        // Valid header but truncated data
        let entry = PipelineModelEntry::new("test".to_string(), 0, 0, 0);
        let bytes = entry.to_bytes();
        let truncated = &bytes[..bytes.len() - 2];
        assert!(PipelineModelEntry::from_bytes(truncated).is_err());
    }

    #[test]
    fn test_holo_header_roundtrip() {
        let header = HoloHeader {
            version: 1,
            flags: FLAG_EXTERNAL_WEIGHTS,
            metadata_offset: 100,
            graph_offset: 200,
            weights_offset: 300,
        };

        let bytes = header.to_bytes();
        let parsed = HoloHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.flags, header.flags);
        assert_eq!(parsed.metadata_offset, header.metadata_offset);
        assert_eq!(parsed.graph_offset, header.graph_offset);
        assert_eq!(parsed.weights_offset, header.weights_offset);
    }

    #[test]
    fn test_holo_header_has_external_weights() {
        let with_external = HoloHeader {
            version: 1,
            flags: FLAG_EXTERNAL_WEIGHTS,
            metadata_offset: 0,
            graph_offset: 0,
            weights_offset: 0,
        };
        assert!(with_external.has_external_weights());

        let without_external = HoloHeader {
            version: 1,
            flags: 0,
            metadata_offset: 0,
            graph_offset: 0,
            weights_offset: 0,
        };
        assert!(!without_external.has_external_weights());
    }
}
