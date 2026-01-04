#![allow(missing_docs)]
//! Serialization stubs for .holo format.
//!
//! STUB MODULE: This module contains only type definitions and constants.
//! The actual serialization functionality using old IR types has been removed
//! in the simplified version. This needs to be reimplemented using the new
//! hologram-ir types.

use serde::{Deserialize, Serialize};
use crate::{OnnxError, Result};

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
pub const FLAG_COMPRESSED: u32 = 0x02;

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
// Stub Implementations (not functional in simplified version)
// =============================================================================

/// Serializer for .holo format (STUB - not implemented)
pub struct HoloSerializer {
    _phantom: std::marker::PhantomData<()>,
}

impl HoloSerializer {
    /// Create a new serializer (STUB)
    pub fn new(_weight_threshold: usize, _pack_weights: bool) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Serialize an operation graph (STUB - not implemented)
    pub fn serialize(&mut self, _func: &hologram_ir::OperationGraph) -> Result<(Vec<u8>, Vec<u8>)> {
        Err(OnnxError::SerializationError(
            "Serialization is not implemented in simplified version. \
             This needs to be reimplemented using hologram-ir types.".to_string()
        ))
    }
}
