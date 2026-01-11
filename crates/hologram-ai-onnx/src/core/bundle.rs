//! Multi-model .holo bundle format.
//!
//! This module provides support for bundling multiple compiled models
//! into a single .holo file for easier distribution and deployment.
//!
//! # Bundle Format (v2)
//!
//! ```text
//! +------------------+
//! | Header (24 bytes)|
//! +------------------+
//! | Model Index      |
//! +------------------+
//! | Model Data       |
//! +------------------+
//! | Shared Weights   |
//! +------------------+
//! ```
//!
//! ## Header
//! - Magic: "HOLO" (4 bytes)
//! - Version: 2 (u32, 4 bytes)
//! - Flags: u32 (4 bytes) - reserved for future use
//! - Model count: u32 (4 bytes)
//! - Index offset: u64 (8 bytes)
//!
//! ## Model Index Entry (per model)
//! - Name length: u32
//! - Name: bytes
//! - Data offset: u64
//! - Data size: u64
//! - Checksum: u32 (CRC32)
//!
//! # Example
//!
//! ```rust,ignore
//! use hologram_ai_onnx::core::bundle::{HoloBundle, BundleBuilder};
//!
//! // Create a bundle
//! let mut builder = BundleBuilder::new();
//! builder.add_model("text_encoder", &text_encoder_bytes)?;
//! builder.add_model("unet", &unet_bytes)?;
//! builder.add_model("vae_decoder", &vae_decoder_bytes)?;
//! let bundle = builder.build()?;
//! bundle.write_to_file("sd-pipeline.holo")?;
//!
//! // Read a bundle
//! let bundle = HoloBundle::from_file("sd-pipeline.holo")?;
//! let text_encoder = bundle.get_model("text_encoder")?;
//! ```

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::{OnnxError, Result};

/// Create an IO error with a custom message.
fn io_error(msg: impl Into<String>) -> OnnxError {
    OnnxError::IoError(io::Error::other(msg.into()))
}

// =============================================================================
// Constants
// =============================================================================

/// Magic bytes identifying a .holo file
pub const HOLO_MAGIC: &[u8; 4] = b"HOLO";

/// Version 1: Single model format
pub const VERSION_SINGLE: u32 = 1;

/// Version 2: Multi-model bundle format
pub const VERSION_BUNDLE: u32 = 2;

/// Header size in bytes
pub const HEADER_SIZE: usize = 24;

// =============================================================================
// Bundle Header
// =============================================================================

/// Header for .holo bundle files.
#[derive(Debug, Clone)]
pub struct BundleHeader {
    /// Format version (1 = single model, 2 = bundle)
    pub version: u32,
    /// Flags (reserved for future use)
    pub flags: u32,
    /// Number of models in the bundle
    pub model_count: u32,
    /// Offset to model index (from start of file)
    pub index_offset: u64,
}

impl BundleHeader {
    /// Create a new bundle header.
    pub fn new(model_count: u32, index_offset: u64) -> Self {
        Self {
            version: VERSION_BUNDLE,
            flags: 0,
            model_count,
            index_offset,
        }
    }

    /// Read header from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(OnnxError::InvalidModel("Header too small".to_string()));
        }

        // Check magic
        if &bytes[0..4] != HOLO_MAGIC {
            return Err(OnnxError::InvalidModel(
                "Invalid magic bytes - not a .holo file".to_string(),
            ));
        }

        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let flags = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let model_count = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let index_offset = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);

        Ok(Self {
            version,
            flags,
            model_count,
            index_offset,
        })
    }

    /// Write header to bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(HOLO_MAGIC);
        bytes[4..8].copy_from_slice(&self.version.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.flags.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.model_count.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.index_offset.to_le_bytes());
        bytes
    }

    /// Check if this is a bundle (v2) or single model (v1).
    pub fn is_bundle(&self) -> bool {
        self.version == VERSION_BUNDLE
    }
}

// =============================================================================
// Model Index Entry
// =============================================================================

/// Index entry for a model in the bundle.
#[derive(Debug, Clone)]
pub struct ModelIndexEntry {
    /// Model name (e.g., "text_encoder", "unet")
    pub name: String,
    /// Offset to model data from start of file
    pub data_offset: u64,
    /// Size of model data in bytes
    pub data_size: u64,
    /// CRC32 checksum of model data
    pub checksum: u32,
}

impl ModelIndexEntry {
    /// Create a new index entry.
    pub fn new(name: String, data_offset: u64, data_size: u64, checksum: u32) -> Self {
        Self {
            name,
            data_offset,
            data_size,
            checksum,
        }
    }

    /// Calculate the serialized size of this entry.
    pub fn serialized_size(&self) -> usize {
        4 + self.name.len() + 8 + 8 + 4 // name_len + name + offset + size + checksum
    }

    /// Write entry to a writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let name_bytes = self.name.as_bytes();
        writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(name_bytes)?;
        writer.write_all(&self.data_offset.to_le_bytes())?;
        writer.write_all(&self.data_size.to_le_bytes())?;
        writer.write_all(&self.checksum.to_le_bytes())?;
        Ok(())
    }

    /// Read entry from a reader.
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        // Read name length and name
        reader.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Read offset, size, checksum
        reader.read_exact(&mut buf8)?;
        let data_offset = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf8)?;
        let data_size = u64::from_le_bytes(buf8);
        reader.read_exact(&mut buf4)?;
        let checksum = u32::from_le_bytes(buf4);

        Ok(Self {
            name,
            data_offset,
            data_size,
            checksum,
        })
    }
}

// =============================================================================
// Bundle Builder
// =============================================================================

/// Builder for creating .holo bundles.
#[derive(Debug, Default)]
pub struct BundleBuilder {
    /// Models to include (name -> data)
    models: Vec<(String, Vec<u8>)>,
}

impl BundleBuilder {
    /// Create a new bundle builder.
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Add a model to the bundle.
    pub fn add_model(&mut self, name: impl Into<String>, data: Vec<u8>) -> &mut Self {
        self.models.push((name.into(), data));
        self
    }

    /// Add a model from a file.
    pub fn add_model_from_file(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<&mut Self> {
        let data = fs::read(path.as_ref())
            .map_err(|e| io_error(format!("Failed to read model file: {}", e)))?;
        self.models.push((name.into(), data));
        Ok(self)
    }

    /// Get the number of models in the builder.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Build the bundle.
    pub fn build(self) -> Result<HoloBundle> {
        if self.models.is_empty() {
            return Err(OnnxError::InvalidModel(
                "Cannot create empty bundle".to_string(),
            ));
        }

        // Calculate offsets
        // Layout: Header | Model Data | Index
        let mut current_offset = HEADER_SIZE as u64;
        let mut entries = Vec::new();
        let mut model_data = Vec::new();

        for (name, data) in self.models {
            let checksum = crc32_checksum(&data);
            let data_size = data.len() as u64;

            entries.push(ModelIndexEntry::new(
                name,
                current_offset,
                data_size,
                checksum,
            ));

            model_data.push(data);
            current_offset += data_size;
        }

        let index_offset = current_offset;
        let header = BundleHeader::new(entries.len() as u32, index_offset);

        Ok(HoloBundle {
            header,
            entries,
            model_data,
        })
    }
}

// =============================================================================
// Holo Bundle
// =============================================================================

/// A .holo bundle containing multiple models.
#[derive(Debug)]
pub struct HoloBundle {
    /// Bundle header
    pub header: BundleHeader,
    /// Model index entries
    pub entries: Vec<ModelIndexEntry>,
    /// Model data (parallel to entries)
    model_data: Vec<Vec<u8>>,
}

impl HoloBundle {
    /// Read a bundle from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).map_err(|e| io_error(format!("Failed to open bundle file: {}", e)))?;
        let mut reader = BufReader::new(file);
        Self::from_reader(&mut reader)
    }

    /// Read a bundle from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = io::Cursor::new(bytes);
        Self::from_reader(&mut cursor)
    }

    /// Read a bundle from a reader.
    pub fn from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| io_error(format!("Failed to read header: {}", e)))?;
        let header = BundleHeader::from_bytes(&header_bytes)?;

        // Check if this is a v1 single model file
        if header.version == VERSION_SINGLE {
            return Self::read_v1_as_bundle(reader, &header_bytes);
        }

        if header.version != VERSION_BUNDLE {
            return Err(OnnxError::InvalidModel(format!(
                "Unsupported .holo version: {}",
                header.version
            )));
        }

        // Seek to index
        reader
            .seek(SeekFrom::Start(header.index_offset))
            .map_err(|e| io_error(format!("Failed to seek to index: {}", e)))?;

        // Read index entries
        let mut entries = Vec::with_capacity(header.model_count as usize);
        for _ in 0..header.model_count {
            let entry = ModelIndexEntry::read_from(reader)
                .map_err(|e| io_error(format!("Failed to read index entry: {}", e)))?;
            entries.push(entry);
        }

        // Read model data
        let mut model_data = Vec::with_capacity(entries.len());
        for entry in &entries {
            reader
                .seek(SeekFrom::Start(entry.data_offset))
                .map_err(|e| io_error(format!("Failed to seek to model data: {}", e)))?;

            let mut data = vec![0u8; entry.data_size as usize];
            reader
                .read_exact(&mut data)
                .map_err(|e| io_error(format!("Failed to read model data: {}", e)))?;

            // Verify checksum
            let actual_checksum = crc32_checksum(&data);
            if actual_checksum != entry.checksum {
                return Err(OnnxError::InvalidModel(format!(
                    "Checksum mismatch for model '{}': expected {:08x}, got {:08x}",
                    entry.name, entry.checksum, actual_checksum
                )));
            }

            model_data.push(data);
        }

        Ok(Self {
            header,
            entries,
            model_data,
        })
    }

    /// Read a v1 single-model file as a bundle with one model.
    fn read_v1_as_bundle<R: Read + Seek>(reader: &mut R, header_bytes: &[u8]) -> Result<Self> {
        // Seek back to start
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|e| io_error(format!("Failed to seek: {}", e)))?;

        // Read entire file
        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|e| io_error(format!("Failed to read v1 model: {}", e)))?;

        // Extract model name from v1 format
        // v1 format: HOLO (4) | version (4) | name_len (4) | name | ...
        let name_len = u32::from_le_bytes([
            header_bytes[8],
            header_bytes[9],
            header_bytes[10],
            header_bytes[11],
        ]) as usize;

        let name = if name_len > 0 && name_len < 256 {
            String::from_utf8_lossy(&data[12..12 + name_len]).to_string()
        } else {
            "model".to_string()
        };

        let checksum = crc32_checksum(&data);

        let header = BundleHeader {
            version: VERSION_BUNDLE,
            flags: 0,
            model_count: 1,
            index_offset: 0, // Not used for in-memory bundle
        };

        let entry = ModelIndexEntry::new(name, 0, data.len() as u64, checksum);

        Ok(Self {
            header,
            entries: vec![entry],
            model_data: vec![data],
        })
    }

    /// Write bundle to a file.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = File::create(path)
            .map_err(|e| io_error(format!("Failed to create bundle file: {}", e)))?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }

    /// Write bundle to a writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write header
        writer
            .write_all(&self.header.to_bytes())
            .map_err(|e| io_error(format!("Failed to write header: {}", e)))?;

        // Write model data
        for data in &self.model_data {
            writer
                .write_all(data)
                .map_err(|e| io_error(format!("Failed to write model data: {}", e)))?;
        }

        // Write index
        for entry in &self.entries {
            entry
                .write_to(writer)
                .map_err(|e| io_error(format!("Failed to write index entry: {}", e)))?;
        }

        writer
            .flush()
            .map_err(|e| io_error(format!("Failed to flush writer: {}", e)))?;

        Ok(())
    }

    /// Get the list of model names in the bundle.
    pub fn model_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// Check if the bundle contains a model.
    pub fn contains_model(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// Get model data by name.
    pub fn get_model(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .position(|e| e.name == name)
            .map(|idx| self.model_data[idx].as_slice())
    }

    /// Get model data by index.
    pub fn get_model_by_index(&self, index: usize) -> Option<&[u8]> {
        self.model_data.get(index).map(|v| v.as_slice())
    }

    /// Get model index entry by name.
    pub fn get_entry(&self, name: &str) -> Option<&ModelIndexEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Get the number of models in the bundle.
    pub fn model_count(&self) -> usize {
        self.entries.len()
    }

    /// Get total size of all model data.
    pub fn total_data_size(&self) -> u64 {
        self.entries.iter().map(|e| e.data_size).sum()
    }

    /// Extract all models to a directory.
    pub fn extract_to_dir(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)
            .map_err(|e| io_error(format!("Failed to create directory: {}", e)))?;

        for (entry, data) in self.entries.iter().zip(self.model_data.iter()) {
            let path = dir.join(format!("{}.holo", entry.name));
            fs::write(&path, data)
                .map_err(|e| io_error(format!("Failed to write {}: {}", path.display(), e)))?;
        }

        Ok(())
    }

    /// Create a bundle from a directory of .holo files.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut builder = BundleBuilder::new();

        for entry in
            fs::read_dir(dir).map_err(|e| io_error(format!("Failed to read directory: {}", e)))?
        {
            let entry =
                entry.map_err(|e| io_error(format!("Failed to read directory entry: {}", e)))?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "holo") {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
                    .to_string();
                builder.add_model_from_file(name, &path)?;
            }
        }

        builder.build()
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Calculate CRC32 checksum of data.
fn crc32_checksum(data: &[u8]) -> u32 {
    // Simple CRC32 implementation (IEEE polynomial)
    const CRC32_TABLE: [u32; 256] = generate_crc32_table();

    let mut crc = 0xFFFFFFFF_u32;
    for byte in data {
        let index = ((crc ^ (*byte as u32)) & 0xFF) as usize;
        crc = CRC32_TABLE[index] ^ (crc >> 8);
    }
    !crc
}

/// Generate CRC32 lookup table at compile time.
const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let polynomial = 0xEDB88320_u32;
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ polynomial;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Check if a file is a .holo bundle (v2) or single model (v1).
pub fn is_bundle(path: impl AsRef<Path>) -> Result<bool> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|e| io_error(format!("Failed to open file: {}", e)))?;

    let mut header_bytes = [0u8; HEADER_SIZE];
    file.read_exact(&mut header_bytes)
        .map_err(|e| io_error(format!("Failed to read header: {}", e)))?;

    let header = BundleHeader::from_bytes(&header_bytes)?;
    Ok(header.is_bundle())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_crc32_checksum() {
        // Test vectors
        assert_eq!(crc32_checksum(b""), 0x00000000);
        assert_eq!(crc32_checksum(b"123456789"), 0xCBF43926);
        assert_eq!(crc32_checksum(b"hello"), 0x3610A686);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = BundleHeader::new(3, 12345);
        let bytes = header.to_bytes();
        let parsed = BundleHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, VERSION_BUNDLE);
        assert_eq!(parsed.model_count, 3);
        assert_eq!(parsed.index_offset, 12345);
    }

    #[test]
    fn test_index_entry_roundtrip() {
        let entry = ModelIndexEntry::new("text_encoder".to_string(), 1000, 5000, 0x12345678);

        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();

        let mut cursor = io::Cursor::new(buf);
        let parsed = ModelIndexEntry::read_from(&mut cursor).unwrap();

        assert_eq!(parsed.name, "text_encoder");
        assert_eq!(parsed.data_offset, 1000);
        assert_eq!(parsed.data_size, 5000);
        assert_eq!(parsed.checksum, 0x12345678);
    }

    #[test]
    fn test_bundle_builder() {
        let mut builder = BundleBuilder::new();
        builder.add_model("model1", vec![1, 2, 3, 4]);
        builder.add_model("model2", vec![5, 6, 7, 8, 9]);

        let bundle = builder.build().unwrap();
        assert_eq!(bundle.model_count(), 2);
        assert_eq!(bundle.model_names(), vec!["model1", "model2"]);
    }

    #[test]
    fn test_bundle_roundtrip() {
        let mut builder = BundleBuilder::new();
        builder.add_model("encoder", vec![1, 2, 3, 4, 5]);
        builder.add_model("decoder", vec![10, 20, 30]);
        let bundle = builder.build().unwrap();

        // Write to bytes
        let mut buf = Vec::new();
        bundle.write_to(&mut buf).unwrap();

        // Read back
        let parsed = HoloBundle::from_bytes(&buf).unwrap();
        assert_eq!(parsed.model_count(), 2);
        assert_eq!(parsed.get_model("encoder"), Some(&[1u8, 2, 3, 4, 5][..]));
        assert_eq!(parsed.get_model("decoder"), Some(&[10u8, 20, 30][..]));
    }

    #[test]
    fn test_bundle_file_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let bundle_path = temp_dir.path().join("test.holo");

        // Create and write bundle
        let mut builder = BundleBuilder::new();
        builder.add_model("model_a", b"hello world".to_vec());
        builder.add_model("model_b", b"goodbye".to_vec());
        let bundle = builder.build().unwrap();
        bundle.write_to_file(&bundle_path).unwrap();

        // Read back
        let parsed = HoloBundle::from_file(&bundle_path).unwrap();
        assert_eq!(parsed.model_count(), 2);
        assert!(parsed.contains_model("model_a"));
        assert!(parsed.contains_model("model_b"));
        assert_eq!(parsed.get_model("model_a"), Some(b"hello world".as_slice()));
    }

    #[test]
    fn test_bundle_extract() {
        let temp_dir = TempDir::new().unwrap();
        let extract_dir = temp_dir.path().join("extracted");

        let mut builder = BundleBuilder::new();
        builder.add_model("model1", vec![1, 2, 3]);
        builder.add_model("model2", vec![4, 5, 6]);
        let bundle = builder.build().unwrap();

        bundle.extract_to_dir(&extract_dir).unwrap();

        assert!(extract_dir.join("model1.holo").exists());
        assert!(extract_dir.join("model2.holo").exists());
    }

    #[test]
    fn test_bundle_from_dir() {
        let temp_dir = TempDir::new().unwrap();
        let dir = temp_dir.path();

        // Create some .holo files
        fs::write(dir.join("encoder.holo"), b"encoder data").unwrap();
        fs::write(dir.join("decoder.holo"), b"decoder data").unwrap();
        fs::write(dir.join("other.txt"), b"not a holo file").unwrap();

        let bundle = HoloBundle::from_dir(dir).unwrap();
        assert_eq!(bundle.model_count(), 2);
        assert!(bundle.contains_model("encoder"));
        assert!(bundle.contains_model("decoder"));
    }

    #[test]
    fn test_empty_bundle_error() {
        let builder = BundleBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_checksum_verification() {
        let mut builder = BundleBuilder::new();
        builder.add_model("test", vec![1, 2, 3, 4]);
        let bundle = builder.build().unwrap();

        // Write to bytes
        let mut buf = Vec::new();
        bundle.write_to(&mut buf).unwrap();

        // Corrupt the data
        if buf.len() > HEADER_SIZE + 2 {
            buf[HEADER_SIZE + 2] ^= 0xFF;
        }

        // Should fail checksum verification
        let result = HoloBundle::from_bytes(&buf);
        assert!(result.is_err());
    }
}

// =============================================================================
// Unified Bundle Format (HOLB) - Single model with embedded weights
// =============================================================================
//
// This format is for single-model bundles that combine the computation graph
// and weights into a single file with page-aligned weights for efficient mmap.
//
// Layout:
// +================================+
// |  Bundle Header (64 bytes)      |  Magic: "HOLB", offsets, checksums
// +================================+
// |  Graph Section (HOLP data)     |  Existing hologram format bytes
// +--------------------------------+
// |  Padding to 4KB boundary       |
// +================================+
// |  Weights Section               |  Page-aligned for mmap
// +================================+

use crate::core::serialization::{
    BUNDLE_HEADER_SIZE, HoloBundleHeader, HoloBundleHeaderV2, HoloFormat, SectionTableEntry,
    serialize_sections_table,
};

/// Internal section data container.
#[derive(Debug)]
struct SectionData {
    /// Section ID (e.g., "vocabulary", "tokenizer_config")
    id: String,
    /// Content type (e.g., "text/plain", "application/json")
    content_type: String,
    /// Section data bytes
    data: Vec<u8>,
}

/// Writer for creating unified bundle files (HOLB format).
///
/// The writer accumulates graph, weight bytes, and optional sections,
/// then produces a properly formatted bundle with page-aligned sections.
///
/// If sections are added, produces a V2 bundle with embedded sections.
/// Otherwise, produces a V1 bundle for backward compatibility.
#[derive(Debug, Default)]
pub struct UnifiedBundleWriter {
    graph_bytes: Vec<u8>,
    weights_bytes: Vec<u8>,
    sections: Vec<SectionData>,
}

impl UnifiedBundleWriter {
    /// Create a new unified bundle writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the graph section bytes (HOLP format data).
    pub fn set_graph_bytes(&mut self, bytes: Vec<u8>) {
        self.graph_bytes = bytes;
    }

    /// Set the weights section bytes.
    pub fn set_weights_bytes(&mut self, bytes: Vec<u8>) {
        self.weights_bytes = bytes;
    }

    /// Get the graph bytes.
    pub fn graph_bytes(&self) -> &[u8] {
        &self.graph_bytes
    }

    /// Get the weights bytes.
    pub fn weights_bytes(&self) -> &[u8] {
        &self.weights_bytes
    }

    /// Add a raw section with explicit ID and content type.
    ///
    /// This is the low-level method for adding sections. For typed sections,
    /// use `add_section` instead.
    ///
    /// # Arguments
    ///
    /// * `id` - Section identifier (e.g., "vocabulary")
    /// * `content_type` - MIME content type (e.g., "text/plain")
    /// * `data` - Section data bytes
    pub fn add_raw_section(&mut self, id: &str, content_type: &str, data: Vec<u8>) {
        self.sections.push(SectionData {
            id: id.to_string(),
            content_type: content_type.to_string(),
            data,
        });
    }

    /// Check if a section with the given ID exists.
    pub fn has_section(&self, id: &str) -> bool {
        self.sections.iter().any(|s| s.id == id)
    }

    /// Get the number of sections.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Check if this will produce a V2 bundle (has sections).
    pub fn is_v2(&self) -> bool {
        !self.sections.is_empty()
    }

    /// Calculate the total bundle size including padding.
    pub fn total_size(&self) -> usize {
        let header_size = BUNDLE_HEADER_SIZE;
        let graph_size = self.graph_bytes.len();
        let weights_offset = HoloBundleHeader::calculate_weights_offset(graph_size as u64) as usize;
        let weights_size = self.weights_bytes.len();

        if weights_size > 0 {
            weights_offset + weights_size
        } else {
            header_size + graph_size
        }
    }

    /// Finish writing and produce the bundle bytes.
    ///
    /// This creates the final bundle with:
    /// - Header with checksums (V1 or V2 depending on sections)
    /// - Graph section
    /// - Sections table and data (V2 only)
    /// - Padding to page boundary
    /// - Weights section (if present)
    pub fn finish(self) -> Vec<u8> {
        if self.sections.is_empty() {
            self.finish_v1()
        } else {
            self.finish_v2()
        }
    }

    /// Produce a V1 bundle (no sections).
    fn finish_v1(self) -> Vec<u8> {
        let graph_size = self.graph_bytes.len() as u64;
        let weights_size = self.weights_bytes.len() as u64;

        // Create header
        let mut header = HoloBundleHeader::new(graph_size, weights_size);

        // Calculate checksums
        let graph_checksum = crc32_checksum(&self.graph_bytes);
        let weights_checksum = if weights_size > 0 {
            crc32_checksum(&self.weights_bytes)
        } else {
            0
        };
        header.set_checksums(graph_checksum, weights_checksum);

        // Calculate total size and allocate
        let total_size = self.total_size();
        let mut output = Vec::with_capacity(total_size);

        // Write header
        output.extend_from_slice(&header.to_bytes());

        // Write graph
        output.extend_from_slice(&self.graph_bytes);

        // Write padding to page boundary (if we have weights)
        if weights_size > 0 {
            let weights_offset = header.weights_offset as usize;
            output.resize(weights_offset, 0);

            // Write weights
            output.extend_from_slice(&self.weights_bytes);
        }

        output
    }

    /// Produce a V2 bundle with sections.
    fn finish_v2(self) -> Vec<u8> {
        let graph_size = self.graph_bytes.len() as u64;
        let weights_size = self.weights_bytes.len() as u64;
        let sections_count = self.sections.len() as u32;

        // Build section table entries and concatenate section data
        let mut section_entries = Vec::with_capacity(self.sections.len());
        let mut sections_data = Vec::new();
        let mut section_offset = 0u64;

        for section in &self.sections {
            let checksum = crc32_checksum(&section.data);
            section_entries.push(SectionTableEntry::new(
                section.id.clone(),
                section.content_type.clone(),
                1, // version
                section_offset,
                section.data.len() as u64,
                checksum,
            ));
            sections_data.extend_from_slice(&section.data);
            section_offset += section.data.len() as u64;
        }

        // Serialize sections table
        let sections_table = serialize_sections_table(&section_entries);
        let sections_table_size = sections_table.len() as u64;
        let sections_data_size = sections_data.len() as u64;

        // Create V2 header
        let mut header = HoloBundleHeaderV2::new(
            graph_size,
            sections_table_size,
            sections_data_size,
            weights_size,
            sections_count,
        );

        // Calculate checksums
        let graph_checksum = crc32_checksum(&self.graph_bytes);
        let weights_checksum = if weights_size > 0 {
            crc32_checksum(&self.weights_bytes)
        } else {
            0
        };
        header.set_checksums(graph_checksum, weights_checksum);

        // Allocate output buffer
        let mut output = Vec::new();

        // Write V2 header (80 bytes)
        output.extend_from_slice(&header.to_bytes());

        // Write graph
        output.extend_from_slice(&self.graph_bytes);

        // Pad to page boundary for sections table
        let sections_table_offset = header.sections_table_offset as usize;
        if sections_table_offset > output.len() {
            output.resize(sections_table_offset, 0);
        }

        // Write sections table
        output.extend_from_slice(&sections_table);

        // Write sections data
        output.extend_from_slice(&sections_data);

        // Pad to page boundary for weights
        if weights_size > 0 {
            let weights_offset = header.weights_offset as usize;
            if weights_offset > output.len() {
                output.resize(weights_offset, 0);
            }
            output.extend_from_slice(&self.weights_bytes);
        }

        output
    }

    /// Finish and write directly to a file.
    pub fn write_to_file(self, path: &Path) -> Result<usize> {
        let bundle = self.finish();
        let size = bundle.len();

        let mut file = File::create(path)
            .map_err(|e| io_error(format!("Failed to create unified bundle file: {}", e)))?;

        file.write_all(&bundle)
            .map_err(|e| io_error(format!("Failed to write unified bundle: {}", e)))?;

        Ok(size)
    }
}

/// Reader for unified bundle files (HOLB format).
///
/// Provides zero-copy access to bundle sections. Can work with borrowed bytes
/// or memory-mapped files.
#[derive(Debug)]
pub struct UnifiedBundleReader<'a> {
    header: HoloBundleHeader,
    data: &'a [u8],
}

impl<'a> UnifiedBundleReader<'a> {
    /// Create a reader from a byte slice.
    ///
    /// Parses and validates the header. The data slice must remain valid
    /// for the lifetime of the reader.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self> {
        if data.len() < BUNDLE_HEADER_SIZE {
            return Err(OnnxError::InvalidModel(
                "Unified bundle too small for header".into(),
            ));
        }

        // Check format
        let format = HoloFormat::detect(&data[0..4]);
        if !format.is_bundle() {
            return Err(OnnxError::InvalidModel(format!(
                "Not a unified bundle file. Detected format: {:?}",
                format
            )));
        }

        // Parse header
        let header = HoloBundleHeader::from_bytes(data)?;
        header.validate()?;

        // Validate data size
        let required_size = if header.has_weights() {
            header.weights_offset as usize + header.weights_size as usize
        } else {
            header.graph_offset as usize + header.graph_size as usize
        };

        if data.len() < required_size {
            return Err(OnnxError::InvalidModel(format!(
                "Unified bundle truncated: need {} bytes, have {}",
                required_size,
                data.len()
            )));
        }

        Ok(Self { header, data })
    }

    /// Get the bundle header.
    pub fn header(&self) -> &HoloBundleHeader {
        &self.header
    }

    /// Get the graph section bytes.
    pub fn graph_bytes(&self) -> &'a [u8] {
        let start = self.header.graph_offset as usize;
        let end = start + self.header.graph_size as usize;
        &self.data[start..end]
    }

    /// Get the weights section bytes.
    ///
    /// Returns an empty slice if no weights are present.
    pub fn weights_bytes(&self) -> &'a [u8] {
        if !self.header.has_weights() {
            return &[];
        }
        let start = self.header.weights_offset as usize;
        let end = start + self.header.weights_size as usize;
        &self.data[start..end]
    }

    /// Get the offset to the weights section for memory-mapping.
    ///
    /// This offset can be used with `PlanExecutor::with_mmap_constants_at_offset`
    /// to create an executor that reads weights directly from the mmap'd bundle.
    ///
    /// Returns `None` if no weights are present.
    pub fn weights_mmap_offset(&self) -> Option<usize> {
        if self.header.has_weights() {
            Some(self.header.weights_offset as usize)
        } else {
            None
        }
    }

    /// Verify the graph section checksum.
    pub fn verify_graph_checksum(&self) -> bool {
        let actual = crc32_checksum(self.graph_bytes());
        actual == self.header.graph_checksum
    }

    /// Verify the weights section checksum.
    ///
    /// Returns `true` if no weights are present.
    pub fn verify_weights_checksum(&self) -> bool {
        if !self.header.has_weights() {
            return true;
        }
        let actual = crc32_checksum(self.weights_bytes());
        actual == self.header.weights_checksum
    }

    /// Verify all checksums.
    pub fn verify_checksums(&self) -> bool {
        self.verify_graph_checksum() && self.verify_weights_checksum()
    }

    /// Get the total bundle size.
    pub fn total_size(&self) -> usize {
        self.data.len()
    }

    /// Get the graph size.
    pub fn graph_size(&self) -> usize {
        self.header.graph_size as usize
    }

    /// Get the weights size.
    pub fn weights_size(&self) -> usize {
        self.header.weights_size as usize
    }
}

/// Load a unified bundle from a file path.
///
/// This reads the entire file into memory. For large files, consider
/// memory-mapping instead.
pub fn read_unified_bundle_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|e| {
        io_error(format!(
            "Failed to read unified bundle file '{}': {}",
            path.display(),
            e
        ))
    })
}

// =============================================================================
// Unified Bundle Tests
// =============================================================================

#[cfg(test)]
mod unified_bundle_tests {
    use super::*;
    use crate::core::serialization::PAGE_SIZE;

    #[test]
    fn test_unified_writer_empty() {
        let writer = UnifiedBundleWriter::new();
        let bundle = writer.finish();

        // Should have just header (64 bytes)
        assert_eq!(bundle.len(), BUNDLE_HEADER_SIZE);

        // Verify header
        let header = HoloBundleHeader::from_bytes(&bundle).unwrap();
        assert_eq!(header.graph_size, 0);
        assert_eq!(header.weights_size, 0);
    }

    #[test]
    fn test_unified_writer_graph_only() {
        let mut writer = UnifiedBundleWriter::new();
        let graph = b"test graph data";
        writer.set_graph_bytes(graph.to_vec());
        let bundle = writer.finish();

        // Should have header + graph
        assert_eq!(bundle.len(), BUNDLE_HEADER_SIZE + graph.len());

        // Verify we can read it back
        let reader = UnifiedBundleReader::from_bytes(&bundle).unwrap();
        assert_eq!(reader.graph_bytes(), graph);
        assert!(reader.weights_bytes().is_empty());
    }

    #[test]
    fn test_unified_writer_with_weights() {
        let mut writer = UnifiedBundleWriter::new();
        let graph = b"test graph data for the model";
        let weights = b"weight data here - pretend this is big";

        writer.set_graph_bytes(graph.to_vec());
        writer.set_weights_bytes(weights.to_vec());
        let bundle = writer.finish();

        // Verify structure
        let reader = UnifiedBundleReader::from_bytes(&bundle).unwrap();
        assert_eq!(reader.graph_bytes(), graph);
        assert_eq!(reader.weights_bytes(), weights);

        // Verify weights are page-aligned
        let weights_offset = reader.weights_mmap_offset().unwrap();
        assert_eq!(weights_offset % PAGE_SIZE, 0);

        // Verify checksums
        assert!(reader.verify_checksums());
    }

    #[test]
    fn test_unified_reader_invalid_magic() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"XXXX");

        let result = UnifiedBundleReader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_unified_reader_too_small() {
        let data = vec![0u8; 32]; // Less than header size

        let result = UnifiedBundleReader::from_bytes(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_unified_reader_truncated() {
        // Create a valid bundle
        let mut writer = UnifiedBundleWriter::new();
        writer.set_graph_bytes(vec![1, 2, 3, 4, 5]);
        let bundle = writer.finish();

        // Truncate it
        let truncated = &bundle[..bundle.len() - 2];

        let result = UnifiedBundleReader::from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_unified_roundtrip_large_graph() {
        let mut writer = UnifiedBundleWriter::new();

        // Create graph that spans multiple pages
        let graph: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        writer.set_graph_bytes(graph.clone());

        let bundle = writer.finish();
        let reader = UnifiedBundleReader::from_bytes(&bundle).unwrap();

        assert_eq!(reader.graph_bytes(), &graph[..]);
        assert!(reader.verify_checksums());
    }

    #[test]
    fn test_unified_roundtrip_with_weights() {
        let mut writer = UnifiedBundleWriter::new();

        // Simulate real model data
        let graph: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        let weights: Vec<u8> = (0..100000).map(|i| ((i * 7) % 256) as u8).collect();

        writer.set_graph_bytes(graph.clone());
        writer.set_weights_bytes(weights.clone());

        let bundle = writer.finish();
        let reader = UnifiedBundleReader::from_bytes(&bundle).unwrap();

        assert_eq!(reader.graph_bytes(), &graph[..]);
        assert_eq!(reader.weights_bytes(), &weights[..]);
        assert!(reader.verify_checksums());

        // Verify mmap offset is usable
        let offset = reader.weights_mmap_offset().unwrap();
        assert!(offset > 0);
        assert_eq!(offset % PAGE_SIZE, 0);
    }

    #[test]
    fn test_unified_checksum_verification() {
        let mut writer = UnifiedBundleWriter::new();
        writer.set_graph_bytes(b"test data".to_vec());
        writer.set_weights_bytes(b"weight data".to_vec());
        let mut bundle = writer.finish();

        // Verify checksums work
        let reader = UnifiedBundleReader::from_bytes(&bundle).unwrap();
        assert!(reader.verify_graph_checksum());
        assert!(reader.verify_weights_checksum());

        // Corrupt the graph
        bundle[BUNDLE_HEADER_SIZE] ^= 0xFF;
        let reader = UnifiedBundleReader::from_bytes(&bundle).unwrap();
        assert!(!reader.verify_graph_checksum());
    }

    #[test]
    fn test_unified_total_size_calculation() {
        let mut writer = UnifiedBundleWriter::new();
        let graph = vec![0u8; 1000];
        let weights = vec![0u8; 5000];

        writer.set_graph_bytes(graph);
        writer.set_weights_bytes(weights);

        let expected_weights_offset = HoloBundleHeader::calculate_weights_offset(1000) as usize;
        let expected_total = expected_weights_offset + 5000;

        assert_eq!(writer.total_size(), expected_total);

        let bundle = writer.finish();
        assert_eq!(bundle.len(), expected_total);
    }

    // =========================================================================
    // V2 Bundle with Sections Tests
    // =========================================================================

    #[test]
    fn test_unified_writer_with_sections() {
        let mut writer = UnifiedBundleWriter::new();
        let graph = b"test graph data";
        let weights = b"weight data here";

        writer.set_graph_bytes(graph.to_vec());
        writer.set_weights_bytes(weights.to_vec());
        writer.add_raw_section("vocabulary", "text/plain", b"hello\nworld\ntest".to_vec());
        writer.add_raw_section(
            "tokenizer_config",
            "application/json",
            b"{\"do_lower_case\": true}".to_vec(),
        );

        assert!(writer.is_v2());
        assert_eq!(writer.section_count(), 2);
        assert!(writer.has_section("vocabulary"));
        assert!(writer.has_section("tokenizer_config"));
        assert!(!writer.has_section("nonexistent"));

        let bundle = writer.finish();

        // Verify bundle is valid V2
        use crate::core::serialization::{BUNDLE_HEADER_SIZE_V2, detect_bundle_version};
        let version = detect_bundle_version(&bundle).unwrap();
        assert_eq!(version, 2);

        // Bundle should be larger than V2 header
        assert!(bundle.len() > BUNDLE_HEADER_SIZE_V2);
    }

    #[test]
    fn test_unified_writer_v1_when_no_sections() {
        let mut writer = UnifiedBundleWriter::new();
        writer.set_graph_bytes(b"test graph".to_vec());
        writer.set_weights_bytes(b"test weights".to_vec());

        // No sections added
        assert!(!writer.is_v2());
        assert_eq!(writer.section_count(), 0);

        let bundle = writer.finish();

        // Verify bundle is V1
        use crate::core::serialization::detect_bundle_version;
        let version = detect_bundle_version(&bundle).unwrap();
        assert_eq!(version, 1);
    }

    #[test]
    fn test_unified_writer_section_count() {
        let mut writer = UnifiedBundleWriter::new();
        assert_eq!(writer.section_count(), 0);
        assert!(!writer.is_v2());

        writer.add_raw_section("sec1", "text/plain", vec![1, 2, 3]);
        assert_eq!(writer.section_count(), 1);
        assert!(writer.is_v2());

        writer.add_raw_section("sec2", "text/plain", vec![4, 5, 6]);
        assert_eq!(writer.section_count(), 2);

        writer.add_raw_section("sec3", "application/json", vec![7, 8, 9]);
        assert_eq!(writer.section_count(), 3);
    }

    #[test]
    fn test_v2_bundle_sections_data_integrity() {
        let mut writer = UnifiedBundleWriter::new();
        writer.set_graph_bytes(b"graph data".to_vec());
        writer.set_weights_bytes(b"weights data".to_vec());

        let vocab_data = b"line1\nline2\nline3";
        let config_data = b"{\"key\": \"value\"}";

        writer.add_raw_section("vocabulary", "text/plain", vocab_data.to_vec());
        writer.add_raw_section("tokenizer_config", "application/json", config_data.to_vec());

        let bundle = writer.finish();

        // Verify the V2 header
        use crate::core::serialization::HoloBundleHeaderV2;
        let header = HoloBundleHeaderV2::from_bytes(&bundle).unwrap();

        assert_eq!(header.version, 2);
        assert_eq!(header.sections_count, 2);
        assert!(header.sections_table_offset > 0);
        assert_eq!(header.sections_table_offset % PAGE_SIZE as u64, 0);
    }
}

// =============================================================================
// Pipeline Bundle Format (HOLM) - Multi-model with embedded weights
// =============================================================================
//
// This format packages multiple HOLB bundles into a single file, enabling
// deployment of complete ML pipelines (encoder, decoder, tokenizer) as a
// single artifact with efficient per-model mmap access.
//
// Layout:
// +================================+
// |  Pipeline Header (64 bytes)    |  Magic: "HOLM", model count, flags
// +================================+
// |  Model Index (variable)        |  Per-model: name, offset, size, checksum
// +--------------------------------+
// |  Padding to 4KB boundary       |
// +================================+
// |  Model 0 (HOLB bundle)         |  Complete HOLB with graph+weights
// +================================+
// |  Model 1 (HOLB bundle)         |  Complete HOLB with graph+weights
// +================================+
// |  ...                           |
// +================================+

use crate::core::serialization::{
    HoloPipelineHeader, PAGE_SIZE, PIPELINE_HEADER_SIZE, PipelineModelEntry,
};

/// Writer for creating pipeline bundle files (HOLM format).
///
/// The writer accumulates multiple HOLB bundles by name, then produces
/// a properly formatted pipeline bundle with page-aligned model sections.
///
/// # Example
///
/// ```rust,ignore
/// let mut writer = PipelineBundleWriter::new();
/// writer.add_model("encoder", encoder_holb_bytes)?;
/// writer.add_model("decoder", decoder_holb_bytes)?;
/// writer.add_model("tokenizer", tokenizer_holb_bytes)?;
/// writer.write_to_file("t5-pipeline.holo")?;
/// ```
#[derive(Debug, Default)]
pub struct PipelineBundleWriter {
    models: Vec<(String, Vec<u8>)>,
}

impl PipelineBundleWriter {
    /// Create a new pipeline bundle writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a model to the pipeline.
    ///
    /// The model bytes should be a complete HOLB bundle (or raw bytes
    /// that will be stored as-is).
    pub fn add_model(&mut self, name: &str, holb_bytes: Vec<u8>) -> Result<()> {
        // Check for duplicate names
        if self.models.iter().any(|(n, _)| n == name) {
            return Err(OnnxError::InvalidModel(format!(
                "Duplicate model name in pipeline: {}",
                name
            )));
        }
        self.models.push((name.to_string(), holb_bytes));
        Ok(())
    }

    /// Get the number of models in the pipeline.
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Get the model names in order.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Calculate the serialized index size.
    fn index_size(&self) -> usize {
        self.models
            .iter()
            .map(|(name, _)| {
                let entry = PipelineModelEntry::new(name.clone(), 0, 0, 0);
                entry.serialized_size()
            })
            .sum()
    }

    /// Calculate the total bundle size.
    pub fn total_size(&self) -> usize {
        let index_size = self.index_size();
        let models_offset = HoloPipelineHeader::calculate_models_offset(index_size as u64) as usize;

        // Each model is page-aligned within the models section
        let mut current_offset = models_offset;
        for (_, data) in &self.models {
            current_offset += data.len();
            // Round up to page boundary for next model (except last)
            current_offset = current_offset.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        }

        // Adjust for last model not needing padding
        if !self.models.is_empty() {
            let last_size = self.models.last().unwrap().1.len();
            current_offset -= PAGE_SIZE - (last_size % PAGE_SIZE);
            if last_size.is_multiple_of(PAGE_SIZE) {
                current_offset += PAGE_SIZE;
            }
        }

        current_offset
    }

    /// Finish writing and produce the pipeline bundle bytes.
    pub fn finish(self) -> Vec<u8> {
        // Handle empty bundle case
        if self.models.is_empty() {
            let header = HoloPipelineHeader::new(0, 0);
            return header.to_bytes().to_vec();
        }

        let index_size = self.index_size();
        let models_offset = HoloPipelineHeader::calculate_models_offset(index_size as u64) as usize;

        // Build index entries with calculated offsets
        let mut entries = Vec::with_capacity(self.models.len());
        let mut current_model_offset = models_offset;

        for (name, data) in &self.models {
            let checksum = crc32_checksum(data);
            entries.push(PipelineModelEntry::new(
                name.clone(),
                current_model_offset as u64,
                data.len() as u64,
                checksum,
            ));
            // Next model starts at page boundary
            current_model_offset += data.len();
            current_model_offset = current_model_offset.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        }

        // Serialize index
        let mut index_bytes = Vec::with_capacity(index_size);
        for entry in &entries {
            index_bytes.extend_from_slice(&entry.to_bytes());
        }

        // Calculate models total size
        let models_total_size: u64 = self.models.iter().map(|(_, d)| d.len() as u64).sum();

        // Create header
        let mut header = HoloPipelineHeader::new(self.models.len() as u32, index_size as u64);
        header.set_models_total_size(models_total_size);
        header.set_index_checksum(crc32_checksum(&index_bytes));

        // Calculate total size
        let last_model_end = if let Some(last_entry) = entries.last() {
            (last_entry.offset + last_entry.size) as usize
        } else {
            models_offset
        };

        // Allocate output buffer
        let mut output = Vec::with_capacity(last_model_end);

        // Write header
        output.extend_from_slice(&header.to_bytes());

        // Write index
        output.extend_from_slice(&index_bytes);

        // Pad to models offset
        output.resize(models_offset, 0);

        // Write models (each at its page-aligned offset)
        for (i, (_, data)) in self.models.iter().enumerate() {
            let target_offset = entries[i].offset as usize;
            // Pad if needed
            if output.len() < target_offset {
                output.resize(target_offset, 0);
            }
            output.extend_from_slice(data);
        }

        output
    }

    /// Finish and write directly to a file.
    pub fn write_to_file(self, path: &Path) -> Result<usize> {
        let bundle = self.finish();
        let size = bundle.len();

        let mut file = File::create(path)
            .map_err(|e| io_error(format!("Failed to create pipeline bundle file: {}", e)))?;

        file.write_all(&bundle)
            .map_err(|e| io_error(format!("Failed to write pipeline bundle: {}", e)))?;

        Ok(size)
    }
}

/// Reader for pipeline bundle files (HOLM format).
///
/// Provides zero-copy access to individual models within the pipeline.
/// Can work with borrowed bytes or memory-mapped files.
///
/// # Example
///
/// ```rust,ignore
/// let data = std::fs::read("t5-pipeline.holo")?;
/// let reader = PipelineBundleReader::from_bytes(&data)?;
///
/// println!("Models: {:?}", reader.model_names());
///
/// let encoder = reader.get_model("encoder")?;
/// let plan = hologram::compiler::read_holo_from_bytes(encoder.graph_bytes())?;
/// ```
#[derive(Debug)]
pub struct PipelineBundleReader<'a> {
    header: HoloPipelineHeader,
    entries: Vec<PipelineModelEntry>,
    data: &'a [u8],
}

impl<'a> PipelineBundleReader<'a> {
    /// Create a reader from a byte slice.
    ///
    /// Parses and validates the header and index. The data slice must
    /// remain valid for the lifetime of the reader.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self> {
        if data.len() < PIPELINE_HEADER_SIZE {
            return Err(OnnxError::InvalidModel(
                "Pipeline bundle too small for header".into(),
            ));
        }

        // Parse header
        let header = HoloPipelineHeader::from_bytes(data)?;
        header.validate()?;

        // Validate we have enough data for index
        let index_end = header.index_offset as usize + header.index_size as usize;
        if data.len() < index_end {
            return Err(OnnxError::InvalidModel(format!(
                "Pipeline bundle truncated: index requires {} bytes, have {}",
                index_end,
                data.len()
            )));
        }

        // Parse index entries
        let index_data = &data[header.index_offset as usize..index_end];
        let mut entries = Vec::with_capacity(header.model_count as usize);
        let mut offset = 0;

        for _ in 0..header.model_count {
            let (entry, consumed) = PipelineModelEntry::from_bytes(&index_data[offset..])?;
            entries.push(entry);
            offset += consumed;
        }

        // Validate all models are within bounds
        for entry in &entries {
            let model_end = entry.offset as usize + entry.size as usize;
            if model_end > data.len() {
                return Err(OnnxError::InvalidModel(format!(
                    "Pipeline model '{}' extends beyond file: {} > {}",
                    entry.name,
                    model_end,
                    data.len()
                )));
            }
        }

        Ok(Self {
            header,
            entries,
            data,
        })
    }

    /// Get the pipeline header.
    pub fn header(&self) -> &HoloPipelineHeader {
        &self.header
    }

    /// Get the number of models in the pipeline.
    pub fn model_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the model names in order.
    pub fn model_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// Get the index entry for a model by name.
    pub fn get_entry(&self, name: &str) -> Option<&PipelineModelEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Get the raw bytes for a model by name.
    pub fn get_model_bytes(&self, name: &str) -> Option<&'a [u8]> {
        self.get_entry(name).map(|entry| {
            let start = entry.offset as usize;
            let end = start + entry.size as usize;
            &self.data[start..end]
        })
    }

    /// Get a UnifiedBundleReader for a model by name.
    ///
    /// Returns None if the model doesn't exist or isn't a valid HOLB bundle.
    pub fn get_model(&self, name: &str) -> Option<UnifiedBundleReader<'a>> {
        let bytes = self.get_model_bytes(name)?;
        UnifiedBundleReader::from_bytes(bytes).ok()
    }

    /// Get the offset to a model's data within the pipeline file.
    ///
    /// This can be used for mmap-based loading of individual models.
    pub fn get_model_offset(&self, name: &str) -> Option<usize> {
        self.get_entry(name).map(|e| e.offset as usize)
    }

    /// Verify the index checksum.
    pub fn verify_index_checksum(&self) -> bool {
        let index_start = self.header.index_offset as usize;
        let index_end = index_start + self.header.index_size as usize;
        let actual = crc32_checksum(&self.data[index_start..index_end]);
        actual == self.header.index_checksum
    }

    /// Verify a specific model's checksum.
    pub fn verify_model_checksum(&self, name: &str) -> Option<bool> {
        self.get_entry(name).map(|entry| {
            let bytes = self.get_model_bytes(name).unwrap();
            let actual = crc32_checksum(bytes);
            actual == entry.checksum
        })
    }

    /// Verify all checksums.
    pub fn verify_all_checksums(&self) -> bool {
        if !self.verify_index_checksum() {
            return false;
        }
        for entry in &self.entries {
            if let Some(false) = self.verify_model_checksum(&entry.name) {
                return false;
            }
        }
        true
    }

    /// Get the total pipeline bundle size.
    pub fn total_size(&self) -> usize {
        self.data.len()
    }
}

// =============================================================================
// Pipeline Bundle Tests
// =============================================================================

#[cfg(test)]
mod pipeline_bundle_tests {
    use super::*;

    fn create_mock_holb(graph: &[u8], weights: &[u8]) -> Vec<u8> {
        let mut writer = UnifiedBundleWriter::new();
        writer.set_graph_bytes(graph.to_vec());
        writer.set_weights_bytes(weights.to_vec());
        writer.finish()
    }

    #[test]
    fn test_pipeline_writer_empty() {
        let writer = PipelineBundleWriter::new();
        assert_eq!(writer.model_count(), 0);
        assert!(writer.model_names().is_empty());

        let bundle = writer.finish();
        // Should have just header (64 bytes)
        assert_eq!(bundle.len(), PIPELINE_HEADER_SIZE);
    }

    #[test]
    fn test_pipeline_writer_single_model() {
        let mut writer = PipelineBundleWriter::new();
        let holb = create_mock_holb(b"test graph", b"test weights");

        writer.add_model("encoder", holb.clone()).unwrap();
        assert_eq!(writer.model_count(), 1);
        assert_eq!(writer.model_names(), vec!["encoder"]);

        let bundle = writer.finish();

        // Verify we can read it back
        let reader = PipelineBundleReader::from_bytes(&bundle).unwrap();
        assert_eq!(reader.model_count(), 1);
        assert_eq!(reader.model_names(), vec!["encoder"]);

        let encoder_bytes = reader.get_model_bytes("encoder").unwrap();
        assert_eq!(encoder_bytes, &holb[..]);
    }

    #[test]
    fn test_pipeline_writer_multiple_models() {
        let mut writer = PipelineBundleWriter::new();

        let encoder = create_mock_holb(b"encoder graph", b"encoder weights");
        let decoder = create_mock_holb(b"decoder graph", b"decoder weights");
        let tokenizer = create_mock_holb(b"tokenizer", b"");

        writer.add_model("encoder", encoder.clone()).unwrap();
        writer.add_model("decoder", decoder.clone()).unwrap();
        writer.add_model("tokenizer", tokenizer.clone()).unwrap();

        assert_eq!(writer.model_count(), 3);

        let bundle = writer.finish();
        let reader = PipelineBundleReader::from_bytes(&bundle).unwrap();

        assert_eq!(reader.model_count(), 3);
        assert_eq!(
            reader.model_names(),
            vec!["encoder", "decoder", "tokenizer"]
        );

        // Verify each model
        assert_eq!(reader.get_model_bytes("encoder").unwrap(), &encoder[..]);
        assert_eq!(reader.get_model_bytes("decoder").unwrap(), &decoder[..]);
        assert_eq!(reader.get_model_bytes("tokenizer").unwrap(), &tokenizer[..]);
    }

    #[test]
    fn test_pipeline_duplicate_model_name() {
        let mut writer = PipelineBundleWriter::new();
        writer.add_model("encoder", vec![1, 2, 3]).unwrap();
        let result = writer.add_model("encoder", vec![4, 5, 6]);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_model_alignment() {
        let mut writer = PipelineBundleWriter::new();

        // Create models of various sizes
        let model1 = vec![0u8; 1000];
        let model2 = vec![0u8; 2000];

        writer.add_model("m1", model1).unwrap();
        writer.add_model("m2", model2).unwrap();

        let bundle = writer.finish();
        let reader = PipelineBundleReader::from_bytes(&bundle).unwrap();

        // First model offset should be page-aligned
        let m1_offset = reader.get_model_offset("m1").unwrap();
        assert_eq!(m1_offset % PAGE_SIZE, 0);

        // Second model offset should also be page-aligned
        let m2_offset = reader.get_model_offset("m2").unwrap();
        assert_eq!(m2_offset % PAGE_SIZE, 0);
    }

    #[test]
    fn test_pipeline_checksum_verification() {
        let mut writer = PipelineBundleWriter::new();
        let holb = create_mock_holb(b"test", b"data");
        writer.add_model("test", holb).unwrap();

        let mut bundle = writer.finish();
        let reader = PipelineBundleReader::from_bytes(&bundle).unwrap();

        // All checksums should be valid
        assert!(reader.verify_index_checksum());
        assert_eq!(reader.verify_model_checksum("test"), Some(true));
        assert!(reader.verify_all_checksums());

        // Corrupt the model data
        let model_offset = reader.get_model_offset("test").unwrap();
        bundle[model_offset] ^= 0xFF;

        let reader2 = PipelineBundleReader::from_bytes(&bundle).unwrap();
        assert_eq!(reader2.verify_model_checksum("test"), Some(false));
        assert!(!reader2.verify_all_checksums());
    }

    #[test]
    fn test_pipeline_get_unified_bundle_reader() {
        let mut writer = PipelineBundleWriter::new();
        let holb = create_mock_holb(b"test graph data", b"test weight data");
        writer.add_model("encoder", holb).unwrap();

        let bundle = writer.finish();
        let reader = PipelineBundleReader::from_bytes(&bundle).unwrap();

        // Should be able to get a UnifiedBundleReader for the model
        let encoder_reader = reader.get_model("encoder").unwrap();
        assert_eq!(encoder_reader.graph_bytes(), b"test graph data");
        assert_eq!(encoder_reader.weights_bytes(), b"test weight data");
    }

    #[test]
    fn test_pipeline_reader_too_small() {
        let small = [0u8; 32];
        assert!(PipelineBundleReader::from_bytes(&small).is_err());
    }

    #[test]
    fn test_pipeline_reader_truncated_index() {
        let mut writer = PipelineBundleWriter::new();
        writer.add_model("test", vec![1, 2, 3]).unwrap();
        let bundle = writer.finish();

        // Truncate after header
        let truncated = &bundle[..PIPELINE_HEADER_SIZE + 5];
        assert!(PipelineBundleReader::from_bytes(truncated).is_err());
    }

    #[test]
    fn test_pipeline_reader_truncated_model() {
        let mut writer = PipelineBundleWriter::new();
        writer.add_model("test", vec![0u8; 1000]).unwrap();
        let bundle = writer.finish();

        // Truncate the model data
        let truncated = &bundle[..bundle.len() - 100];
        assert!(PipelineBundleReader::from_bytes(truncated).is_err());
    }

    #[test]
    fn test_pipeline_nonexistent_model() {
        let mut writer = PipelineBundleWriter::new();
        writer.add_model("encoder", vec![1, 2, 3]).unwrap();
        let bundle = writer.finish();

        let reader = PipelineBundleReader::from_bytes(&bundle).unwrap();
        assert!(reader.get_model_bytes("decoder").is_none());
        assert!(reader.get_entry("decoder").is_none());
        assert!(reader.get_model_offset("decoder").is_none());
    }
}
