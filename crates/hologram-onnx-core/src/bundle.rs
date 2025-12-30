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
//! use hologram_onnx_core::bundle::{HoloBundle, BundleBuilder};
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
            return Err(OnnxError::InvalidModel(
                "Header too small".to_string(),
            ));
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
            bytes[16], bytes[17], bytes[18], bytes[19],
            bytes[20], bytes[21], bytes[22], bytes[23],
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
    pub fn add_model_from_file(&mut self, name: impl Into<String>, path: impl AsRef<Path>) -> Result<&mut Self> {
        let data = fs::read(path.as_ref()).map_err(|e| {
            io_error(format!("Failed to read model file: {}", e))
        })?;
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
        let file = File::open(path).map_err(|e| {
            io_error(format!("Failed to open bundle file: {}", e))
        })?;
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
        reader.read_exact(&mut header_bytes).map_err(|e| {
            io_error(format!("Failed to read header: {}", e))
        })?;
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
        reader.seek(SeekFrom::Start(header.index_offset)).map_err(|e| {
            io_error(format!("Failed to seek to index: {}", e))
        })?;

        // Read index entries
        let mut entries = Vec::with_capacity(header.model_count as usize);
        for _ in 0..header.model_count {
            let entry = ModelIndexEntry::read_from(reader).map_err(|e| {
                io_error(format!("Failed to read index entry: {}", e))
            })?;
            entries.push(entry);
        }

        // Read model data
        let mut model_data = Vec::with_capacity(entries.len());
        for entry in &entries {
            reader.seek(SeekFrom::Start(entry.data_offset)).map_err(|e| {
                io_error(format!("Failed to seek to model data: {}", e))
            })?;

            let mut data = vec![0u8; entry.data_size as usize];
            reader.read_exact(&mut data).map_err(|e| {
                io_error(format!("Failed to read model data: {}", e))
            })?;

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
        reader.seek(SeekFrom::Start(0)).map_err(|e| {
            io_error(format!("Failed to seek: {}", e))
        })?;

        // Read entire file
        let mut data = Vec::new();
        reader.read_to_end(&mut data).map_err(|e| {
            io_error(format!("Failed to read v1 model: {}", e))
        })?;

        // Extract model name from v1 format
        // v1 format: HOLO (4) | version (4) | name_len (4) | name | ...
        let name_len = u32::from_le_bytes([
            header_bytes[8], header_bytes[9], header_bytes[10], header_bytes[11]
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
        let file = File::create(path).map_err(|e| {
            io_error(format!("Failed to create bundle file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }

    /// Write bundle to a writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Write header
        writer.write_all(&self.header.to_bytes()).map_err(|e| {
            io_error(format!("Failed to write header: {}", e))
        })?;

        // Write model data
        for data in &self.model_data {
            writer.write_all(data).map_err(|e| {
                io_error(format!("Failed to write model data: {}", e))
            })?;
        }

        // Write index
        for entry in &self.entries {
            entry.write_to(writer).map_err(|e| {
                io_error(format!("Failed to write index entry: {}", e))
            })?;
        }

        writer.flush().map_err(|e| {
            io_error(format!("Failed to flush writer: {}", e))
        })?;

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
        fs::create_dir_all(dir).map_err(|e| {
            io_error(format!("Failed to create directory: {}", e))
        })?;

        for (entry, data) in self.entries.iter().zip(self.model_data.iter()) {
            let path = dir.join(format!("{}.holo", entry.name));
            fs::write(&path, data).map_err(|e| {
                io_error(format!("Failed to write {}: {}", path.display(), e))
            })?;
        }

        Ok(())
    }

    /// Create a bundle from a directory of .holo files.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut builder = BundleBuilder::new();

        for entry in fs::read_dir(dir).map_err(|e| {
            io_error(format!("Failed to read directory: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                io_error(format!("Failed to read directory entry: {}", e))
            })?;
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
    let mut file = File::open(path).map_err(|e| {
        io_error(format!("Failed to open file: {}", e))
    })?;

    let mut header_bytes = [0u8; HEADER_SIZE];
    file.read_exact(&mut header_bytes).map_err(|e| {
        io_error(format!("Failed to read header: {}", e))
    })?;

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
        let entry = ModelIndexEntry::new(
            "text_encoder".to_string(),
            1000,
            5000,
            0x12345678,
        );

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
