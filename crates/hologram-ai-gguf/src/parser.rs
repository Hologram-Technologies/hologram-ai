//! GGUF v2/v3 binary format parser.
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

/// GGUF magic bytes: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46475547; // 'G','G','U','F'

/// Parsed GGUF file header + metadata + tensor descriptors.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorDescriptor>,
    /// Byte offset where tensor data begins (after header + metadata + tensor info).
    pub data_offset: u64,
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::U32(v) => Some(*v),
            MetaValue::I32(v) => Some(*v as u32),
            MetaValue::U64(v) => Some(*v as u32),
            MetaValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U64(v) => Some(*v),
            MetaValue::U32(v) => Some(*v as u64),
            MetaValue::I64(v) => Some(*v as u64),
            MetaValue::I32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    out.push(v.as_str()?);
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    out.push(v.as_f32()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

/// GGUF tensor type IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            30 => Ok(Self::BF16),
            _ => bail!("unknown GGML type: {v}"),
        }
    }

    /// Bytes per element for non-quantized types, or block size info for quantized.
    pub fn type_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18, // block of 32 elements = 2 (scale) + 16 (nibbles)
            Self::Q4_1 => 20, // block of 32 = 2+2+16
            Self::Q5_0 => 22, // block of 32 = 2+4+16
            Self::Q5_1 => 24, // block of 32 = 2+2+4+16
            Self::Q8_0 => 34, // block of 32 = 2+32
            Self::Q8_1 => 40, // block of 32 = 4+4+32
            Self::Q2K => 256,
            Self::Q3K => 256,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            _ => 1, // IQ types — not commonly used, treat as 1
        }
    }

    /// Number of elements per block for quantized types.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            _ => 32,
        }
    }

    /// Total byte size for `n_elements` of this type.
    pub fn byte_size(&self, n_elements: u64) -> u64 {
        let bs = self.block_size() as u64;
        let n_blocks = n_elements.div_ceil(bs);
        n_blocks * self.type_size() as u64
    }
}

/// Descriptor for a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset of this tensor's data relative to `data_offset`.
    pub offset: u64,
}

impl TensorDescriptor {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    pub fn byte_size(&self) -> u64 {
        self.ggml_type.byte_size(self.n_elements())
    }
}

/// Parse a GGUF file from a byte slice.
pub fn parse_gguf(data: &[u8]) -> Result<GgufFile> {
    let mut r = Reader::new(data);

    let magic = r.u32().context("reading magic")?;
    if magic != GGUF_MAGIC {
        bail!("not a GGUF file (magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})");
    }

    let version = r.u32().context("reading version")?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version: {version} (expected 2 or 3)");
    }

    let tensor_count = r.u64_compat(version).context("reading tensor_count")?;
    let metadata_kv_count = r.u64_compat(version).context("reading metadata_kv_count")?;

    // Parse metadata KV pairs.
    let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        let key = r.gguf_string(version).context("reading metadata key")?;
        let value = r.meta_value(version).context("reading metadata value")?;
        metadata.insert(key, value);
    }

    // Parse tensor descriptors.
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.gguf_string(version).context("reading tensor name")?;
        let n_dims = r.u32().context("reading tensor n_dims")? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.u64_compat(version).context("reading tensor dim")?);
        }
        let type_id = r.u32().context("reading tensor type")?;
        let ggml_type = GgmlType::from_u32(type_id)?;
        let offset = r.u64().context("reading tensor offset")?;
        tensors.push(TensorDescriptor {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    // Data starts at the next alignment boundary after the current position.
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    let data_offset = align_up(r.pos, alignment) as u64;

    Ok(GgufFile {
        version,
        metadata,
        tensors,
        data_offset,
    })
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

// ── Binary reader ──────────────────────────────────────────────────────

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn ensure(&self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() {
            bail!(
                "unexpected end of data at offset {}: need {n} bytes, {} remaining",
                self.pos,
                self.data.len() - self.pos
            );
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes(self.data[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    fn f32(&mut self) -> Result<f32> {
        self.ensure(4)?;
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn f64(&mut self) -> Result<f64> {
        self.ensure(8)?;
        let v = f64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    /// In GGUF v2 counts are u64; in v3 they are also u64.
    /// In older versions they may be u32 — we handle both.
    fn u64_compat(&mut self, version: u32) -> Result<u64> {
        if version >= 2 {
            self.u64()
        } else {
            Ok(self.u32()? as u64)
        }
    }

    fn gguf_string(&mut self, version: u32) -> Result<String> {
        let len = self.u64_compat(version)? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .context("invalid UTF-8 in GGUF string")?
            .to_owned();
        self.pos += len;
        Ok(s)
    }

    fn meta_value(&mut self, version: u32) -> Result<MetaValue> {
        let type_id = self.u32()?;
        self.read_typed_value(type_id, version)
    }

    fn read_typed_value(&mut self, type_id: u32, version: u32) -> Result<MetaValue> {
        match type_id {
            0 => Ok(MetaValue::U8(self.u8()?)),
            1 => Ok(MetaValue::I8(self.i8()?)),
            2 => Ok(MetaValue::U16(self.u16()?)),
            3 => Ok(MetaValue::I16(self.i16()?)),
            4 => Ok(MetaValue::U32(self.u32()?)),
            5 => Ok(MetaValue::I32(self.i32()?)),
            6 => Ok(MetaValue::F32(self.f32()?)),
            7 => Ok(MetaValue::Bool(self.bool()?)),
            8 => Ok(MetaValue::Str(self.gguf_string(version)?)),
            9 => {
                // Array: element_type (u32) + count (u64) + elements
                let elem_type = self.u32()?;
                let count = self.u64_compat(version)? as usize;
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(self.read_typed_value(elem_type, version)?);
                }
                Ok(MetaValue::Array(arr))
            }
            10 => Ok(MetaValue::U64(self.u64()?)),
            11 => Ok(MetaValue::I64(self.i64()?)),
            12 => Ok(MetaValue::F64(self.f64()?)),
            _ => bail!("unknown GGUF metadata type: {type_id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gguf_header(version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf
    }

    fn append_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        // Key: len (u64) + bytes
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        // Type: 8 = string
        buf.extend_from_slice(&8u32.to_le_bytes());
        // Value: len (u64) + bytes
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
    }

    #[test]
    fn parse_empty_gguf() {
        let mut buf = make_gguf_header(3, 0, 1);
        append_kv_string(&mut buf, "general.architecture", "llama");
        // Pad to alignment
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let gguf = parse_gguf(&buf).unwrap();
        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.tensors.len(), 0);
        assert_eq!(
            gguf.metadata.get("general.architecture").unwrap().as_str(),
            Some("llama")
        );
    }

    #[test]
    fn reject_bad_magic() {
        let buf = vec![0u8; 32];
        assert!(parse_gguf(&buf).is_err());
    }
}
