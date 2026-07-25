//! Canonical deterministic encoding.
//!
//! All canonical bytes in hologram-ai (bundle manifests, provenance records)
//! use this little-endian TLV-style encoding. Determinism rules:
//!
//! - integers are fixed-width little-endian;
//! - strings/bytes are `u32` length-prefixed, length checked on read;
//! - sequences are `u32` count-prefixed and encoded in declaration order —
//!   there are no maps, so iteration order can never leak into output;
//! - optional values are `bool` tag + payload;
//! - readers reject trailing bytes, truncated input, and lengths beyond
//!   [`MAX_LEN`], with checked arithmetic on every offset.

/// Maximum byte length accepted for a single length-prefixed item or sequence.
pub const MAX_LEN: u32 = 1 << 26; // 64 MiB

/// Encoding/decoding failure. Trivially copyable so it is usable in `no_std`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonError {
    /// Input ended before a complete value could be read.
    Truncated,
    /// A length prefix exceeded [`MAX_LEN`] or the remaining input.
    LengthOutOfRange,
    /// Bytes remained after the top-level value was fully read.
    TrailingBytes,
    /// A string was not valid UTF-8.
    InvalidUtf8,
    /// A discriminant did not correspond to a known variant.
    UnknownDiscriminant,
    /// A boolean tag was not 0 or 1.
    InvalidBool,
}

impl core::fmt::Display for CanonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::Truncated => "truncated canonical input",
            Self::LengthOutOfRange => "length prefix out of range",
            Self::TrailingBytes => "trailing bytes after canonical value",
            Self::InvalidUtf8 => "invalid UTF-8 in canonical string",
            Self::UnknownDiscriminant => "unknown discriminant",
            Self::InvalidBool => "invalid boolean tag",
        };
        f.write_str(msg)
    }
}

/// Canonical byte writer. Allocation happens only here, at encode time.
#[derive(Debug, Default)]
pub struct CanonWriter {
    buf: alloc::vec::Vec<u8>,
}

impl CanonWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finish(self) -> alloc::vec::Vec<u8> {
        self.buf
    }

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn bool(&mut self, v: bool) {
        self.buf.push(u8::from(v));
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn bytes(&mut self, v: &[u8]) {
        self.u32(v.len() as u32);
        self.buf.extend_from_slice(v);
    }

    pub fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }

    pub fn opt_str(&mut self, v: Option<&str>) {
        match v {
            Some(s) => {
                self.bool(true);
                self.str(s);
            }
            None => self.bool(false),
        }
    }

    /// Encode a sequence length. Callers encode exactly `n` items after this.
    pub fn seq_len(&mut self, n: usize) {
        self.u32(n as u32);
    }
}

/// Canonical byte reader over a borrowed slice. All arithmetic is checked.
#[derive(Debug, Clone)]
pub struct CanonReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> CanonReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.pos == self.buf.len()
    }

    /// Assert the whole input was consumed.
    pub fn finish(&self) -> Result<(), CanonError> {
        if self.is_empty() {
            Ok(())
        } else {
            Err(CanonError::TrailingBytes)
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CanonError> {
        let end = self.pos.checked_add(n).ok_or(CanonError::Truncated)?;
        let slice = self.buf.get(self.pos..end).ok_or(CanonError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    pub fn u8(&mut self) -> Result<u8, CanonError> {
        Ok(self.take(1)?[0])
    }

    pub fn bool(&mut self) -> Result<bool, CanonError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CanonError::InvalidBool),
        }
    }

    pub fn u16(&mut self) -> Result<u16, CanonError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> Result<u32, CanonError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Result<u64, CanonError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn len(&mut self) -> Result<u32, CanonError> {
        let n = self.u32()?;
        if n > MAX_LEN {
            return Err(CanonError::LengthOutOfRange);
        }
        Ok(n)
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], CanonError> {
        let n = self.len()? as usize;
        self.take(n)
    }

    pub fn str(&mut self) -> Result<&'a str, CanonError> {
        core::str::from_utf8(self.bytes()?).map_err(|_| CanonError::InvalidUtf8)
    }

    pub fn opt_str(&mut self) -> Result<Option<&'a str>, CanonError> {
        if self.bool()? {
            Ok(Some(self.str()?))
        } else {
            Ok(None)
        }
    }

    /// Read a sequence count; the caller reads exactly that many items.
    pub fn seq_len(&mut self) -> Result<u32, CanonError> {
        self.len()
    }
}
