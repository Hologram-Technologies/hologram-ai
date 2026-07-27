//! Multimodal value types for the inference ABI.
//!
//! Modality lives in operation/value descriptors, never in layer kinds.
//! Large media payloads may travel inline or by content-addressed κ
//! reference (`blake3:<64 hex>`), so application layers do not copy bulk
//! data between each other.

use alloc::string::String;
use alloc::vec::Vec;

use crate::canon::{CanonError, CanonReader, CanonWriter};

/// The kind of a declared operation input or output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// UTF-8 text.
    Text,
    /// A sequence of token identifiers.
    Tokens,
    /// An image (media type required, e.g. `image/png`).
    Image,
    /// Audio (media type required, e.g. `audio/wav`; sample format in the
    /// model's audio-processor contract).
    Audio,
    /// Opaque inline bytes with a media type.
    Bytes,
    /// A content-addressed κ reference to a blob.
    ContentRef,
    /// Canonical-encoded structured request data.
    Structured,
    /// A caller-owned output buffer (see the `*_into` engine APIs).
    Buffer,
}

impl ValueKind {
    fn discriminant(self) -> u8 {
        match self {
            Self::Text => 0,
            Self::Tokens => 1,
            Self::Image => 2,
            Self::Audio => 3,
            Self::Bytes => 4,
            Self::ContentRef => 5,
            Self::Structured => 6,
            Self::Buffer => 7,
        }
    }

    fn from_discriminant(d: u8) -> Result<Self, CanonError> {
        match d {
            0 => Ok(Self::Text),
            1 => Ok(Self::Tokens),
            2 => Ok(Self::Image),
            3 => Ok(Self::Audio),
            4 => Ok(Self::Bytes),
            5 => Ok(Self::ContentRef),
            6 => Ok(Self::Structured),
            7 => Ok(Self::Buffer),
            _ => Err(CanonError::UnknownDiscriminant),
        }
    }
}

/// A declared input or output of an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueDescriptor {
    pub name: String,
    pub kind: ValueKind,
    /// Media type (required for image/audio/bytes/content-ref values).
    pub media_type: Option<String>,
    pub required: bool,
}

impl ValueDescriptor {
    pub fn encode(&self, w: &mut CanonWriter) {
        w.str(&self.name);
        w.u8(self.kind.discriminant());
        w.opt_str(self.media_type.as_deref());
        w.bool(self.required);
    }

    pub fn decode(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        let name = r.str()?.into();
        let kind = ValueKind::from_discriminant(r.u8()?)?;
        let media_type = r.opt_str()?.map(Into::into);
        let required = r.bool()?;
        Ok(Self {
            name,
            kind,
            media_type,
            required,
        })
    }
}

/// Inline bytes or a κ reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    Inline(Vec<u8>),
    /// Content-addressed κ label (`blake3:<64 hex>`).
    Ref(String),
}

/// A media payload with its media type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaData {
    pub media_type: String,
    pub payload: Payload,
}

/// A runtime value passed to or returned from an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Text(String),
    Tokens(Vec<u32>),
    Image(MediaData),
    Audio(MediaData),
    Bytes { media_type: String, data: Vec<u8> },
    ContentRef { media_type: String, kappa: String },
    Structured(Vec<u8>),
}

impl Value {
    /// The [`ValueKind`] this value satisfies.
    pub fn kind(&self) -> ValueKind {
        match self {
            Self::Text(_) => ValueKind::Text,
            Self::Tokens(_) => ValueKind::Tokens,
            Self::Image(_) => ValueKind::Image,
            Self::Audio(_) => ValueKind::Audio,
            Self::Bytes { .. } => ValueKind::Bytes,
            Self::ContentRef { .. } => ValueKind::ContentRef,
            Self::Structured(_) => ValueKind::Structured,
        }
    }

    pub fn encode(&self, w: &mut CanonWriter) {
        match self {
            Self::Text(s) => {
                w.u8(0);
                w.str(s);
            }
            Self::Tokens(ts) => {
                w.u8(1);
                w.seq_len(ts.len());
                for t in ts {
                    w.u32(*t);
                }
            }
            Self::Image(m) => {
                w.u8(2);
                encode_media(w, m);
            }
            Self::Audio(m) => {
                w.u8(3);
                encode_media(w, m);
            }
            Self::Bytes { media_type, data } => {
                w.u8(4);
                w.str(media_type);
                w.bytes(data);
            }
            Self::ContentRef { media_type, kappa } => {
                w.u8(5);
                w.str(media_type);
                w.str(kappa);
            }
            Self::Structured(data) => {
                w.u8(6);
                w.bytes(data);
            }
        }
    }

    pub fn decode(r: &mut CanonReader<'_>) -> Result<Self, CanonError> {
        match r.u8()? {
            0 => Ok(Self::Text(r.str()?.into())),
            1 => {
                let n = r.seq_len()?;
                let mut ts = Vec::with_capacity(n.min(1 << 20) as usize);
                for _ in 0..n {
                    ts.push(r.u32()?);
                }
                Ok(Self::Tokens(ts))
            }
            2 => Ok(Self::Image(decode_media(r)?)),
            3 => Ok(Self::Audio(decode_media(r)?)),
            4 => Ok(Self::Bytes {
                media_type: r.str()?.into(),
                data: r.bytes()?.into(),
            }),
            5 => Ok(Self::ContentRef {
                media_type: r.str()?.into(),
                kappa: r.str()?.into(),
            }),
            6 => Ok(Self::Structured(r.bytes()?.into())),
            _ => Err(CanonError::UnknownDiscriminant),
        }
    }
}

fn encode_media(w: &mut CanonWriter, m: &MediaData) {
    w.str(&m.media_type);
    match &m.payload {
        Payload::Inline(data) => {
            w.bool(true);
            w.bytes(data);
        }
        Payload::Ref(kappa) => {
            w.bool(false);
            w.str(kappa);
        }
    }
}

fn decode_media(r: &mut CanonReader<'_>) -> Result<MediaData, CanonError> {
    let media_type = r.str()?.into();
    let payload = if r.bool()? {
        Payload::Inline(r.bytes()?.into())
    } else {
        Payload::Ref(r.str()?.into())
    };
    Ok(MediaData {
        media_type,
        payload,
    })
}
