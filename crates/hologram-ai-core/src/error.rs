//! Stable, typed error categories for SDK mapping.
//!
//! Every public failure in hologram-ai is an [`AiError`] carrying a
//! [`ErrorCategory`]. Categories have stable numeric identifiers
//! ([`ErrorCategory::code`]) so Hologram's FFI can map them into Python and
//! TypeScript error classes without string matching.

use alloc::string::String;

/// Stable error category. Numeric codes are part of the public ABI:
/// append new categories, never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    InvalidArgument,
    SourceAcquisition,
    Authentication,
    UnsupportedModel,
    UnsupportedCapability,
    Download,
    Cache,
    Compile,
    QualityGate,
    BundleEncode,
    BundleDecode,
    ArchiveEncode,
    ArchiveDecode,
    IntegrityMismatch,
    ModelSelection,
    ProcessorMismatch,
    EngineInit,
    Inference,
    Cancelled,
    AbiMismatch,
}

impl ErrorCategory {
    /// Stable numeric identifier for FFI/SDK mapping.
    pub fn code(self) -> u16 {
        match self {
            Self::InvalidArgument => 1,
            Self::SourceAcquisition => 2,
            Self::Authentication => 3,
            Self::UnsupportedModel => 4,
            Self::UnsupportedCapability => 5,
            Self::Download => 6,
            Self::Cache => 7,
            Self::Compile => 8,
            Self::QualityGate => 9,
            Self::BundleEncode => 10,
            Self::BundleDecode => 11,
            Self::ArchiveEncode => 12,
            Self::ArchiveDecode => 13,
            Self::IntegrityMismatch => 14,
            Self::ModelSelection => 15,
            Self::ProcessorMismatch => 16,
            Self::EngineInit => 17,
            Self::Inference => 18,
            Self::Cancelled => 19,
            Self::AbiMismatch => 20,
        }
    }

    /// Map an FFI code back to a category.
    pub fn from_code(code: u16) -> Option<Self> {
        Some(match code {
            1 => Self::InvalidArgument,
            2 => Self::SourceAcquisition,
            3 => Self::Authentication,
            4 => Self::UnsupportedModel,
            5 => Self::UnsupportedCapability,
            6 => Self::Download,
            7 => Self::Cache,
            8 => Self::Compile,
            9 => Self::QualityGate,
            10 => Self::BundleEncode,
            11 => Self::BundleDecode,
            12 => Self::ArchiveEncode,
            13 => Self::ArchiveDecode,
            14 => Self::IntegrityMismatch,
            15 => Self::ModelSelection,
            16 => Self::ProcessorMismatch,
            17 => Self::EngineInit,
            18 => Self::Inference,
            19 => Self::Cancelled,
            20 => Self::AbiMismatch,
            _ => return None,
        })
    }

    /// Stable machine-readable name (e.g. `unsupported-model`).
    pub fn name(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid-argument",
            Self::SourceAcquisition => "source-acquisition",
            Self::Authentication => "authentication",
            Self::UnsupportedModel => "unsupported-model",
            Self::UnsupportedCapability => "unsupported-capability",
            Self::Download => "download",
            Self::Cache => "cache",
            Self::Compile => "compile",
            Self::QualityGate => "quality-gate",
            Self::BundleEncode => "bundle-encode",
            Self::BundleDecode => "bundle-decode",
            Self::ArchiveEncode => "archive-encode",
            Self::ArchiveDecode => "archive-decode",
            Self::IntegrityMismatch => "integrity-mismatch",
            Self::ModelSelection => "model-selection",
            Self::ProcessorMismatch => "processor-mismatch",
            Self::EngineInit => "engine-init",
            Self::Inference => "inference",
            Self::Cancelled => "cancelled",
            Self::AbiMismatch => "abi-mismatch",
        }
    }
}

/// The single public error type of hologram-ai.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiError {
    category: ErrorCategory,
    message: String,
}

impl AiError {
    pub fn new(category: ErrorCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
        }
    }

    pub fn category(&self) -> ErrorCategory {
        self.category
    }

    /// Stable numeric FFI identifier of the category.
    pub fn code(&self) -> u16 {
        self.category.code()
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCategory::InvalidArgument, message)
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(ErrorCategory::Cancelled, message)
    }
}

impl core::fmt::Display for AiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.category.name(), self.message)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AiError {}

/// Result alias used across hologram-ai.
pub type AiResult<T> = Result<T, AiError>;
