//! Portable, modality-neutral inference schemas for hologram-ai.
//!
//! `no_std + alloc`: value/operation/capability descriptors, inference
//! requests and completions, resolution status and finish reasons, stable
//! error categories, and canonical (deterministic) encoding rules. No I/O,
//! no engine, no container dependencies.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod canon;
pub mod error;
pub mod inference;
pub mod manifest;
pub mod progress;
pub mod value;

pub use error::{AiError, AiResult, ErrorCategory};
pub use inference::{
    EventSink, FinishReason, InferenceCompletion, InferenceOutput, InferenceRequest,
    InferenceRequestBuilder, NullEventSink, ResolutionStatus, StreamEvent, Witness,
};
pub use manifest::{
    AbiVersions, ArtifactDescriptor, ArtifactRole, InferenceModelManifest, ModelProvenance,
    OperationDescriptor, StatusPolicy, ARTIFACT_FORMAT_R4G1, ENGINE_UOR_R4,
    MANIFEST_SCHEMA_VERSION,
};
pub use progress::{CancellationToken, NullProgressSink, ProgressEvent, ProgressSink};
pub use value::{MediaData, Payload, Value, ValueDescriptor, ValueKind};
