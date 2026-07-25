//! hologram-ai: production integration layer between Hugging Face model
//! sources, the `uor-r4` R4G1 compiler/runtime, and Hologram `.holo`
//! applications.
//!
//! Public surface: the [`Compiler`] builder (source → `.holo`), the
//! [`Application`] registry (`.holo` → model services), and [`Session`]
//! (invocation, prediction, generation).

#![forbid(unsafe_code)]

mod application;
pub mod cli;
mod compiler;
pub mod ffi;
mod json;
mod model;
mod package;
mod session;

pub use application::{Application, ModelDescriptor};
pub use compiler::{
    CompileOptions, CompiledModel, Compiler, CompilerBuilder, HuggingFaceSource, LocalSource,
    Source, DEFAULT_ENTRY,
};
pub use model::Model;
pub use package::{add_model_layers, build_model_archive, ModelLayer};
pub use session::Session;

pub use hologram_ai_core::{
    AiError, AiResult, CancellationToken, ErrorCategory, FinishReason, InferenceCompletion,
    InferenceOutput, InferenceRequest, ProgressEvent, ProgressSink, ResolutionStatus, StreamEvent,
    Value, ValueKind, Witness,
};
