//! Runner abstraction for side-effecting model execution.

use thiserror::Error;

use crate::{InferenceOutput, InferenceRequest, RunnerManifest};

#[cfg(test)]
use crate::{InferenceProvenance, KappaRef};

/// A capability-scoped model runner.
///
/// Implementations observe `InferenceRequest`, execute through a real engine or
/// worker boundary, and return an `InferenceOutput`. Reducers never invoke this
/// trait directly; they only fold the resulting `AiEvent`s.
pub trait ModelRunner {
    /// The manifest describing this runner surface.
    fn manifest(&self) -> &RunnerManifest;

    /// Execute a request and produce an output payload.
    fn run(&self, request: &InferenceRequest) -> Result<InferenceOutput, ModelRunnerError>;
}

/// Runner-level execution error.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModelRunnerError {
    /// The request could not be accepted or interpreted by the runner.
    #[error("invalid inference request: {0}")]
    InvalidRequest(String),
    /// The runner attempted execution and it failed.
    #[error("inference execution failed: {0}")]
    ExecutionFailed(String),
}

#[cfg(test)]
/// Deterministic echo runner reserved for unit tests.
///
/// It does not mint κ-labels or perform hashing. All κ-labels are injected by
/// the caller so tests remain explicit about the adapter boundary.
pub struct DeterministicEchoRunner {
    manifest: RunnerManifest,
    worker_kappa: KappaRef,
    output_kappa: KappaRef,
}

#[cfg(test)]
impl DeterministicEchoRunner {
    /// Create a deterministic echo runner for tests.
    pub fn new(manifest: RunnerManifest, worker_kappa: KappaRef, output_kappa: KappaRef) -> Self {
        Self {
            manifest,
            worker_kappa,
            output_kappa,
        }
    }
}

#[cfg(test)]
impl ModelRunner for DeterministicEchoRunner {
    fn manifest(&self) -> &RunnerManifest {
        &self.manifest
    }

    fn run(&self, request: &InferenceRequest) -> Result<InferenceOutput, ModelRunnerError> {
        if request.prompt.text.is_empty() {
            return Err(ModelRunnerError::InvalidRequest(
                "prompt text must not be empty".to_string(),
            ));
        }
        Ok(InferenceOutput {
            request_kappa: request.request_kappa.clone(),
            output_kappa: self.output_kappa.clone(),
            content: request.prompt.text.clone(),
            provenance: InferenceProvenance {
                request_kappa: request.request_kappa.clone(),
                input_event_kappa: request.prompt.prompt_kappa.clone(),
                prompt_kappa: request.prompt.prompt_kappa.clone(),
                model_kappa: request.model_kappa.clone(),
                runner_kappa: self.manifest.runner_kappa.clone(),
                worker_kappa: self.worker_kappa.clone(),
                params_kappa: request.params.params_kappa.clone(),
                output_kappa: self.output_kappa.clone(),
            },
        })
    }
}
