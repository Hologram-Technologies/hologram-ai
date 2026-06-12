//! App-domain foundations for AI holospace applications.
//!
//! This crate defines deterministic event-sourced state, inference-facing
//! manifests, and the runner abstraction that a higher-level `holo-apps` layer
//! can build on. It does **not** perform model execution inside the reducer.

#![forbid(unsafe_code)]

mod domain;
mod reducer;
mod runner;

pub use domain::{
    AiAppManifest, AiEvent, AiView, AppEntryKind, CompletedInference, FailedInference,
    InferenceOutput, InferenceParams, InferenceProvenance, InferenceRequest, KappaRef,
    ModelManifest, PendingInference, PendingPhase, Prompt, RunnerKind, RunnerManifest,
};
pub use reducer::reduce;
pub use runner::{ModelRunner, ModelRunnerError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::DeterministicEchoRunner;

    fn kappa(label: &str) -> KappaRef {
        KappaRef::new(label)
    }

    fn model_manifest() -> ModelManifest {
        ModelManifest {
            model_kappa: kappa("blake3:model-manifest"),
            archive_kappa: kappa("blake3:model-archive"),
            name: "bert-base-uncased".to_string(),
            description: Some("Compiled BERT archive".to_string()),
        }
    }

    fn runner_manifest() -> RunnerManifest {
        RunnerManifest {
            runner_kappa: kappa("blake3:runner-manifest"),
            name: "local-holo-engine".to_string(),
            kind: RunnerKind::HoloEngine,
        }
    }

    fn prompt() -> Prompt {
        Prompt {
            prompt_kappa: kappa("blake3:prompt"),
            text: "this is a test".to_string(),
        }
    }

    fn request() -> InferenceRequest {
        InferenceRequest {
            request_kappa: kappa("blake3:request"),
            app_kappa: Some(kappa("blake3:app")),
            model_kappa: kappa("blake3:model-manifest"),
            runner_kappa: kappa("blake3:runner-manifest"),
            prompt: prompt(),
            params: InferenceParams {
                params_kappa: Some(kappa("blake3:params")),
                max_output_tokens: Some(64),
                temperature_milli: Some(0),
                stop_sequences: vec!["</s>".to_string()],
            },
        }
    }

    fn prompt_submitted_event() -> AiEvent {
        AiEvent::PromptSubmitted {
            event_kappa: kappa("blake3:event-prompt"),
            request: request(),
        }
    }

    fn completed_output() -> InferenceOutput {
        InferenceOutput {
            request_kappa: kappa("blake3:request"),
            output_kappa: kappa("blake3:output"),
            content: "echo: this is a test".to_string(),
            provenance: InferenceProvenance {
                request_kappa: kappa("blake3:request"),
                input_event_kappa: kappa("blake3:event-prompt"),
                prompt_kappa: kappa("blake3:prompt"),
                model_kappa: kappa("blake3:model-manifest"),
                runner_kappa: kappa("blake3:runner-manifest"),
                worker_kappa: kappa("blake3:worker"),
                params_kappa: Some(kappa("blake3:params")),
                output_kappa: kappa("blake3:output"),
            },
        }
    }

    #[test]
    fn reducing_empty_event_list_yields_empty_view() {
        let view = reduce(&[]);

        assert!(view.models.is_empty());
        assert!(view.pending_jobs.is_empty());
        assert!(view.completed_jobs.is_empty());
        assert!(view.failed_jobs.is_empty());
    }

    #[test]
    fn registering_a_model_populates_the_model_registry() {
        let events = [AiEvent::ModelRegistered {
            event_kappa: kappa("blake3:event-model"),
            manifest: model_manifest(),
        }];
        let view = reduce(&events);

        let registered = view
            .models
            .get(&kappa("blake3:model-manifest"))
            .expect("model should be registered");
        assert_eq!(registered.archive_kappa, kappa("blake3:model-archive"));
    }

    #[test]
    fn submitting_a_prompt_creates_a_pending_job() {
        let events = [prompt_submitted_event()];
        let view = reduce(&events);

        let pending = view
            .pending_jobs
            .get(&kappa("blake3:request"))
            .expect("request should be pending");
        assert_eq!(pending.phase, PendingPhase::Queued);
        assert_eq!(pending.request.prompt.text, "this is a test");
        assert!(view.completed_jobs.is_empty());
        assert!(view.failed_jobs.is_empty());
    }

    #[test]
    fn completing_inference_moves_a_job_from_pending_to_completed() {
        let events = [
            prompt_submitted_event(),
            AiEvent::InferenceCompleted {
                event_kappa: kappa("blake3:event-complete"),
                output: completed_output(),
            },
        ];
        let view = reduce(&events);

        assert!(!view.completed_jobs.is_empty());
        assert!(!view.completed_jobs.contains_key(&kappa("blake3:missing")));
        assert!(!view.pending_jobs.contains_key(&kappa("blake3:request")));
        let completed = view
            .completed_jobs
            .get(&kappa("blake3:request"))
            .expect("request should be completed");
        assert_eq!(completed.output.content, "echo: this is a test");
    }

    #[test]
    fn failed_inference_records_failure() {
        let events = [
            prompt_submitted_event(),
            AiEvent::InferenceFailed {
                event_kappa: kappa("blake3:event-failed"),
                request_kappa: kappa("blake3:request"),
                model_kappa: kappa("blake3:model-manifest"),
                runner_kappa: kappa("blake3:runner-manifest"),
                worker_kappa: kappa("blake3:worker"),
                error: "model execution failed".to_string(),
            },
        ];
        let view = reduce(&events);

        assert!(!view.failed_jobs.is_empty());
        assert!(!view.pending_jobs.contains_key(&kappa("blake3:request")));
        let failure = view
            .failed_jobs
            .get(&kappa("blake3:request"))
            .expect("request should be failed");
        assert_eq!(failure.error, "model execution failed");
    }

    #[test]
    fn reducer_uses_event_order_for_terminal_job_state() {
        let events = [
            prompt_submitted_event(),
            AiEvent::InferenceFailed {
                event_kappa: kappa("blake3:event-failed"),
                request_kappa: kappa("blake3:request"),
                model_kappa: kappa("blake3:model-manifest"),
                runner_kappa: kappa("blake3:runner-manifest"),
                worker_kappa: kappa("blake3:worker"),
                error: "transient failure".to_string(),
            },
            AiEvent::InferenceCompleted {
                event_kappa: kappa("blake3:event-complete"),
                output: completed_output(),
            },
        ];
        let view = reduce(&events);

        assert!(view.failed_jobs.is_empty());
        assert!(view.completed_jobs.contains_key(&kappa("blake3:request")));
    }

    #[test]
    fn started_event_keeps_the_job_pending_but_marks_it_running() {
        let events = [
            prompt_submitted_event(),
            AiEvent::InferenceStarted {
                event_kappa: kappa("blake3:event-started"),
                request_kappa: kappa("blake3:request"),
                model_kappa: kappa("blake3:model-manifest"),
                runner: runner_manifest(),
                worker_kappa: kappa("blake3:worker"),
            },
        ];
        let view = reduce(&events);

        let pending = view
            .pending_jobs
            .get(&kappa("blake3:request"))
            .expect("request should remain pending");
        assert_eq!(pending.phase, PendingPhase::Running);
        assert_eq!(
            pending
                .runner
                .as_ref()
                .expect("runner should be recorded")
                .name,
            "local-holo-engine"
        );
    }

    #[test]
    fn deterministic_echo_runner_returns_a_replayable_output() {
        let runner = DeterministicEchoRunner::new(
            RunnerManifest {
                runner_kappa: kappa("blake3:test-runner"),
                name: "echo".to_string(),
                kind: RunnerKind::TestEcho,
            },
            kappa("blake3:test-worker"),
            kappa("blake3:test-output"),
        );

        let output = runner.run(&request()).expect("echo runner should succeed");

        assert_eq!(output.content, "this is a test");
        assert_eq!(output.output_kappa, kappa("blake3:test-output"));
        assert_eq!(output.provenance.runner_kappa, kappa("blake3:test-runner"));
        assert_eq!(output.provenance.worker_kappa, kappa("blake3:test-worker"));
    }
}
