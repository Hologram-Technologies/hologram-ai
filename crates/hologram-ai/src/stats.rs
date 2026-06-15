//! Lightweight timing surfaces for CLI benchmarking.
//!
//! The current priority is generation latency on compiled causal LMs:
//! prompt encoding, first-step prefill, and steady-state decode throughput.
//! These stats are intentionally host-side and low-overhead so they can be
//! emitted from normal CLI runs without changing runtime behavior.

use std::io::{self, Write};
use std::time::Duration;

/// Timing summary for one autoregressive generation request.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Number of prompt tokens presented to the model.
    pub prompt_tokens: usize,
    /// Number of newly generated tokens emitted by the loop.
    pub generated_tokens: usize,
    /// Host-side tokenizer encode time for the prompt/template text.
    pub prompt_encode: Duration,
    /// Time spent acquiring the first runnable session/window.
    pub prefill_session_prepare: Duration,
    /// Time spent in the first model forward pass over the prompt window.
    pub prefill_forward: Duration,
    /// Time spent acquiring sessions for decode steps after the first token.
    pub decode_session_prepare: Duration,
    /// Time spent in decode-step forward passes after the first token.
    pub decode_forward: Duration,
    /// Total wall-clock time for the full request, including prompt encode.
    pub total: Duration,
}

impl GenerationStats {
    /// Wall-clock latency through the first model decision.
    pub fn time_to_first_token(&self) -> Option<Duration> {
        let total = self.prompt_encode + self.prefill_session_prepare + self.prefill_forward;
        (!total.is_zero()).then_some(total)
    }

    /// Prompt-token throughput of the first forward pass.
    pub fn prefill_tokens_per_second(&self) -> Option<f64> {
        duration_rate(self.prompt_tokens, self.prefill_forward)
    }

    /// Throughput after the first generated token.
    pub fn decode_tokens_per_second(&self) -> Option<f64> {
        let decode_tokens = self.generated_tokens.checked_sub(1)?;
        duration_rate(
            decode_tokens,
            self.decode_session_prepare + self.decode_forward,
        )
    }

    /// End-to-end generated-token throughput over the full request wall time.
    pub fn overall_tokens_per_second(&self) -> Option<f64> {
        duration_rate(self.generated_tokens, self.total)
    }

    /// Print a human-readable summary to `out`.
    pub fn display(&self, out: &mut dyn Write) -> io::Result<()> {
        writeln!(out, "generation stats:")?;
        writeln!(out, "  prompt tokens: {}", self.prompt_tokens)?;
        writeln!(out, "  generated tokens: {}", self.generated_tokens)?;
        writeln!(out, "  prompt encode: {}", fmt_duration(self.prompt_encode))?;
        writeln!(
            out,
            "  prefill session prep: {}",
            fmt_duration(self.prefill_session_prepare)
        )?;
        writeln!(
            out,
            "  prefill forward: {}",
            fmt_duration(self.prefill_forward)
        )?;
        if let Some(ttft) = self.time_to_first_token() {
            writeln!(out, "  time to first token: {}", fmt_duration(ttft))?;
        }
        if let Some(tps) = self.prefill_tokens_per_second() {
            writeln!(out, "  prefill throughput: {:.2} tok/s", tps)?;
        }
        if self.generated_tokens > 1 {
            writeln!(
                out,
                "  decode steps after first: {}",
                self.generated_tokens - 1
            )?;
            writeln!(
                out,
                "  decode session prep: {}",
                fmt_duration(self.decode_session_prepare)
            )?;
            writeln!(
                out,
                "  decode forward: {}",
                fmt_duration(self.decode_forward)
            )?;
            if let Some(tps) = self.decode_tokens_per_second() {
                writeln!(out, "  decode throughput: {:.2} tok/s", tps)?;
            }
        }
        writeln!(out, "  total wall: {}", fmt_duration(self.total))?;
        if let Some(tps) = self.overall_tokens_per_second() {
            writeln!(out, "  overall throughput: {:.2} tok/s", tps)?;
        }
        Ok(())
    }
}

/// Timing summary for a single non-generative forward pass.
#[derive(Debug, Clone, Default)]
pub struct ForwardStats {
    /// Archive/model load time.
    pub load: Duration,
    /// Model execute time.
    pub execute: Duration,
    /// Total wall-clock time for the request.
    pub total: Duration,
}

impl ForwardStats {
    /// Print a human-readable summary to `out`.
    pub fn display(&self, out: &mut dyn Write) -> io::Result<()> {
        writeln!(out, "run stats:")?;
        writeln!(out, "  load: {}", fmt_duration(self.load))?;
        writeln!(out, "  execute: {}", fmt_duration(self.execute))?;
        writeln!(out, "  total wall: {}", fmt_duration(self.total))?;
        Ok(())
    }
}

fn duration_rate(tokens: usize, duration: Duration) -> Option<f64> {
    if tokens == 0 || duration.is_zero() {
        return None;
    }
    Some(tokens as f64 / duration.as_secs_f64())
}

fn fmt_duration(duration: Duration) -> String {
    if duration.as_secs() >= 1 {
        format!("{:.3}s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{:.3}ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.3}us", duration.as_secs_f64() * 1_000_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{fmt_duration, ForwardStats, GenerationStats};
    use std::time::Duration;

    #[test]
    fn generation_stats_compute_throughputs() {
        let stats = GenerationStats {
            prompt_tokens: 8,
            generated_tokens: 5,
            prompt_encode: Duration::from_millis(5),
            prefill_session_prepare: Duration::from_millis(10),
            prefill_forward: Duration::from_millis(200),
            decode_session_prepare: Duration::from_millis(20),
            decode_forward: Duration::from_millis(800),
            total: Duration::from_millis(1_035),
        };

        assert_eq!(
            stats.time_to_first_token().expect("ttft"),
            Duration::from_millis(215)
        );
        assert!((stats.prefill_tokens_per_second().expect("prefill tps") - 40.0).abs() < 0.01);
        assert!((stats.decode_tokens_per_second().expect("decode tps") - 4.878).abs() < 0.01);
        assert!((stats.overall_tokens_per_second().expect("overall tps") - 4.830).abs() < 0.01);
    }

    #[test]
    fn generation_stats_skip_decode_rate_without_second_token() {
        let stats = GenerationStats {
            generated_tokens: 1,
            total: Duration::from_millis(10),
            ..Default::default()
        };
        assert!(stats.decode_tokens_per_second().is_none());
    }

    #[test]
    fn display_formats_are_human_readable() {
        assert_eq!(fmt_duration(Duration::from_secs(2)), "2.000s");
        assert_eq!(fmt_duration(Duration::from_millis(12)), "12.000ms");
        assert_eq!(fmt_duration(Duration::from_micros(250)), "250.000us");

        let mut out = Vec::new();
        ForwardStats {
            load: Duration::from_millis(1),
            execute: Duration::from_millis(2),
            total: Duration::from_millis(3),
        }
        .display(&mut out)
        .expect("display forward stats");
        let rendered = String::from_utf8(out).expect("utf8");
        assert!(rendered.contains("run stats:"));
        assert!(rendered.contains("execute: 2.000ms"));
    }
}
