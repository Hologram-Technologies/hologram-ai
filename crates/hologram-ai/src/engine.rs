//! Length-adaptive inference sessions — the UOR-native streaming architecture
//! for **arbitrary-length** input and output on a static-shape backend.
//!
//! hologram compiles a graph at one concrete sequence length, so a single
//! `.holo` runs at a fixed window. To support an arbitrarily long prompt and an
//! arbitrarily long generated continuation without an artificial cap, generation
//! is driven through a [`SessionProvider`]: it yields an [`HoloRunner`] whose
//! window is at least the requested length, growing on demand.
//!
//! Two providers:
//!
//! - [`FixedSession`] — one precompiled `.holo`. Its window is the baked
//!   `seq_len`; a request beyond it is a clear error (recompile larger, or run
//!   the model source for an auto-growing window).
//!
//! - [`GrowableSession`] — holds the imported + optimized [`PreparedModel`]
//!   (the length-independent prefix of compilation) and compiles a concrete
//!   window on demand. Windows grow geometrically up to the model's real
//!   `context_length`, and only the current window is kept resident
//!   (generation grows monotonically, so a smaller window is never needed
//!   again — dropping it bounds memory to one session). Recompiling a window
//!   skips import + optimization (≈⅓ of compile cost, and length-independent),
//!   so growth costs only the per-length concretize → lower → compile.
//!
//! Within a window, autoregressive reuse is content-addressed κ-label elision
//! (the unchanged prefix is recognized by address and not recomputed) — the
//! UOR-native replacement for a mutable KV-cache (architecture §5.3, class CE).

use anyhow::{bail, Context, Result};

use crate::compiler::PreparedModel;
use crate::runner::HoloRunner;

/// Smallest window a [`GrowableSession`] compiles — avoids tiny graphs and gives
/// short prompts room to generate a few tokens before the first regrow.
const MIN_WINDOW: usize = 64;

/// Yields an inference session whose token window is at least `want` long.
///
/// Generation calls [`Self::session_for`] each step with the current window
/// length; the provider returns a session that can run it (growing/recompiling
/// if needed). [`Self::max_window`] is the ceiling the window may reach — the
/// model's trained context for a growable provider, or the baked `seq_len` for a
/// fixed one — beyond which the caller slides the window.
pub trait SessionProvider {
    /// A session whose `input_ids` length (the compiled window) is ≥ `want`.
    fn session_for(&mut self, want: usize) -> Result<&mut HoloRunner>;

    /// The largest window this provider can serve. Generation never requests
    /// more than this; longer sequences slide within it.
    fn max_window(&self) -> usize;
}

/// Read the compiled token-window length (the `input_ids` element count) from a
/// loaded runner. Falls back to the largest input port when there is no
/// `input_ids` (non-LM graphs), and to 0 if there are no inputs.
fn window_of(runner: &HoloRunner) -> usize {
    if let Some(i) = runner.input_index_by_name("input_ids") {
        return runner.input_port_info()[i].element_count;
    }
    runner
        .input_port_info()
        .iter()
        .map(|p| p.element_count)
        .max()
        .unwrap_or(0)
}

/// A single precompiled `.holo` — a fixed-window provider.
pub struct FixedSession {
    runner: HoloRunner,
    seq_len: usize,
}

impl FixedSession {
    /// Wrap a loaded runner; its window is the compiled `input_ids` length.
    pub fn new(runner: HoloRunner) -> Self {
        let seq_len = window_of(&runner);
        Self { runner, seq_len }
    }
}

impl SessionProvider for FixedSession {
    fn session_for(&mut self, want: usize) -> Result<&mut HoloRunner> {
        if want > self.seq_len {
            bail!(
                "the sequence needs a window of {want} tokens but this archive was compiled at a \
                 fixed seq_len of {}; recompile with a larger `--seq-len`, or run the model \
                 source (the .onnx or its directory) directly for an auto-growing window",
                self.seq_len
            );
        }
        Ok(&mut self.runner)
    }

    fn max_window(&self) -> usize {
        self.seq_len
    }
}

/// A length-adaptive provider: compiles the window on demand from a retained
/// prepared (imported + optimized) model, growing geometrically up to the
/// model's `context_length` and keeping only the current window resident.
///
/// The prepared model is imported + optimized **once** and held; each window is
/// minted by cloning it and `compile_window` (consuming the clone), so a regrow
/// never re-parses the source (the protobuf parse is the largest transient). The
/// old window is dropped before the new one compiles, so peak memory is the
/// prepared template + one compile, not two sessions.
pub struct GrowableSession {
    /// Imported + optimized model, cloned to mint each window.
    prepared: PreparedModel,
    /// The model's trained context length — the window ceiling.
    max_window: usize,
    /// The currently-resident `(window_len, session)`, if any window is compiled.
    current: Option<(usize, HoloRunner)>,
}

impl GrowableSession {
    /// Build from a prepared model (import + optimize already done once). The
    /// window grows up to the model's real `context_length`.
    pub fn new(prepared: PreparedModel) -> Self {
        // The ceiling is the model's real trained context — never more (positions
        // beyond it are out of the model's range), never artificially less.
        let max_window = (prepared.context_length() as usize).max(1);
        Self {
            prepared,
            max_window,
            current: None,
        }
    }

    /// The smallest window ≥ `want`: geometric doubling from [`MIN_WINDOW`],
    /// capped at the model's context. Doubling keeps regrows to O(log N) over a
    /// long generation, so the ~per-length compile cost is paid a few times, not
    /// per token.
    fn window_for(&self, want: usize) -> usize {
        let want = want.clamp(1, self.max_window);
        let mut w = MIN_WINDOW;
        while w < want {
            w = w.saturating_mul(2);
        }
        w.min(self.max_window)
    }
}

impl SessionProvider for GrowableSession {
    fn session_for(&mut self, want: usize) -> Result<&mut HoloRunner> {
        // Reuse the resident window if it still fits; generation only grows, so
        // once we regrow the smaller window is gone for good (bounds memory).
        let fits = matches!(&self.current, Some((cur, _)) if *cur >= want);
        if !fits {
            let window = self.window_for(want);
            tracing::info!(window, want, "compiling generation window");
            // Drop the previous window first so peak resident memory is the
            // prepared template + one compile, not two sessions.
            self.current = None;
            let archive = self
                .prepared
                .clone()
                .compile_window(window as u64)
                .with_context(|| format!("compiling a {window}-token window"))?;
            let runner = HoloRunner::from_bytes(archive.bytes)
                .context("loading the freshly-compiled window archive")?;
            self.current = Some((window, runner));
        }
        Ok(&mut self.current.as_mut().expect("window just ensured").1)
    }

    fn max_window(&self) -> usize {
        self.max_window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A growable session's bucket policy is a pure function — test it directly
    // without compiling, so the contract (geometric, capped) is pinned cheaply.
    struct Policy {
        max_window: usize,
    }
    impl Policy {
        fn window_for(&self, want: usize) -> usize {
            let want = want.clamp(1, self.max_window);
            let mut w = MIN_WINDOW;
            while w < want {
                w = w.saturating_mul(2);
            }
            w.min(self.max_window)
        }
    }

    #[test]
    fn windows_grow_geometrically_and_cap() {
        let p = Policy { max_window: 8192 };
        assert_eq!(p.window_for(1), 64);
        assert_eq!(p.window_for(64), 64);
        assert_eq!(p.window_for(65), 128);
        assert_eq!(p.window_for(200), 256);
        assert_eq!(p.window_for(1000), 1024);
        // Capped at the model context, never above it.
        assert_eq!(p.window_for(9000), 8192);
        assert_eq!(p.window_for(8192), 8192);
    }

    #[test]
    fn small_context_caps_below_min_window() {
        // A model with a tiny context caps there (never grows past the model).
        let p = Policy { max_window: 32 };
        assert_eq!(p.window_for(1), 32);
        assert_eq!(p.window_for(100), 32);
    }
}
