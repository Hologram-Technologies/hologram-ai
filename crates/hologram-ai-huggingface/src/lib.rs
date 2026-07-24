//! Model-source acquisition for hologram-ai.
//!
//! `ModelSourceProvider` implementations: pinned Hugging Face download into a
//! content-addressed, locked, resumable cache, and local-directory sources.
//! Networking lives here and only here — never in the runtime path.

#![forbid(unsafe_code)]
