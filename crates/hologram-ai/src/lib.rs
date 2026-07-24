//! hologram-ai: production integration layer between Hugging Face model
//! sources, the `uor-r4` R4G1 compiler/runtime, and Hologram `.holo`
//! applications.
//!
//! Public surface: the compiler builder (source → `.holo`), the application
//! registry (`.holo` → model services), and sessions (invocation, prediction,
//! generation).

#![forbid(unsafe_code)]
