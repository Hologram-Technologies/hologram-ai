//! Private adapter between hologram-ai and `uor-r4`.
//!
//! All current-upstream compatibility code lives behind this narrow module.
//! Nothing in this crate is re-exported with `uor-r4` types in its signature;
//! the public `hologram-ai` facade is independent of `uor-r4` crate layout.

#![forbid(unsafe_code)]
