//! ONNX protobuf definitions for Hologram.
//!
//! This crate provides the official ONNX protobuf message types
//! compiled from the upstream ONNX specification.

#![allow(clippy::doc_overindented_list_items)]

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
