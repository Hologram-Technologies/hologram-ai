@row:safetensors-header-streaming @stage:S1 @status:verified @executor:rust @lane:default
Feature: Safetensors header streaming
  Files produced by the reference `safetensors` crate (the format authority)
  stream-parse through our import path — 8-byte little-endian header length,
  JSON header, tensor byte ranges — to identical tensor names, dtypes,
  shapes, and data bytes. The parse under test is the same header-walking
  code the streamed acquisition path uses; the reference crate's own view of
  the identical file is the oracle it is diffed against.

  Scenario: a reference-crate file streams to identical tensors
    Given a multi-tensor safetensors file serialized by the reference crate
    When the file is stream-parsed by the import path
    Then every tensor name, dtype, shape, and data range matches the reference crate's view
