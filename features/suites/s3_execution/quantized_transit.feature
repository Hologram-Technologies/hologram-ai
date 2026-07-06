@row:quantized-transit @stage:S3 @status:build @executor:rust @lane:default
Feature: The wide form moves at most once
  A quantized weight form is DERIVED CONTENT — the closure of the known set
  over derivation applied to weights: computed deterministically from the
  wide tensor's κ, addressed by its own κ, persisted in the κ-store like any
  content. The artifact is stored matmul-ready (transposed at derivation,
  per-channel symmetric int8), and stage graphs bind it as two ranged
  sub-tensor κ-bindings — the i8 block and the f32 scales are ranges of one
  content — feeding Dequantize adjacent to its MatMul, the shape the
  substrate fuses. Once the derivation crystallizes, the wide blob goes
  gas-phase: it never re-transits and never re-materializes; recovery from a
  corrupted artifact is re-derivation, fail-closed. Quantization is a
  semantic tier, never silent: the quantized pipeline is its own model whose
  staged and monolithic executions must agree; quality against the wide
  tier is measured, never asserted.

  Background:
    Given a decoder fixture with quantizable projection weights in a κ-store

  Scenario: the quantized derivation is deterministic content
    When the projection weights derive their quantized artifacts twice
    Then re-derivation reproduces each artifact κ bit-identically and every artifact is strictly smaller than its wide form

  Scenario: staged and monolithic execution agree on the quantized tier
    When the quantized stages and the quantized monolithic archive generate from the same prompt
    Then both quantized completions are identical and non-empty

  Scenario: after crystallization the wide form never moves
    When the quantized staged session generates with the wide projection blobs evicted
    Then the completion still produces and no wide projection κ is ever resolved
