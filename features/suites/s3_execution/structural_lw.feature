@row:structural-lw @stage:S3 @status:verified @executor:rust @lane:ort
Feature: Structural class LW — every lowering is held to an external reference
  Every desugared lowering is held to the external reference of the op it
  replaces: the lowering surface is total and each canonical realization is
  diffed against an independent reference implementation, with ONNX Runtime
  anchoring the operator semantics on this lane. The witness is the
  substrate-contract test `structural_lw`, run in its own isolated release
  process — the runner asserts that process exits green.

  @serial
  Scenario: the LW witness is green in an isolated process
    When the structural witness "structural_lw" runs in an isolated release process
    Then the witness process exits green
