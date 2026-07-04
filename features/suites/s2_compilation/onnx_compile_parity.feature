@row:onnx-compile-parity @stage:S2 @status:verified @executor:rust @lane:ort
Feature: ONNX compile parity against ONNX Runtime
  Compiled ONNX graphs execute to ONNX Runtime's outputs within tolerance.
  The committed fixture is integrity-tracked (oracles/onnx, generated inputs
  whose correctness authority is ORT itself, never our own output); it is
  compiled by the holographic compiler and diffed element-by-element against
  a live ONNX Runtime v1.18.1 session. The official operator node corpus is
  exercised by the conformance suite on the same lane.

  Scenario: the mlp fixture matches ONNX Runtime within tolerance
    Given the external authoritative ONNX fixture "mlp"
    When the fixture is compiled and executed by hologram and by ONNX Runtime
    Then the outputs match ONNX Runtime within tolerance
