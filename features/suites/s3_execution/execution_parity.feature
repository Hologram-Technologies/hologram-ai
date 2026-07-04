@row:execution-parity @stage:S3 @status:verified @executor:rust @lane:model
Feature: Execution parity with ONNX Runtime on the pinned model
  The materialized parametric pipeline reproduces ONNX Runtime's logits for
  the pinned SmolLM2-135M-Instruct export within tolerance, and greedy
  decoding agrees with ORT token for token. The oracles are ONNX Runtime
  v1.18.1 and the pinned model revision. This lane runs where the pinned
  model is on disk; an absent model fails loud naming the expected location.

  Background:
    Given the pinned SmolLM2 export on disk

  Scenario: prefill logits match ONNX Runtime within tolerance
    When the prompt "The capital of France is" is executed by hologram and by ONNX Runtime
    Then the last-position logits agree within tolerance
    And both engines agree on the greedy next token

  Scenario: greedy continuation matches ONNX Runtime token-for-token
    When both engines greedily decode 8 tokens from "The capital of France is"
    Then the decoded continuations are identical
