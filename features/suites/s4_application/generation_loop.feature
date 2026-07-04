@row:generation-loop @stage:S4 @status:build @executor:rust @lane:default
Feature: The generation loop is re-execution
  Autoregressive generation is re-execution of the compiled graph with
  streamed token emission and config-driven sampling: each decode step feeds
  the grown token sequence back through the same loaded runner — no
  KV-cache machinery, no per-step recompilation, no mutable cache object.
  Validated against a deterministic reference: a tiny compiled language
  model whose logits are a known function of the input tokens, decoded
  greedily, must emit the successor sequence identically on every run.

  Scenario: greedy decode emits deterministic tokens across runs
    Given a tiny compiled language model whose logits depend on the input tokens
    When greedy decoding runs for 4 steps twice from the same prompt
    Then each run emits 4 tokens
    And both runs emit the identical token sequence
    And the emitted tokens follow the model's deterministic successor table
