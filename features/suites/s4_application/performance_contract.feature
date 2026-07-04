@row:performance-contract @stage:S4 @status:open @executor:rust @lane:default @target
Feature: Performance contract, measured
  Throughput, compile time, and reuse floors are genuine unknowns per
  environment: they are measured and reported here, never asserted as
  invariants. The probe compiles the tiny deterministic language model,
  times a fixed number of greedy decode steps on the pinned substrate, and
  reports compile time, tokens per second, and the dispatched and skipped
  kernel counts.

  Scenario: decode throughput, compile time, and reuse are measured
    Given a tiny compiled language model whose logits depend on the input tokens
    When 32 decode steps are timed
    Then the compile time, tokens per second, and reuse counters are reported
