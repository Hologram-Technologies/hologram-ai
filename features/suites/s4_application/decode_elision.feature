@row:decode-elision @stage:S4 @status:build @executor:rust @lane:default
Feature: Decode elision — the measured k-scaling witness
  Consecutive decode steps report elided (skipped) dispatches for the
  unchanged prefix cone: a decode step re-executes the compiled graph, and
  the pinned substrate skips every node whose output κ-label is already
  resident. This is the structural reuse that replaces a mutable KV-cache.
  The witness reads the session's own dispatch and skip counters step by
  step and prints them.

  Scenario: consecutive decode steps skip the unchanged prefix cone
    Given a tiny compiled language model whose logits depend on the input tokens
    When greedy decoding runs for 4 steps reporting dispatch counters
    Then step 1 dispatches at least one kernel
    And every step from 2 on reports skipped dispatches for the unchanged prefix
    And the per-step dispatched and skipped counts are printed
