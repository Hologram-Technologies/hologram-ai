@row:single-position-head @stage:S3 @status:build @executor:rust @lane:default
Feature: The head computes only the consumed position
  The generation loop consumes exactly one logit row per step, so a
  whole-window head is per-token work that is not decode — a defect by the
  critical-path rule, window-multiplied at real vocabularies. The pipeline
  gathers the consumed position's hidden state after the final norm
  (`last_pos`, a runtime i64 input the generation loop synthesizes as
  `cur_len - 1`, named like every auxiliary) and the head — whole or
  chunked — computes O(vocab·d) per step, never O(window·vocab·d). The
  staged-equals-monolithic and chunked parity witnesses run through the
  same gather; greedy completions are unchanged by construction (the same
  row was always the one consumed).

  Scenario: the pipeline emits one logit row for the consumed position
    Given a staged decoder fixture in a κ-store for the performance probe
    When a staged decode step runs at a mid-window position
    Then the logits are a single vocabulary row and the pipeline declares the position input
