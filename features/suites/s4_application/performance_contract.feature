@row:performance-contract @stage:S4 @status:open @executor:rust @lane:default @target
Feature: Performance contract, measured
  Throughput, compile time, and reuse floors are genuine unknowns per
  environment: they are measured and reported here, never asserted as
  invariants. The probe compiles the tiny deterministic language model,
  times a fixed number of greedy decode steps on the pinned substrate, and
  reports compile time, tokens per second, and the dispatched and skipped
  kernel counts. The decode floor is the memory roofline — the time to stream
  a pass's weight bytes once at the measured bandwidth — so the one in-repo
  lever over that floor is streaming FEWER bytes: the quantized decode probe
  measures the int8 weight-byte reduction (a strictly lower floor) and confirms
  the int8 walk still runs. int8 is a LOSSY tier — its greedy completion may
  diverge from F32 on a tiny synthetic fixture, so the match is reported, never
  asserted (real-scale faithfulness is the deployed journey's job, against the
  model's own tokenizer). The residual above the reduced floor is the
  substrate's single-position kernel throughput, which this repository does
  not own.

  Scenario: decode throughput, compile time, and reuse are measured
    Given a tiny compiled language model whose logits depend on the input tokens
    When 32 decode steps are timed
    Then the compile time, tokens per second, and reuse counters are reported

  Scenario: the staged decode ratio is measured against the calibrated floor
    Given a staged decoder fixture in a κ-store for the performance probe
    When the environment stream bandwidth is calibrated and 8 staged decode steps are timed
    Then the per-token decode ratio and its attribution are reported

  Scenario: the decode-plan ratio is measured against the same floor
    Given a decode-step archive over the staged fixture with a bucket of 64 rows
    When the environment stream bandwidth is calibrated and 8 decode-plan steps are timed
    Then the decode-plan ratio and its attribution are reported

  Scenario: quantized decode lowers the weight-streaming floor
    Given a quantized decode-step archive over the quantizable fixture with a bucket of 64 rows
    When the environment stream bandwidth is calibrated and 8 quantized decode-plan steps are timed
    Then the quantized decode-plan floor and its reduction against the F32 stream are reported
