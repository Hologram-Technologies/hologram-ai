@row:decode-plan @stage:S3 @status:build @executor:rust @lane:default
Feature: The per-token pass computes one position
  The measured decode frontier: layer kernels span the whole compiled
  window, so a step costs a window-sized forward however much of it is
  elided. The decode plan removes the window from the step instead of
  skipping inside it — the same decoder recipe emitted at seq = 1, every
  fused attention node decomposed into masked past-attention over a fixed
  bucket of carried rows. Carried K/V is derived content through named
  ports: past_k/past_v enter as inputs, k_new/v_new leave as outputs, and
  the engine splices each step's rows into its buffers between steps.
  Positions are runtime data — rope_cos/rope_sin tables synthesized at the
  absolute position and the additive decode_mask that erases unrealized
  bucket rows inside the softmax — so one compiled artifact serves every
  step, and bucket exhaustion is a geometric recompile, never a ceiling.

  Scenario: the decode plan matches the whole-window plan at every position
    Given a decode-step archive over the staged fixture with a bucket of 8 rows
    When the fixture tokens replay through the decode plan one position at a time
    Then every decode step's logit row matches the whole-window plan at its position

  Scenario: an exhausted bucket regrows without changing the numbers
    Given a decode-step archive over the staged fixture with a bucket of 2 rows
    When the fixture tokens replay through the decode plan one position at a time
    Then every decode step's logit row matches the whole-window plan at its position
    And the bucket regrew geometrically while the carried rows stayed intact
