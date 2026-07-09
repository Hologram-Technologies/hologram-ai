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
  That recompile hands off residency rather than stacking it: before the wider
  bucket's stages are compiled and materialized, the OUTGOING runner's resident
  stages (and the now-stale prefill seeder) are freed, so the grow-only wasm
  linear memory never holds the old resident set AND the new bucket's
  compilation at once — the over-commit that otherwise aborts a large model's
  first growth with a bare `RuntimeError: unreachable`. Witnessed natively
  (`tests/decode_growth_residency.rs`): the resident footprint is zero at the
  instant each wider bucket is built. A fresh turn also avoids the
  regrow it can foresee: the initial bucket holds the prompt AND the generation
  the caller DECLARED, so a turn of known length is sized once instead of
  re-materializing every stage part-way through. An UNDECLARED budget may run to
  the model's context, and pinning a context-sized K/V is impossible at scale, so
  it starts at the prompt's window and climbs the ladder — which is what the
  ladder is for. The rule is the caller's own numbers (prompt, declared budget,
  context) and nothing else: a twelve-token prompt and a million-token prompt
  take the identical path.

  Scenario: the decode plan matches the whole-window plan at every position
    Given a decode-step archive over the staged fixture with a bucket of 8 rows
    When the fixture tokens replay through the decode plan one position at a time
    Then every decode step's logit row matches the whole-window plan at its position

  Scenario: an exhausted bucket regrows without changing the numbers
    Given a decode-step archive over the staged fixture with a bucket of 2 rows
    When the fixture tokens replay through the decode plan one position at a time
    Then every decode step's logit row matches the whole-window plan at its position
    And the bucket regrew geometrically while the carried rows stayed intact

  Scenario: the staged decode pipeline equals the monolithic decode plan
    Given a staged decode pipeline over the staged fixture with a bucket of 8 rows
    When the fixture tokens replay through both decode plans one position at a time
    Then every staged decode step is byte-identical to the monolithic decode step

  Scenario: greedy completions are identical across the decode and whole-window plans
    Given a staged decode pipeline over the staged fixture with a bucket of 64 rows
    When the same greedy completion is generated through the decode plan and the whole-window plan
    Then both plans emit the identical completion

  Scenario: a turn extending the transcript pays only its novel suffix
    Given a staged decode pipeline over the staged fixture with a bucket of 64 rows
    When two chat turns extend one transcript through the decode session
    Then the second turn steps only its suffix and matches a fresh replay
