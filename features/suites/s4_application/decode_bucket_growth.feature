@row:decode-bucket-growth @stage:S4 @status:build @executor:browser
Feature: A decode session that outgrows its bucket keeps streaming in the browser
  The decode plan carries K/V in a fixed-size bucket of past rows; when a
  sequence outgrows it the bucket regrows geometrically (64 → 128 → …). That
  regrowth is a residency HANDOFF, never a stack: before the wider bucket's
  stages are compiled and materialized, the outgoing runner's resident stages
  (and the stale prefill seeder) are freed, so the grow-only wasm linear memory
  never holds the old resident set AND the new bucket's compilation at once —
  the over-commit that aborted a large model's first growth with a bare
  `RuntimeError: unreachable`. The residency INVARIANT (footprint zero at the
  instant each wider bucket is built) is witnessed natively
  (`tests/decode_growth_residency.rs`); this witnesses the BROWSER path — that
  the real wasm decode session actually crosses a bucket boundary and streams
  on, so the growth code we ship is exercised end to end, not just reasoned
  about. The fixture is staged at its OWN context (128) rather than the
  aggressive stage knob that shrinks it to 64, so a prompt just under the
  initial 64-row bucket grows to 128 mid-turn — the same transition the deployed
  model took.

  Background:
    Given the app is open in the browser against the hermetic model server
    And the fixture is staged at its full context

  Scenario: a warm transcript outgrows the initial bucket and keeps streaming
    When the fixture model is downloaded
    And the user sends handshake message 1
    And the user sends handshake message 2
    Then the decode bucket regrew to a wider window during the journey
    And assistant turn 2 streams a non-empty completion
