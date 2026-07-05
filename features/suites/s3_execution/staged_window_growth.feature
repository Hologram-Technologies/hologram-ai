@row:staged-window-growth @stage:S3 @status:build @executor:rust @lane:default
Feature: The staged window follows the sequence, never the model
  A staged pipeline compiled once at the model's full context makes every
  token of a short prompt pay a full-context forward pass — O(context²)
  attention per layer, a months-long "hang" for a chat message against a
  32k-context model in a browser tab. But stage archives are weightless
  k-forms: recompiling them at a smaller window moves no weights. The
  growable staged session therefore serves geometric window buckets that
  track the running sequence — the same policy as the monolithic growable
  session — capped at the model's own context, with peak weight residency
  still one stage. The window is a function of the SEQUENCE; the model's
  context is a ceiling, never a cost.

  Background:
    Given the deterministic tiny decoder fixture with its weights in a κ-store

  Scenario: a short prompt executes in a sequence-sized window
    When a short prompt is generated through the growable staged session
    Then the served window is the smallest geometric bucket holding the sequence
    And the served window is smaller than the model's context length

  Scenario: the window regrows geometrically as the sequence crosses a bucket
    When generation pushes the sequence across a window bucket boundary
    Then the session recompiles the stages exactly once per crossed bucket
    And the window never exceeds the model's context length

  Scenario: growable staged generation equals the fixed-window completion
    When the same greedy completion is generated through the growable staged session and the fixed-window staged runner
    Then both completions are identical and non-empty

  Scenario: a sequence beyond the model's context fails loud
    When a growable staged session is asked for a window past the model's context
    Then the refusal names the requested window and the model's context length
