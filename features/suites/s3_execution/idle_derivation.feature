@row:idle-derivation @stage:S3 @status:build @executor:rust @lane:default
Feature: Idle time feeds the anneal
  Between turns the session pre-derives entailed work off the critical path:
  the next geometric window bucket's stage archives, derived into the
  derived store while nothing else is waiting. Stage k-forms are weightless,
  so speculation moves no weights, touches no resident state, and never
  competes with the admission probe; a later crossing RESOLVES the window
  instead of compiling it on the per-token path. Abandoned speculation is
  ordinary derived content — evaporable by the same lifecycle that admitted
  it — and nothing speculative is trusted beyond its derived κ.

  Background:
    Given the deterministic tiny decoder fixture with its weights in a κ-store

  Scenario: pre-derivation moves no weights and touches no resident state
    When a turn completes and the session pre-derives the next window bucket
    Then the pre-derivation moved no weights and left the resident window untouched

  Scenario: a later crossing resolves the pre-derived window
    When a turn completes and the session pre-derives the next window bucket
    And a following turn crosses the window bucket boundary
    Then the crossing resolves the window from the derived store instead of compiling
