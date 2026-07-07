@row:chunked-prefill @stage:S3 @status:build @executor:rust @lane:default
Feature: Prefill amortizes the weight stream across a chunk
  The measured decode residual is weights-side: a single-position pass
  streams the full weight set, so an n-token prefill pays that stream n
  times. The decode plan generalized to seq = C processes C positions per
  pass over the same carried bucket — one weight stream per chunk, not per
  token. Intra-chunk causality (no longer vacuous at C > 1) enters through
  the same additive mask that erases unrealized bucket rows; rope tables
  arrive pre-expanded to the plan's head-major row layout, exact-shape
  arithmetic with zero broadcast assumptions. A partial final chunk PADS:
  padded rows land above the realized length, where the mask makes them
  unreachable until real content overwrites them — sound by the same law
  that makes a fixed bucket sound.

  Scenario: a chunk-seeded prefill is indistinguishable from a stepped one
    Given a staged decode pipeline over the staged fixture with a bucket of 64 rows
    When the fixture transcript prefills through a chunk-4 seeder and through steps
    Then the seeded session matches the stepped session in ceil-n-over-chunk passes
