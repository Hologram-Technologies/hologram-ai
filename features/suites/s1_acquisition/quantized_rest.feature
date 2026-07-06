@row:quantized-rest @stage:S1 @status:build @executor:browser @lane:default
Feature: A quant-tiered model rests quantized
  The Rest axis applied to the quantized tier (row `quantized-transit`,
  stated per model by the catalogue — data, never code, and never silent):
  the download derives each fully-retirable projection's matmul-ready int8
  artifact IN THE BROWSER and evaporates its wide blob. The wide form
  transits once and rests nowhere; recorded provenance keeps it recoverable,
  and a missing artifact re-derives at session warm, fail-closed on its
  recorded κ (derive-as-recovery). The κ-store holds the quantized form —
  roughly a quarter of the wide bytes — and the chat journey runs on it,
  narrating the tier.

  Background:
    Given the app is open in the browser against the hermetic model server
    And a forced single-layer execution window
    And the quantized tier is forced

  Scenario: the download derives artifacts and the wide forms go gas-phase
    When the fixture model is downloaded
    Then the κ-store holds every quantized artifact and no gas-phase wide blob

  Scenario: chat runs on the quantized tier and says so
    Given the fixture model is downloaded
    When a single-turn prompt is sent
    Then a non-empty completion streams back
    And the session narration states the quantized tier

  Scenario: a saturated quota degrades the tier, never the journey
    Given the origin's storage quota is capped below the model's size
    And the fixture model is downloaded
    Then the download narrated the quantized tier without erroring
    When a single-turn prompt is sent
    Then a non-empty completion streams back
    And the session narration states the quantized tier
