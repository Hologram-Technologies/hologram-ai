@row:supported-search @stage:S1 @status:build @executor:browser
Feature: Derivability-preflighted model discovery
  Search runs the SAME config preflight the download journey runs (the wasm
  binding over the parametric registry + derivation — never a name list in the
  app). A derivable model is selectable; a model the preflight refuses stays
  VISIBLE, greyed out, annotated with the refusal reason verbatim — an honest
  refusal is information, not something to hide.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: search surfaces derivable models and annotates refusals verbatim
    When searching the catalog for "tiny"
    Then the search results include the supported fixture model
    And the unsupported-family model appears refused with the preflight reason
