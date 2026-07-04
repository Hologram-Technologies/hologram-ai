@row:supported-search @stage:S1 @status:build @executor:browser
Feature: Supported-only model discovery
  Search lists only models whose architecture family the parametric registry
  supports — the journey never begins on a model that preflight would reject
  for family support. The supported set comes from the registry itself (the
  wasm binding), never from a hard-coded list in the app.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: search lists supported families and hides unsupported ones
    When searching the catalog for "tiny"
    Then the search results include the supported fixture model
    And the search results do not include the unsupported-family model
    And each search result names its architecture family
