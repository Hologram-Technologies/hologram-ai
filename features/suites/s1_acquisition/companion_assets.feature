@row:companion-assets @stage:S1 @status:build @executor:browser
Feature: Companion assets persist with the model
  config.json, tokenizer.json, and generation_config.json are fetched and
  persisted under the model's directory, byte-identical to the source.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: companions are persisted byte-identical
    When the fixture model is downloaded
    Then "config.json" is persisted under the model directory byte-identical to the server copy
    And "tokenizer.json" is persisted under the model directory byte-identical to the server copy
