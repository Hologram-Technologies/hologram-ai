@row:chat-model-menu @stage:S4 @status:build @executor:browser
Feature: The chat model menu is populated only by compiled models
  The chat interface never hard-codes a model list: its selection menu is
  driven SOLELY by the compiled archives present in local storage (OPFS). A
  fresh instance offers nothing to chat with; a model the user has merely added
  (or is still downloading) does not appear until it is compiled; and a
  compiled model appears by exactly its on-disk archive — so what the menu
  shows is always, only, what the user actually has.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: only compiled models populate the selection menu
    Given a fresh instance with no compiled models
    Then the chat model menu offers no models to select
    When a model is added to the catalogue but not downloaded
    Then the chat model menu still offers no models to select
    When the fixture model is downloaded
    Then the chat model menu lists exactly the compiled fixture model

  Scenario: removing a stored model clears it from the menu and from storage
    When the fixture model is downloaded
    Then the chat model menu lists exactly the compiled fixture model
    And the stored-models list shows the fixture model
    When the user removes the fixture model from this device
    Then the stored-models list is empty
    And the chat model menu offers no models to select
