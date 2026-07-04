@row:app-domain-events @stage:S4 @status:build @executor:rust @lane:default
Feature: The app-domain projection is pure and replayable
  reduce(events) is a pure, deterministic, order-respecting fold with no
  side effects: the same event stream always projects to the same AiView;
  events on independent requests commute; and on the same request the later
  terminal event wins. Content identity is preserved — a manifest registered
  under a κ-label is projected unchanged under that κ.

  Scenario: the same event stream reduces to the same view
    Given an event stream covering registration, submission, start, completion, and failure
    When every stream is reduced
    Then all reductions project the identical view

  Scenario: independent events commute
    Given two interleavings of the same events on two independent requests
    When every stream is reduced
    Then all reductions project the identical view

  Scenario: order decides the terminal state of a request
    Given a stream where a request fails and then completes
    When the stream is reduced
    Then the request is completed, not failed

  Scenario: a registered model manifest preserves its κ
    Given a model manifest carrying a κ-label
    When the manifest is registered and the stream is reduced
    Then the view holds the manifest under its κ unchanged
