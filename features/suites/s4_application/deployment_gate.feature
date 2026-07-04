@row:deployment-gate @stage:S4 @status:build @executor:rust @lane:default
Feature: The Pages deployment requires the journey gate
  The Pages deployment cannot publish without the browser journey suites
  green: in the deploy workflow, the job that publishes to GitHub Pages
  declares a `needs:` dependency on the `journey` job, and the workflow
  triggers on pushes to the default branch — so every published site has
  passed the journey. The witness reads the committed workflow definition
  itself; the deterministic reference is the workflow file in this
  repository.

  Scenario: the deploy workflow gates publishing on the journey
    Given the Pages deployment workflow
    Then the workflow triggers on pushes to the default branch
    And the publish job requires the "journey" job
