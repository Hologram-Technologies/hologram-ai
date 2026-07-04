@row:arbitrary-architecture-coverage @stage:S2 @status:open @executor:rust @lane:default @target
Feature: Architecture-family coverage of the parametric registry, measured
  How much of the HuggingFace Hub's architecture space the parametric family
  registry covers is a genuine unknown: it is measured and reported here,
  never asserted universal. "Arbitrary" means parametric over the family
  registry and the model's own configuration — not every architecture on the
  Hub. The probe instantiates the registry against a fixed list of common
  Hub architecture strings and reports the supported and unsupported counts.

  Scenario: the registry is probed against common Hub architectures
    Given a fixed list of common HuggingFace architecture families
    When each family is probed against the parametric registry
    Then the supported and unsupported counts are reported for every probed family
