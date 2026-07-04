@row:parametric-graph @stage:S2 @status:build @executor:rust @lane:default
Feature: The parametric decoder graph
  The decoder graph is a function of config.json plus the tensor manifest
  alone: hidden size, layers, heads, KV heads, head dim, vocabulary,
  rope_theta, rms_norm_eps, weight tying, context length, and tensor dtypes
  all come from the model's own published configuration. The architecture
  family registry selects the builder from `config.architectures[0]`; an
  unsupported family fails loud naming the family and the supported set.
  Validated against the deterministic quantities of the arbitrary
  handshake-tiny use-case (model/usecases.toml) — never a canonical constant.

  Scenario: config quantities flow into the graph
    Given a Llama-family config with the handshake-tiny quantities and an untied manifest
    When the parametric graph is built from config and manifest alone
    Then every RmsNorm epsilon equals the config's rms_norm_eps
    And every attention node carries the config's heads, KV heads, head dim, and rope_theta
    And the graph declares a separate lm_head weight
    And the graph metadata carries the model's own context length

  Scenario: weight tying reuses the embedding weight
    Given a Llama-family config with the handshake-tiny quantities and a tied manifest
    When the parametric graph is built from config and manifest alone
    Then the graph declares no separate lm_head weight
    And the embedding weight feeds both the token gather and the head projection

  Scenario: an unsupported family fails loud naming the family
    Given a config naming the architecture family "MambaForCausalLM"
    When the parametric graph build is attempted
    Then the build fails naming "MambaForCausalLM" and the supported families
