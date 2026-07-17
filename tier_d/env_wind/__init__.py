"""Wind-augmented environment for RL complexity comparison.

Same as tier_d/env/, but with stochastic wind disturbances that change the intended
action with some probability. This breaks the deterministic guarantee and makes
the task genuinely harder — the agent must learn robustness, not just memorize.

Use alongside tier_d/env/ for controlled comparison:
    baseline = train(GridWorld(...))
    wind_trained = train(GridWorldWind(...))
"""
