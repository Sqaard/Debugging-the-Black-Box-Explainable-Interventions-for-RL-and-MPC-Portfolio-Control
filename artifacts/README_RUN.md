# XRL MPC real market experiment

Run: `xrl_mpc_real_market_20261124`

Data: local `_dow_extended` panel, cross-checked against `/home/vanya/projects/trustbench_stress_vanya/data/data_RLagent_for_Joseph/data/_dow_extended`.

The RL challenger is tabular Q-learning over causal stress states; the MPC controller uses a rolling mean-covariance estimate and a discrete action grid. Stage 1 primitives are discovered on train+validation only. Stage 4 is locked before intervention. Stage 8 reports matched-random and opposite-direction controls.

Evidence status: research-only. The frozen test is used once for the A-C comparison; no product or client claim is made.
