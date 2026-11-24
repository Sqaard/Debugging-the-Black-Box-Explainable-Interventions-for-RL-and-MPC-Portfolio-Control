# Debugging the Black Box

An open course project implementing and evaluating explainable interventions for reinforcement-learning and model-predictive portfolio control.

The project compares a tabular Q-learning challenger with discrete receding-horizon MPC on real US market and macro data. It includes primitive discovery, a frozen-policy intervention, matched-random controls, state-dependent robust MPC, equal-weight ensemble MPC, and minimax robust MPC.

## Quick start

Run from this repository root:

```powershell
python run_experiment.py
python run_extra_mpc.py
```

The scripts expect the real-market input panel at `../data/_dow_extended` and FRED/ALFRED parquet files at `../data/_ext_2009_2026/fred`. The original working repository contains the full input data; this repository stores the implementation, LaTeX report, figures, and generated result artifacts.

## What is implemented

- RL baseline: tabular Q-learning with 18 causal stress states and six allocation actions.
- RL repair: a train-only KMeans behavior primitive mapped to one frozen execution rule.
- MPC baseline: rolling 60-day mean and covariance with turnover penalty.
- State-dependent robust MPC: VIX-linked uncertainty penalty and stress equity constraint.
- Ensemble MPC: equal-weight 20/60/120-day rolling estimators.
- Minimax robust MPC: worst-case score over the same three estimators.

## Reproducibility

The primary result artifacts are in `artifacts/stage0` through `artifacts/stage9`. Additional MPC outputs are in `artifacts/xrl_mpc_extra_models_20261124` and `artifacts/stage9/xrl_mpc_extra_models_20261124`.

The LaTeX course report is `latex_article/main.tex`; the compiled PDF is `latex_article/main.pdf`.

The frozen test is used once for the before/after comparison. The outer holdout is exploratory. Results are research-only and do not certify an investment policy.

## Repository layout

```text
run_experiment.py       Main RL, MPC, primitive, and intervention pipeline
run_extra_mpc.py        Ensemble and minimax robust MPC experiments
latex_article/          Course report source, figures, and compiled PDF
artifacts/               Stage outputs and machine-readable result tables
```

## References

The report cites work on forecast combinations, covariance shrinkage, EWMA risk forecasts, constrained MPC, robust MPC, worst-case portfolio risk, and multi-period portfolio MPC. The README structure follows the practical documentation principles described in the [Art of README](https://github.com/hackergrrl/art-of-readme) and the Russian overview [Как писать README](https://habr.com/ru/articles/810537/).
