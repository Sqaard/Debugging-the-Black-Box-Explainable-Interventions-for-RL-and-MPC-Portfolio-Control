"""Additional ensemble and minimax-robust MPC experiments.

The two controllers use the same six-action allocation grid and the same
receding-horizon objective as the baseline MPC. Their uncertainty set is the
three causal rolling estimators L in {20, 60, 120} trading days.
"""
from __future__ import annotations

import json
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_experiment as base


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent if PROJECT.name == "XRL_MPC_project" else PROJECT
RID = "xrl_mpc_extra_models_20261124"
OUT = ROOT / "XRL_MPC_project" / "artifacts" / RID
TOP = ROOT / "artifacts" / "stage9" / RID
WINDOWS = (20, 60, 120)
HORIZON = 5
RISK_AVERSION = 8.0
COST = 0.0006


def score_models(df: pd.DataFrame, i: int, prev_action: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (model_scores, actions) for the six action candidates."""
    returns = df[["r_SPY", "r_TLT", "r_HYG"]].fillna(0.0).to_numpy()
    model_scores = np.zeros((len(WINDOWS), len(base.ACTION_WEIGHTS)), dtype=float)
    for mi, window in enumerate(WINDOWS):
        lo = max(0, i - window)
        hist = returns[lo:i]
        if len(hist) < 10:
            mu = np.zeros(3)
            cov = np.eye(3) * 1e-4
        else:
            mu = hist.mean(axis=0)
            cov = np.atleast_2d(np.cov(hist.T)) + np.eye(3) * 1e-6
        for action, weights in enumerate(base.ACTION_WEIGHTS):
            turnover = np.abs(weights - base.ACTION_WEIGHTS[prev_action]).sum()
            model_scores[mi, action] = HORIZON * (weights @ mu) - RISK_AVERSION * HORIZON * (weights @ cov @ weights) - COST * turnover
    return model_scores, np.arange(len(base.ACTION_WEIGHTS))


def rollout(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    returns = df[["r_SPY", "r_TLT", "r_HYG"]].fillna(0.0).to_numpy()
    actions = []
    values = []
    model_score_rows = []
    for i in range(len(df)):
        prev = actions[-1] if actions else 2
        scores, action_ids = score_models(df, i, prev)
        if mode == "ensemble":
            # Equal-weight combination of objective values; equivalent to
            # applying the objective to averaged mu and Sigma here.
            combined = scores.mean(axis=0)
        elif mode == "minimax":
            # Worst-case over the three model windows. The score already
            # contains the common turnover cost, so it is not double-counted.
            combined = scores.min(axis=0)
        else:
            raise ValueError(mode)
        action = int(action_ids[np.argmax(combined)])
        actions.append(action)
        values.append(float(combined[action]))
        model_score_rows.append(scores[:, action].tolist())
    actions = np.asarray(actions, dtype=int)
    rets = base.portfolio_return(returns, actions)
    out = pd.DataFrame({"date": df.index, "action": actions, "ret": rets, "score": values, "stress": base.stress_mask(df)})
    for j, window in enumerate(WINDOWS):
        out[f"score_L{window}"] = [row[j] for row in model_score_rows]
    return out


def metrics(frame: pd.DataFrame, label: str) -> dict:
    return base.metrics(frame.set_index("date")["ret"], frame["action"].to_numpy(), label)


def main() -> None:
    data = base.load_market()
    splits = {
        "validation": base.make_split(data, base.SPLIT.validation_start, base.SPLIT.validation_end),
        "frozen_test": base.make_split(data, base.SPLIT.frozen_test_start, base.SPLIT.frozen_test_end),
        "outer_holdout": base.make_split(data, base.SPLIT.outer_holdout_start, base.SPLIT.outer_holdout_end),
    }
    rows = []
    trajectories = {}
    for split_name, frame in splits.items():
        for mode in ["ensemble", "minimax"]:
            traj = rollout(frame, mode)
            label = f"MPC {mode} {split_name}"
            rows.append(metrics(traj, label))
            trajectories[f"{split_name}_{mode}"] = traj
    # Include existing controllers on the same splits for an apples-to-apples table.
    for split_name, frame in splits.items():
        baseline = base.mpc_policy(frame, robust=False)
        state_robust = base.mpc_policy(frame, robust=True)
        rows.extend([metrics(baseline, f"MPC baseline {split_name}"), metrics(state_robust, f"MPC state-robust {split_name}")])
        trajectories[f"{split_name}_baseline"] = baseline
        trajectories[f"{split_name}_state_robust"] = state_robust
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    TOP.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "extra_mpc_metrics.csv", index=False)
    result.to_csv(TOP / "extra_mpc_metrics.csv", index=False)
    for name, traj in trajectories.items():
        traj.to_csv(OUT / f"{name}.csv", index=False)
    manifest = {
        "run_id": RID,
        "models": ["ensemble_mpc", "minimax_robust_mpc"],
        "window_days": list(WINDOWS),
        "weights": "equal weights 1/3; fixed before validation and test",
        "horizon": HORIZON,
        "risk_aversion": RISK_AVERSION,
        "turnover_cost": COST,
        "uncertainty_set": "causal rolling mean/covariance estimators with L=20,60,120",
        "selection": "argmax over the same six action candidates; no weights fitted on validation",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (TOP / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Frozen-test comparison plot.
    test = splits["frozen_test"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    plot_names = ["baseline", "state_robust", "ensemble", "minimax"]
    labels = {"baseline": "MPC baseline", "state_robust": "MPC state-robust", "ensemble": "MPC ensemble", "minimax": "MPC minimax robust"}
    colors = {"baseline": "#F28E2B", "state_robust": "#D73027", "ensemble": "#4575B4", "minimax": "#1A9850"}
    for name in plot_names:
        traj = trajectories[f"frozen_test_{name}"]
        ax.plot(pd.to_datetime(traj["date"]), (1 + traj["ret"]).cumprod(), label=labels[name], color=colors[name], lw=1.4)
    ax.set_title("Frozen test: ensemble and minimax MPC")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth, initial value = 1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.annotate("Minimax: cash on 290/290 days; zero interest",
                xy=(pd.Timestamp("2022-10-03"), 1.0),
                xytext=(pd.Timestamp("2022-04-01"), 1.06),
                arrowprops={"arrowstyle": "->", "color": "#1A9850"},
                fontsize=8, color="#176b38")
    fig.tight_layout()
    fig.savefig(OUT / "extra_mpc_frozen_test_wealth.pdf", bbox_inches="tight")
    fig.savefig(TOP / "extra_mpc_frozen_test_wealth.pdf", bbox_inches="tight")
    plt.close(fig)

    # Summary figure: Sharpe and drawdown on frozen test and outer holdout.
    selected = result[result["model"].isin([
        "MPC baseline frozen_test", "MPC state-robust frozen_test", "MPC ensemble frozen_test", "MPC minimax frozen_test",
        "MPC baseline outer_holdout", "MPC state-robust outer_holdout", "MPC ensemble outer_holdout", "MPC minimax outer_holdout",
    ])].copy()
    selected["split"] = selected["model"].str.extract(r"(frozen_test|outer_holdout)$")[0].map({"frozen_test": "Frozen test", "outer_holdout": "Outer holdout"})
    selected["controller"] = selected["model"].str.replace(r" (frozen_test|outer_holdout)$", "", regex=True).str.replace("MPC ", "", regex=False).str.replace("state-robust", "state_robust", regex=False)
    expected = {(split, controller) for split in ["Frozen test", "Outer holdout"]
                for controller in ["baseline", "state_robust", "ensemble", "minimax"]}
    actual = set(zip(selected["split"], selected["controller"]))
    if len(selected) != 8 or actual != expected:
        raise ValueError(f"Incomplete comparison: missing {expected - actual}")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    order = ["baseline", "state_robust", "ensemble", "minimax"]
    pretty = {"baseline": "Baseline", "state_robust": "State robust", "ensemble": "Ensemble", "minimax": "Minimax robust"}
    colors = {"baseline": "#F28E2B", "state_robust": "#D73027", "ensemble": "#4575B4", "minimax": "#1A9850"}
    x = np.arange(len(order))
    width = 0.36
    for j, split in enumerate(["Frozen test", "Outer holdout"]):
        part = selected[selected["split"] == split].set_index("controller").reindex(order)
        if not np.isfinite(part[["sharpe", "max_drawdown"]].to_numpy()).all():
            raise ValueError(f"Non-finite comparison values for {split}")
        axes[0].bar(x + (j - 0.5) * width, part["sharpe"], width, label=split)
        axes[1].bar(x + (j - 0.5) * width, part["max_drawdown"] * 100, width, label=split)
    axes[0].set_title("Sharpe")
    axes[0].set_xticks(x, [pretty[o] for o in order], rotation=25, ha="right")
    axes[0].axhline(0, color="black", lw=0.7)
    axes[1].set_title("Maximum drawdown (%)")
    axes[1].set_xticks(x, [pretty[o] for o in order], rotation=25, ha="right")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[0].legend(frameon=False)
    axes[0].text(3 - width / 2, 0.04, "N/A*", ha="center", va="bottom", fontsize=8)
    fig.text(0.5, 0.015, "* All-cash test path: zero volatility, Sharpe undefined; stored zero is a numerical convention.",
             ha="center", fontsize=8)
    fig.suptitle("Additional MPC experiments")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "extra_mpc_comparison.pdf", bbox_inches="tight")
    fig.savefig(TOP / "extra_mpc_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # Model-dispersion diagnostic: how often minimax differs from ensemble.
    ens = trajectories["frozen_test_ensemble"]
    mm = trajectories["frozen_test_minimax"]
    diagnostic = pd.DataFrame({
        "date": ens["date"],
        "ensemble_action": ens["action"],
        "minimax_action": mm["action"],
        "action_disagreement": (ens["action"] != mm["action"]).astype(int),
        "score_L20": ens["score_L20"],
        "score_L60": ens["score_L60"],
        "score_L120": ens["score_L120"],
    })
    diagnostic.to_csv(OUT / "frozen_test_model_disagreement.csv", index=False)
    diagnostic.to_csv(TOP / "frozen_test_model_disagreement.csv", index=False)
    # Audit every candidate, not only the selected cash score.
    audit_rows = []
    prev = 2
    for i, row in mm.iterrows():
        model_scores, _ = score_models(test, i, prev)
        worst = model_scores.min(axis=0)
        chosen = int(row["action"])
        audit_rows.append({
            "date": row["date"], "previous_action": prev,
            "chosen_action": chosen, "cash_score": worst[0],
            "best_risky_score": worst[1:].max(),
            "cash_margin": worst[0] - worst[1:].max(),
            **{f"worst_score_a{a}": worst[a] for a in range(6)}
        })
        if chosen != int(np.argmax(worst)):
            raise AssertionError(f"Minimax action inconsistent on {row['date']}")
        prev = chosen
    minimax_audit = pd.DataFrame(audit_rows)
    minimax_audit.to_csv(OUT / "minimax_decision_audit.csv", index=False)
    minimax_audit.to_csv(TOP / "minimax_decision_audit.csv", index=False)
    audit_summary = {
        "days": len(mm),
        "cash_days": int((mm.action == 0).sum()),
        "zero_return_days": int((mm.ret == 0).sum()),
        "strict_cash_wins": int((minimax_audit.cash_margin > 0).sum()),
        "min_cash_margin": float(minimax_audit.cash_margin.min()),
        "max_best_risky_score": float(minimax_audit.best_risky_score.max()),
        "initial_cash_score": float(minimax_audit.cash_score.iloc[0]),
        "gross_wealth": "1.000 for all days",
        "accounting": "Stored return excludes transaction costs; score includes them. "
                      "Initial reference action is 2; its one-off move to cash costs "
                      "0.0006 in the score but is not debited from stored wealth. "
                      "Cash earns zero interest. Zero-volatility Sharpe is undefined; "
                      "the legacy metrics helper stores 0 by numerical convention."
    }
    (OUT / "minimax_audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    (TOP / "minimax_audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    summary = {
        "frozen_test_disagreement_rate": float(diagnostic["action_disagreement"].mean()),
        "frozen_test_ensemble": next(r for r in rows if r["model"] == "MPC ensemble frozen_test"),
        "frozen_test_minimax": next(r for r in rows if r["model"] == "MPC minimax frozen_test"),
        "outer_holdout_ensemble": next(r for r in rows if r["model"] == "MPC ensemble outer_holdout"),
        "outer_holdout_minimax": next(r for r in rows if r["model"] == "MPC minimax outer_holdout"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (TOP / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
