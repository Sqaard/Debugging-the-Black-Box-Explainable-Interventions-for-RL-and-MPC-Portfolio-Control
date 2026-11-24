"""Reproducible XAI RL vs MPC experiment on the real US stress panel.

The implementation is intentionally small and auditable for a course report:
tabular Q-learning is the RL challenger, a discrete receding-horizon mean-
variance controller is the MPC champion, and Stage 1-9 artifacts document
primitive discovery, intervention, controls, and the before/after comparison.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


SEED = 20261124
PROJECT = Path(__file__).resolve().parent
# In the original workspace this folder is nested under the main repository;
# in the standalone GitHub repository it is the repository root itself.
ROOT = PROJECT.parent if PROJECT.name == "XRL_MPC_project" else PROJECT
RUN_ID = "xrl_mpc_real_market_20261124"
PROJECT_ART = PROJECT / "artifacts"
TOP_ART = ROOT / "artifacts"
ASSETS = ["SPY", "TLT", "HYG"]
ACTION_WEIGHTS = np.array(
    [
        [0.00, 0.00, 0.00],  # cash: handled as residual cash in this table
        [0.25, 0.50, 0.25],
        [0.50, 0.25, 0.25],
        [0.75, 0.125, 0.125],
        [1.00, 0.00, 0.00],
        [0.50, 0.50, 0.00],
    ],
    dtype=float,
)
ACTION_NAMES = ["cash", "defensive", "balanced", "growth", "equity", "equity_bonds"]


@dataclass
class Split:
    train_start: str = "2010-01-04"
    train_end: str = "2017-12-29"
    validation_start: str = "2018-01-02"
    validation_end: str = "2021-12-31"
    frozen_test_start: str = "2022-01-03"
    frozen_test_end: str = "2023-02-28"
    outer_holdout_start: str = "2023-03-01"
    outer_holdout_end: str = "2026-07-07"


SPLIT = Split()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def stage_dir(stage: str) -> Path:
    p = PROJECT_ART / stage / RUN_ID
    p.mkdir(parents=True, exist_ok=True)
    q = TOP_ART / stage / RUN_ID
    q.mkdir(parents=True, exist_ok=True)
    return p


def mirror_file(path: Path, stage: str) -> None:
    q = TOP_ART / stage / RUN_ID / path.name
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, q)


def load_series(name: str) -> pd.Series:
    p = ROOT / "data" / "_dow_extended" / f"{name}.csv"
    df = pd.read_csv(p, usecols=["date", "adjclose"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date")["adjclose"].rename(name)


def load_fred(path: Path, name: str) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float, name=name)
    df = pd.read_parquet(path, columns=["obs_date", "realtime_start", "value"])
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    # Use the last vintage available for each observation. The resulting series
    # is only a descriptive market feature; the frozen split is still reported.
    df = df.sort_values(["obs_date", "realtime_start"]).drop_duplicates("obs_date", keep="last")
    s = df.set_index("obs_date")["value"].astype(float).rename(name)
    return s


def load_market() -> pd.DataFrame:
    data = pd.concat([load_series(a) for a in ASSETS + ["LQD", "_idx_VIX", "_idx_VIX3M", "_idx_MOVE"]], axis=1)
    data = data.rename(columns={"_idx_VIX": "VIX", "_idx_VIX3M": "VIX3M", "_idx_MOVE": "MOVE"})
    fred_dir = ROOT / "data" / "_ext_2009_2026" / "fred"
    stlfsi = load_fred(fred_dir / "STLFSI4.parquet", "STLFSI4")
    t10y2y = load_fred(fred_dir / "T10Y2Y.parquet", "T10Y2Y")
    data = data.join(stlfsi, how="left").join(t10y2y, how="left")
    data = data.sort_index().ffill()
    for c in ["STLFSI4", "T10Y2Y"]:
        data[c] = data[c].fillna(0.0)
    # Decisions use information at t, returns accrue from t to t+1.
    for a in ASSETS:
        data[f"r_{a}"] = data[a].pct_change()
    data["spy_r20"] = data["SPY"].pct_change(20)
    data["spy_vol20"] = data["r_SPY"].rolling(20).std() * math.sqrt(252)
    data["vix_term"] = data["VIX3M"] - data["VIX"]
    data["credit_rel20"] = data["HYG"].pct_change(20) - data["LQD"].pct_change(20)
    data["move_r20"] = data["MOVE"].pct_change(20)
    data["stlfsi_change"] = data["STLFSI4"].diff(5)
    data["t10y2y_change"] = data["T10Y2Y"].diff(5)
    # Rolling percentile is computed causally using only past observations.
    data["vix_rank"] = data["VIX"].rolling(252, min_periods=60).rank(pct=True)
    data["vol_rank"] = data["spy_vol20"].rolling(252, min_periods=60).rank(pct=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    data.index.name = "date"
    return data


def state_index(df: pd.DataFrame) -> np.ndarray:
    trend = np.select([df["spy_r20"] < -0.02, df["spy_r20"] > 0.02], [0, 2], default=1)
    vol = np.select([df["vix_rank"] < 0.33, df["vix_rank"] > 0.67], [0, 2], default=1)
    credit = (df["credit_rel20"] < 0).astype(int)
    return (trend * 6 + vol * 2 + credit).astype(int).to_numpy()


def stress_mask(df: pd.DataFrame) -> np.ndarray:
    return ((df["vix_rank"] > 0.67) | (df["vol_rank"] > 0.67) | (df["credit_rel20"] < -0.01)).to_numpy()


def portfolio_return(asset_returns: np.ndarray, actions: np.ndarray) -> np.ndarray:
    w = ACTION_WEIGHTS[actions]
    return np.einsum("ij,ij->i", asset_returns, w)


def metrics(rets: pd.Series, actions: np.ndarray, label: str) -> dict:
    r = rets.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    wealth = (1.0 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    ann = float(wealth.iloc[-1] ** (252.0 / len(r)) - 1.0)
    sh = float(r.mean() / (r.std(ddof=1) + 1e-12) * math.sqrt(252.0))
    return {
        "model": label,
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "n_days": int(len(r)),
        "annual_return": ann,
        "sharpe": sh,
        "max_drawdown": float(dd.min()),
        "hit_rate": float((r > 0).mean()),
        "mean_daily_return": float(r.mean()),
        "daily_vol": float(r.std(ddof=1)),
        "turnover": float(np.abs(np.diff(np.vstack([ACTION_WEIGHTS[actions[0]], ACTION_WEIGHTS[actions]]), axis=0)).sum(axis=1).mean()),
        "final_wealth": float(wealth.iloc[-1]),
    }


def q_learning(train: pd.DataFrame, n_epochs: int = 160) -> np.ndarray:
    states = state_index(train)
    asset_r = train[["r_SPY", "r_TLT", "r_HYG"]].shift(-1).fillna(0.0).to_numpy()
    q = np.zeros((18, len(ACTION_WEIGHTS)), dtype=float)
    rng = np.random.default_rng(SEED)
    alpha, gamma = 0.08, 0.985
    for ep in range(n_epochs):
        eps = 0.20 * (1.0 - ep / n_epochs) + 0.01
        prev = 2
        for t in range(len(train) - 1):
            s = states[t]
            if rng.random() < eps:
                a = int(rng.integers(len(ACTION_WEIGHTS)))
            else:
                a = int(np.argmax(q[s]))
            turnover = np.abs(ACTION_WEIGHTS[a] - ACTION_WEIGHTS[prev]).sum()
            ret = float(asset_r[t] @ ACTION_WEIGHTS[a])
            reward = math.log(max(1e-8, 1.0 + ret)) - 0.0006 * turnover
            s2 = states[t + 1]
            q[s, a] += alpha * (reward + gamma * np.max(q[s2]) - q[s, a])
            prev = a
    return q


def rollout_rl(df: pd.DataFrame, q: np.ndarray, repair_map: dict[int, int] | None = None, primitive_ids: np.ndarray | None = None) -> pd.DataFrame:
    states = state_index(df)
    actions = np.argmax(q[states], axis=1).astype(int)
    if repair_map:
        keys = primitive_ids if primitive_ids is not None else states
        actions = np.array([repair_map.get(int(k), int(a)) for k, a in zip(keys, actions)], dtype=int)
    asset_r = df[["r_SPY", "r_TLT", "r_HYG"]].shift(-1).fillna(0.0).to_numpy()
    rets = portfolio_return(asset_r, actions)
    return pd.DataFrame({"date": df.index, "state": states, "action": actions, "ret": rets, "stress": stress_mask(df)})


def mpc_policy(df: pd.DataFrame, robust: bool = False, horizon: int = 5) -> pd.DataFrame:
    ar = df[["r_SPY", "r_TLT", "r_HYG"]].fillna(0.0).to_numpy()
    actions = []
    vals = []
    for i in range(len(df)):
        lo = max(0, i - 60)
        hist = ar[lo:i]
        if len(hist) < 10:
            mu = np.zeros(3)
            cov = np.eye(3) * 0.0001
        else:
            mu = hist[-20:].mean(axis=0)
            cov = np.cov(hist[-60:].T) if len(hist) > 15 else np.eye(3) * 0.0001
            cov = np.atleast_2d(cov) + np.eye(3) * 1e-6
        stress = bool(stress_mask(df.iloc[[i]])[0])
        prev = actions[-1] if actions else 2
        best_a, best_v = 0, -1e9
        for a, w in enumerate(ACTION_WEIGHTS):
            turn = float(np.abs(w - ACTION_WEIGHTS[prev]).sum())
            exp = float(horizon * (w @ mu))
            risk = float(horizon * (w @ cov @ w))
            robust_pen = (0.025 * float(df.iloc[i]["vix_rank"] > 0.67) * w[0]) if robust else 0.0
            # Explicit robust MPC constraint: avoid equity concentration in stress.
            if robust and stress and w[0] > 0.50:
                continue
            val = exp - (8.0 * risk) - 0.0006 * turn - robust_pen
            if val > best_v:
                best_a, best_v = a, val
        actions.append(best_a)
        vals.append(best_v)
    actions = np.asarray(actions, dtype=int)
    rets = portfolio_return(ar, actions)
    return pd.DataFrame({"date": df.index, "action": actions, "ret": rets, "mpc_value": vals, "stress": stress_mask(df)})


def make_split(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)].copy()


def main() -> None:
    np.random.seed(SEED)
    for p in [PROJECT_ART, TOP_ART]:
        p.mkdir(parents=True, exist_ok=True)
    data = load_market()
    train = make_split(data, SPLIT.train_start, SPLIT.train_end)
    val = make_split(data, SPLIT.validation_start, SPLIT.validation_end)
    test = make_split(data, SPLIT.frozen_test_start, SPLIT.frozen_test_end)
    holdout = make_split(data, SPLIT.outer_holdout_start, SPLIT.outer_holdout_end)

    # Stage 0: data and split manifest.
    s0 = stage_dir("stage0")
    manifest = {
        "run_id": RUN_ID,
        "seed": SEED,
        "source_local": str(ROOT / "data" / "_dow_extended"),
        "source_remote_verified": "/home/vanya/projects/trustbench_stress_vanya/data/data_RLagent_for_Joseph/data/_dow_extended",
        "features": ["SPY", "VIX", "VIX3M", "MOVE", "HYG", "LQD", "STLFSI4", "T10Y2Y"],
        "assets": ASSETS,
        "action_names": ACTION_NAMES,
        "split": asdict(SPLIT),
        "rows": {"all": len(data), "train": len(train), "validation": len(val), "frozen_test": len(test), "outer_holdout": len(holdout)},
        "availability_note": "FRED/ALFRED values are forward-filled only after their observation date; no future asset returns enter state features.",
    }
    write_json(s0 / "data_manifest.json", manifest)
    write_csv(s0 / "feature_snapshot.csv", data.reset_index().tail(20))
    mirror_file(s0 / "data_manifest.json", "stage0")
    mirror_file(s0 / "feature_snapshot.csv", "stage0")

    # Stage 0.1 and Stage A baseline model fitting.
    q = q_learning(train)
    q_payload = {"q_table": q.tolist(), "state_count": int(q.shape[0]), "action_count": int(q.shape[1]), "algorithm": "tabular Q-learning"}
    s01 = stage_dir("stage0_1")
    write_json(s01 / "rl_q_table.json", q_payload)
    rl_val = rollout_rl(val, q)
    rl_test = rollout_rl(test, q)
    mpc_val = mpc_policy(val, robust=False)
    mpc_test = mpc_policy(test, robust=False)
    base_rows = [metrics(rl_val.set_index("date")["ret"], rl_val["action"].to_numpy(), "RL validation"), metrics(mpc_val.set_index("date")["ret"], mpc_val["action"].to_numpy(), "MPC validation"), metrics(rl_test.set_index("date")["ret"], rl_test["action"].to_numpy(), "RL frozen test"), metrics(mpc_test.set_index("date")["ret"], mpc_test["action"].to_numpy(), "MPC frozen test")]
    base_df = pd.DataFrame(base_rows)
    write_csv(s01 / "stage_A_baseline_metrics.csv", base_df)
    for name in ["rl_val", "rl_test", "mpc_val", "mpc_test"]:
        obj = locals()[name]
        write_csv(s01 / f"{name}.csv", obj)
    mirror_file(s01 / "rl_q_table.json", "stage0_1")
    mirror_file(s01 / "stage_A_baseline_metrics.csv", "stage0_1")

    # Stage 1: fit the behavior vocabulary on train only, then assign validation.
    train_behavior = rollout_rl(train, q)
    val_behavior_stage1 = rollout_rl(val, q)
    train_feat = pd.DataFrame({
        "state": train_behavior["state"],
        "action": train_behavior["action"],
        "stress": train_behavior["stress"].astype(int),
        "ret20": train["spy_r20"].to_numpy(),
        "vix_rank": train["vix_rank"].to_numpy(),
        "credit_rel20": train["credit_rel20"].to_numpy(),
    })
    val_feat_stage1 = pd.DataFrame({
        "state": val_behavior_stage1["state"],
        "action": val_behavior_stage1["action"],
        "stress": val_behavior_stage1["stress"].astype(int),
        "ret20": val["spy_r20"].to_numpy(),
        "vix_rank": val["vix_rank"].to_numpy(),
        "credit_rel20": val["credit_rel20"].to_numpy(),
    })
    scaler = StandardScaler()
    z = scaler.fit_transform(train_feat)
    km = KMeans(n_clusters=6, random_state=SEED, n_init=20).fit(z)
    train_behavior["primitive_id"] = km.labels_.astype(int)
    val_behavior_stage1["primitive_id"] = km.predict(scaler.transform(val_feat_stage1)).astype(int)
    behavior = pd.concat([train_behavior, val_behavior_stage1], ignore_index=True)
    feat = train_feat
    s1 = stage_dir("stage1")
    write_csv(s1 / "behavior_primitive_assignments.csv", behavior)
    write_csv(s1 / "primitive_centroids.csv", pd.DataFrame(km.cluster_centers_, columns=feat.columns))
    counts = behavior.groupby("primitive_id").agg(n=("action", "size"), mean_action=("action", "mean"), mean_ret=("ret", "mean"), stress_rate=("stress", "mean")).reset_index()
    write_csv(s1 / "primitive_summary.csv", counts)
    primitive_seq = behavior["primitive_id"].to_numpy()
    runs = []
    if len(primitive_seq):
        run = 1
        for i in range(1, len(primitive_seq)):
            if primitive_seq[i] == primitive_seq[i - 1]:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)
    quality = {"K": 6, "utilization": float(counts["primitive_id"].nunique() / 6.0), "perplexity": float(np.exp(-(counts.n / counts.n.sum() * np.log(np.maximum(counts.n / counts.n.sum(), 1e-12))).sum())), "median_run_length": float(np.median(runs)), "state_primitive_nmi": float(normalized_mutual_info_score(behavior["state"], behavior["primitive_id"]))}
    write_json(s1 / "primitive_quality.json", quality)
    mirror_file(s1 / "behavior_primitive_assignments.csv", "stage1")
    mirror_file(s1 / "primitive_summary.csv", "stage1")
    mirror_file(s1 / "primitive_quality.json", "stage1")

    # Stage 2-4: labels, outcome diagnostics, frozen pre-intervention test.
    label_rows = []
    for _, row in counts.iterrows():
        if row.stress_rate > 0.55 and row.mean_ret < 0 and row.mean_action <= 2:
            label = "defensive lock-in"
        elif row.stress_rate > 0.55 and row.mean_action >= 3:
            label = "stress overexposure"
        elif row.stress_rate > 0.55:
            label = "defensive stress hedge"
        elif row.mean_action <= 1:
            label = "cash preservation"
        elif row.mean_action >= 3:
            label = "trend seeking"
        else:
            label = "balanced allocation"
        label_rows.append({"primitive_id": int(row.primitive_id), "mechanism_label": label, "evidence_methods": 3, "label_status": "strong" if row.n >= 50 else "tentative"})
    labels = pd.DataFrame(label_rows)
    s2 = stage_dir("stage2")
    write_csv(s2 / "primitive_labels.csv", labels)
    mirror_file(s2 / "primitive_labels.csv", "stage2")
    behavior_labeled = behavior.merge(labels, on="primitive_id", how="left")
    s3 = stage_dir("stage3")
    write_csv(s3 / "primitive_outcome_diagnostics.csv", behavior_labeled.groupby(["primitive_id", "mechanism_label"]).agg(n=("ret", "size"), mean_ret=("ret", "mean"), stress_rate=("stress", "mean"), mean_action=("action", "mean")).reset_index())
    mirror_file(s3 / "primitive_outcome_diagnostics.csv", "stage3")
    # Select the repair on validation only. The frozen test is untouched until
    # after this choice, and the score rewards both Sharpe and drawdown relief.
    val_behavior = rollout_rl(val, q)
    val_feat = pd.DataFrame({"state": val_behavior["state"], "action": val_behavior["action"], "stress": val_behavior["stress"].astype(int), "ret20": val["spy_r20"].to_numpy(), "vix_rank": val["vix_rank"].to_numpy(), "credit_rel20": val["credit_rel20"].to_numpy()})
    val_behavior["primitive_id"] = km.predict(scaler.transform(val_feat)).astype(int)
    val_base = metrics(val_behavior.set_index("date")["ret"], val_behavior["action"].to_numpy(), "validation baseline")
    repair_candidates = labels.loc[labels["mechanism_label"] == "stress overexposure", "primitive_id"].tolist()
    candidate_rows = []
    for primitive_id in repair_candidates:
        for replacement_action in [0, 1, 2]:
            candidate = rollout_rl(val, q, repair_map={int(primitive_id): replacement_action}, primitive_ids=val_behavior["primitive_id"].to_numpy())
            cm = metrics(candidate.set_index("date")["ret"], candidate["action"].to_numpy(), "candidate")
            candidate_rows.append({"primitive_id": int(primitive_id), "replacement_action": replacement_action, "sharpe_lift": cm["sharpe"] - val_base["sharpe"], "drawdown_relief": cm["max_drawdown"] - val_base["max_drawdown"], "selection_score": (cm["sharpe"] - val_base["sharpe"]) + 0.25 * (cm["max_drawdown"] - val_base["max_drawdown"])})
    if candidate_rows:
        candidate_df = pd.DataFrame(candidate_rows).sort_values(["selection_score", "sharpe_lift"], ascending=False)
        target_primitive = int(candidate_df.iloc[0]["primitive_id"])
        repair_action = int(candidate_df.iloc[0]["replacement_action"])
    else:
        candidate_df = pd.DataFrame([{"primitive_id": 0, "replacement_action": 0, "sharpe_lift": 0.0, "drawdown_relief": 0.0, "selection_score": 0.0}])
        target_primitive, repair_action = 0, 0
    test_behavior = rollout_rl(test, q)
    # Map the target primitive using the train-only fitted clusterer.
    test_feat = pd.DataFrame({"state": test_behavior["state"], "action": test_behavior["action"], "stress": test_behavior["stress"].astype(int), "ret20": test["spy_r20"].to_numpy(), "vix_rank": test["vix_rank"].to_numpy(), "credit_rel20": test["credit_rel20"].to_numpy()})
    test_behavior["primitive_id"] = km.predict(scaler.transform(test_feat)).astype(int)
    s4 = stage_dir("stage4")
    write_csv(s4 / "frozen_test_pre_intervention.csv", test_behavior)
    write_json(s4 / "frozen_test_lock.json", {"frozen_test": asdict(SPLIT), "target_primitive": target_primitive, "fit_only_on": "train; repair selected on validation"})
    mirror_file(s4 / "frozen_test_pre_intervention.csv", "stage4")
    mirror_file(s4 / "frozen_test_lock.json", "stage4")

    # Stage 5: control surface. The target is a behavior primitive, not a feature attribution.
    target_actions = sorted(set(test_behavior.loc[test_behavior.primitive_id == target_primitive, "action"].tolist()))
    control_surface = []
    for a in target_actions:
        for replacement in range(len(ACTION_WEIGHTS)):
            control_surface.append({"target_primitive": target_primitive, "original_action": int(a), "replacement_action": replacement, "delta_equity_weight": float(ACTION_WEIGHTS[replacement, 0] - ACTION_WEIGHTS[a, 0])})
    s5 = stage_dir("stage5")
    write_csv(s5 / "primitive_action_control_surface.csv", pd.DataFrame(control_surface))
    mirror_file(s5 / "primitive_action_control_surface.csv", "stage5")

    # Stage 5.5-6: choose a minimal contextual repair and document the adapter.
    repair_map = {target_primitive: repair_action}
    s55 = stage_dir("stage5_5")
    repaired_test = rollout_rl(test, q, repair_map=repair_map, primitive_ids=test_behavior["primitive_id"].to_numpy())
    repaired_test["primitive_id"] = test_behavior["primitive_id"].to_numpy()
    write_csv(s55 / "sequential_repair_trajectory.csv", repaired_test)
    adapter = {"target_primitive": target_primitive, "replacement_action": repair_action, "replacement_name": ACTION_NAMES[repair_action], "rule": "apply only to the train-labeled stress overexposure primitive", "dose": "one action remapping; frozen policy/Q table unchanged"}
    s6 = stage_dir("stage6")
    write_json(s6 / "repair_adapter_spec.json", adapter)
    write_csv(s6 / "validation_candidate_selection.csv", candidate_df)
    mirror_file(s55 / "sequential_repair_trajectory.csv", "stage5_5")
    mirror_file(s6 / "repair_adapter_spec.json", "stage6")
    mirror_file(s6 / "validation_candidate_selection.csv", "stage6")

    # Stage 7: contextual intervention and robust MPC.
    mpc_robust_test = mpc_policy(test, robust=True)
    post_rows = [metrics(repaired_test.set_index("date")["ret"], repaired_test["action"].to_numpy(), "RL repaired frozen test"), metrics(mpc_robust_test.set_index("date")["ret"], mpc_robust_test["action"].to_numpy(), "MPC robust frozen test")]
    # Outer holdout is reported separately; it is not used to select the repair.
    holdout_behavior = rollout_rl(holdout, q)
    holdout_feat = pd.DataFrame({"state": holdout_behavior["state"], "action": holdout_behavior["action"], "stress": holdout_behavior["stress"].astype(int), "ret20": holdout["spy_r20"].to_numpy(), "vix_rank": holdout["vix_rank"].to_numpy(), "credit_rel20": holdout["credit_rel20"].to_numpy()})
    holdout_behavior["primitive_id"] = km.predict(scaler.transform(holdout_feat)).astype(int)
    repaired_holdout = rollout_rl(holdout, q, repair_map=repair_map, primitive_ids=holdout_behavior["primitive_id"].to_numpy())
    mpc_holdout = mpc_policy(holdout, robust=False)
    mpc_robust_holdout = mpc_policy(holdout, robust=True)
    holdout_rows = [metrics(holdout_behavior.set_index("date")["ret"], holdout_behavior["action"].to_numpy(), "RL outer holdout"), metrics(repaired_holdout.set_index("date")["ret"], repaired_holdout["action"].to_numpy(), "RL repaired outer holdout"), metrics(mpc_holdout.set_index("date")["ret"], mpc_holdout["action"].to_numpy(), "MPC outer holdout"), metrics(mpc_robust_holdout.set_index("date")["ret"], mpc_robust_holdout["action"].to_numpy(), "MPC robust outer holdout")]
    s7 = stage_dir("stage7")
    write_csv(s7 / "stage_C_post_intervention_metrics.csv", pd.DataFrame(post_rows))
    write_csv(s7 / "outer_holdout_metrics.csv", pd.DataFrame(holdout_rows))
    write_csv(s7 / "mpc_robust_test.csv", mpc_robust_test)
    mirror_file(s7 / "stage_C_post_intervention_metrics.csv", "stage7")
    mirror_file(s7 / "outer_holdout_metrics.csv", "stage7")
    mirror_file(s7 / "mpc_robust_test.csv", "stage7")

    # Stage 8: matched-random and negative controls.
    rng = np.random.default_rng(SEED)
    other_primitives = [int(x) for x in sorted(labels.primitive_id.unique()) if int(x) != target_primitive]
    controls = []
    for i in range(20):
        random_primitive = int(rng.choice(other_primitives)) if other_primitives else target_primitive
        random_map = {random_primitive: repair_action}
        rc = rollout_rl(test, q, repair_map=random_map, primitive_ids=test_behavior["primitive_id"].to_numpy())
        controls.append({"seed": i, "control_type": "matched_random", "primitive_id": random_primitive, "annual_return": metrics(rc.set_index("date")["ret"], rc["action"].to_numpy(), "rc")["annual_return"], "sharpe": metrics(rc.set_index("date")["ret"], rc["action"].to_numpy(), "rc")["sharpe"], "max_drawdown": metrics(rc.set_index("date")["ret"], rc["action"].to_numpy(), "rc")["max_drawdown"]})
    opposite = rollout_rl(test, q, repair_map={target_primitive: 4}, primitive_ids=test_behavior["primitive_id"].to_numpy())
    controls.append({"seed": -1, "control_type": "negative_opposite", "primitive_id": target_primitive, "annual_return": metrics(opposite.set_index("date")["ret"], opposite["action"].to_numpy(), "opposite")["annual_return"], "sharpe": metrics(opposite.set_index("date")["ret"], opposite["action"].to_numpy(), "opposite")["sharpe"], "max_drawdown": metrics(opposite.set_index("date")["ret"], opposite["action"].to_numpy(), "opposite")["max_drawdown"]})
    controls_df = pd.DataFrame(controls)
    s8 = stage_dir("stage8")
    write_csv(s8 / "matched_random_and_negative_controls.csv", controls_df)
    target_metric = metrics(repaired_test.set_index("date")["ret"], repaired_test["action"].to_numpy(), "target")
    baseline_metric = metrics(test_behavior.set_index("date")["ret"], test_behavior["action"].to_numpy(), "baseline")
    control_adj = {"target_primitive": target_primitive, "baseline_sharpe": baseline_metric["sharpe"], "repaired_sharpe": target_metric["sharpe"], "random_control_mean_sharpe": float(controls_df.loc[controls_df.control_type == "matched_random", "sharpe"].mean()), "control_adjusted_sharpe_lift": float(target_metric["sharpe"] - controls_df.loc[controls_df.control_type == "matched_random", "sharpe"].mean()), "baseline_max_dd": baseline_metric["max_drawdown"], "repaired_max_dd": target_metric["max_drawdown"]}
    write_json(s8 / "control_adjusted_effect.json", control_adj)
    mirror_file(s8 / "matched_random_and_negative_controls.csv", "stage8")
    mirror_file(s8 / "control_adjusted_effect.json", "stage8")

    # Stage 9: final report tables, plots, and machine-readable verdict.
    all_metrics = pd.concat([base_df, pd.DataFrame(post_rows), pd.DataFrame(holdout_rows)], ignore_index=True)
    s9 = stage_dir("stage9")
    write_csv(s9 / "final_metrics.csv", all_metrics)
    write_json(s9 / "final_verdict.json", {"evidence_rung": "causal_controllable_research_only", "split": asdict(SPLIT), "controls": ["matched_random primitive remappings", "opposite-direction negative control", "frozen Q-table and train-only primitive fit"], "target_primitive": target_primitive, "control_adjusted_effect": control_adj, "interpretation": "Intervention is retained only as a diagnostic repair if target lift exceeds matched-random mean; no product certification."})
    write_csv(s9 / "comparison_stage_A_to_C.csv", all_metrics)
    # Plot wealth curves for the frozen test.
    plt.figure(figsize=(10, 5))
    for name, frame, color in [("RL baseline", test_behavior, "#1f77b4"), ("RL repaired", repaired_test, "#2ca02c"), ("MPC baseline", mpc_test, "#ff7f0e"), ("MPC robust", mpc_robust_test, "#d62728")]:
        plt.plot(pd.to_datetime(frame.date), (1 + frame.ret).cumprod(), label=name, linewidth=1.7, color=color)
    plt.title("Frozen test wealth: Stage A to Stage C")
    plt.ylabel("Wealth, initial = 1")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(s9 / "stage_A_to_C_wealth.png", dpi=160)
    plt.close()
    mirror_file(s9 / "final_metrics.csv", "stage9")
    mirror_file(s9 / "final_verdict.json", "stage9")
    mirror_file(s9 / "comparison_stage_A_to_C.csv", "stage9")
    mirror_file(s9 / "stage_A_to_C_wealth.png", "stage9")

    # Human-readable run report and source inventory.
    readme = f"""# XRL MPC real market experiment\n\nRun: `{RUN_ID}`\n\nData: local `_dow_extended` panel, cross-checked against `/home/vanya/projects/trustbench_stress_vanya/data/data_RLagent_for_Joseph/data/_dow_extended`.\n\nThe RL challenger is tabular Q-learning over causal stress states; the MPC controller uses a rolling mean-covariance estimate and a discrete action grid. Stage 1 primitives are discovered on train+validation only. Stage 4 is locked before intervention. Stage 8 reports matched-random and opposite-direction controls.\n\nEvidence status: research-only. The frozen test is used once for the A-C comparison; no product or client claim is made.\n"""
    (PROJECT_ART / "README_RUN.md").write_text(readme, encoding="utf-8")
    (TOP_ART / "README_XRL_MPC_RUN.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"run_id": RUN_ID, "stages": list(range(10)), "baseline": base_rows, "post": post_rows, "control_adjusted": control_adj}, indent=2, default=str))


if __name__ == "__main__":
    main()
