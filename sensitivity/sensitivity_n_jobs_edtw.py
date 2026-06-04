"""
Event-Driven TW n_jobs sensitivity sweep.

Runs EventDrivenTWPolicy (single-arm, budget=1) across the same n_jobs_sample
values as sensitivity_n_jobs.py, then merges with the existing
sensitivity_n_jobs_summary.csv to produce sensitivity_n_jobs_summary_new.csv.

Outputs
-------
- results/sensitivity_n_jobs_summary_new.csv  (merged with existing CSV)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_armed_bandits_mdp_thompson_whittle_greedy import (
    Config,
    OracleWhittlePolicy,
    build_reward_table,
    build_true_transitions,
    RMABEnvironment,
)
from multi_armed_bandits_improved_tw import EventDrivenTWPolicy


POLICY_NAME = "Event-Driven TW"
EXISTING_CSV = "results/sensitivity_n_jobs_summary.csv"
OUTPUT_CSV   = "results/sensitivity_n_jobs_summary_new.csv"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_n_jobs_edtw(
    raw_dfs: List[pd.DataFrame],
    cfg: Config,
    n_jobs: int,
) -> Tuple[float, float, float]:
    """
    Run n_sims simulations for EventDrivenTWPolicy at the given n_jobs level.
    Returns (avg_reward, avg_regret_vs_oracle, avg_sim_time_s).
    """
    cum_rewards_edtw   = np.zeros(cfg.n_rounds)
    cum_rewards_oracle = np.zeros(cfg.n_rounds)
    sim_times: List[float] = []

    for sim in range(cfg.n_sims):
        sim_seed = cfg.random_state + sim * 13
        R = build_reward_table(
            raw_dfs=raw_dfs,
            random_state=sim_seed,
            n_jobs_sample=n_jobs,
            n_dc=cfg.n_dc,
            batch_size=cfg.batch_size,
            lookahead_size=cfg.lookahead_size,
        )
        p_active_true, p_passive_true = build_true_transitions(
            n_states=n_jobs,
            batch_size=cfg.batch_size,
            n_dc=cfg.n_dc,
            noise=cfg.transition_noise,
        )

        def make_env(seed: int) -> RMABEnvironment:
            return RMABEnvironment(
                reward_table=R,
                p_active=p_active_true,
                p_passive=p_passive_true,
                reward_noise_std=cfg.reward_noise_std,
                rng_seed=seed,
            )

        def run(policy, env):
            states = env.reset()
            rewards = np.zeros(cfg.n_rounds, dtype=float)
            for t in range(cfg.n_rounds):
                arm = policy.select_arm(states, t)
                next_states, reward = env.step(states, arm)
                policy.update(states, arm, reward, next_states)
                rewards[t] = reward
                states = next_states
            return rewards

        oracle = OracleWhittlePolicy(
            reward_table=R, p_active=p_active_true, p_passive=p_passive_true, cfg=cfg
        )
        oracle_rewards = run(oracle, make_env(sim_seed + 101))

        edtw = EventDrivenTWPolicy(
            n_dc=cfg.n_dc, n_states=n_jobs, batch_size=cfg.batch_size,
            cfg=cfg, rng_seed=sim_seed + 451,
        )
        t0 = time.perf_counter()
        edtw_rewards = run(edtw, make_env(sim_seed + 452))
        sim_times.append(time.perf_counter() - t0)

        cum_rewards_edtw   += edtw_rewards
        cum_rewards_oracle += oracle_rewards

    mean_edtw   = cum_rewards_edtw   / cfg.n_sims
    mean_oracle = cum_rewards_oracle / cfg.n_sims
    avg_regret  = float(np.mean(np.cumsum(mean_oracle) - np.cumsum(mean_edtw)))
    return (
        float(np.mean(mean_edtw)),
        avg_regret,
        round(float(np.mean(sim_times)), 4),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="n_jobs sensitivity for Event-Driven TW; merges into existing CSV"
    )
    parser.add_argument(
        "--n-jobs-values", type=str, default="20,40,60,80,100",
        help="Comma-separated n_jobs_sample values (default matches existing CSV)",
    )
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--sims",   type=int, default=1)
    parser.add_argument("--replan-interval", type=int, default=10)
    parser.add_argument("--seed",   type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    n_jobs_values = [int(x) for x in args.n_jobs_values.split(",")]
    if any(v <= 0 for v in n_jobs_values):
        raise ValueError("n_jobs values must be > 0")

    cfg = Config(
        n_rounds=args.rounds,
        n_sims=args.sims,
        random_state=args.seed,
        replan_interval=args.replan_interval,
    )

    os.makedirs("results", exist_ok=True)

    datacenter_files = sorted(
        glob.glob("datacenter_with_metrics/datacenter_*_with_metrics.csv")
    )[: cfg.n_dc]
    if len(datacenter_files) < cfg.n_dc:
        raise RuntimeError(f"Expected {cfg.n_dc} datacenter files, found {len(datacenter_files)}")
    raw_dfs = [pd.read_csv(fp) for fp in datacenter_files]
    print(f"Loaded {cfg.n_dc} datacenters.")

    print("=" * 72)
    print(
        f"Event-Driven TW n_jobs sensitivity | n_jobs_values={n_jobs_values}\n"
        f"rounds={cfg.n_rounds}  sims={cfg.n_sims}  seed={cfg.random_state}  "
        f"replan_interval={cfg.replan_interval}"
    )
    print("=" * 72)

    new_rows = []
    for n_jobs in n_jobs_values:
        print(f"\n--- n_jobs_sample={n_jobs} ---", flush=True)
        avg_r, avg_reg, avg_t = run_n_jobs_edtw(raw_dfs=raw_dfs, cfg=cfg, n_jobs=n_jobs)
        print(
            f"  {POLICY_NAME:<25} avg_reward={avg_r:.4f}  "
            f"avg_regret={avg_reg:.4f}  avg_sim_time={avg_t:.3f}s"
        )
        new_rows.append({
            "n_jobs_sample": n_jobs,
            "policy": POLICY_NAME,
            "avg_reward": avg_r,
            "avg_regret_vs_oracle": avg_reg,
            "avg_sim_time_s": avg_t,
        })

    # Load existing CSV and merge
    if not os.path.exists(EXISTING_CSV):
        raise FileNotFoundError(f"Existing summary CSV not found: {EXISTING_CSV}")
    existing_df = pd.read_csv(EXISTING_CSV)

    # Drop any stale ED-TW rows (in case of re-run)
    existing_df = existing_df[existing_df["policy"] != POLICY_NAME].copy()

    merged_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    merged_df = merged_df.sort_values(["n_jobs_sample", "policy"]).reset_index(drop=True)
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
