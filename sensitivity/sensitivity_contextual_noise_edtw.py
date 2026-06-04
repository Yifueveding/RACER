"""
Event-Driven TW contextual noise sensitivity sweep.

Runs EventDrivenTWPolicy (single-arm) across the same state_noise levels as
sensitivity_contextual_noise.py, then merges with the existing
sensitivity_contextual_noise_summary.csv to produce
sensitivity_contextual_noise_summary_new.csv.

Uses NoisyContextRMABEnvironment from sensitivity_contextual_noise.py so
the noise model is identical.

Outputs
-------
- results/sensitivity_contextual_noise_summary_new.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multi_armed_bandits_mdp_thompson_whittle_greedy import (
    Config,
    OracleWhittlePolicy,
    build_reward_table,
    build_true_transitions,
)
from multi_armed_bandits_improved_tw import EventDrivenTWPolicy
from sensitivity_contextual_noise import NoisyContextRMABEnvironment


POLICY_NAME  = "Event-Driven TW"
EXISTING_CSV = "results/sensitivity_contextual_noise_summary.csv"
OUTPUT_CSV   = "results/sensitivity_contextual_noise_summary_new.csv"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_noise_edtw(
    raw_dfs: List[pd.DataFrame],
    cfg: Config,
    state_noise: float,
) -> tuple[float, float]:
    """
    Run n_sims simulations for EventDrivenTWPolicy at the given noise level.
    Returns (avg_reward, avg_regret_vs_oracle).
    """
    cum_rewards_edtw   = np.zeros(cfg.n_rounds)
    cum_rewards_oracle = np.zeros(cfg.n_rounds)

    for sim in range(cfg.n_sims):
        sim_seed = cfg.random_state + sim * 13
        R = build_reward_table(
            raw_dfs=raw_dfs,
            random_state=sim_seed,
            n_jobs_sample=cfg.n_jobs_sample,
            n_dc=cfg.n_dc,
            batch_size=cfg.batch_size,
            lookahead_size=cfg.lookahead_size,
        )
        p_active_true, p_passive_true = build_true_transitions(
            n_states=cfg.n_jobs_sample,
            batch_size=cfg.batch_size,
            n_dc=cfg.n_dc,
            noise=cfg.transition_noise,
        )

        def make_env(seed: int) -> NoisyContextRMABEnvironment:
            return NoisyContextRMABEnvironment(
                reward_table=R,
                p_active=p_active_true,
                p_passive=p_passive_true,
                reward_noise_std=cfg.reward_noise_std,
                rng_seed=seed,
                state_noise=state_noise,
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
            n_dc=cfg.n_dc, n_states=cfg.n_jobs_sample, batch_size=cfg.batch_size,
            cfg=cfg, rng_seed=sim_seed + 451,
        )
        edtw_rewards = run(edtw, make_env(sim_seed + 452))

        cum_rewards_edtw   += edtw_rewards
        cum_rewards_oracle += oracle_rewards

    mean_edtw   = cum_rewards_edtw   / cfg.n_sims
    mean_oracle = cum_rewards_oracle / cfg.n_sims
    avg_regret  = float(np.mean(np.cumsum(mean_oracle) - np.cumsum(mean_edtw)))
    return float(np.mean(mean_edtw)), avg_regret


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contextual noise sensitivity for Event-Driven TW; merges into existing CSV"
    )
    parser.add_argument(
        "--noise-levels", type=str, default="",
        help="Comma-separated state_noise values; defaults to levels found in existing CSV",
    )
    parser.add_argument("--rounds",          type=int, default=600)
    parser.add_argument("--sims",            type=int, default=4)
    parser.add_argument("--replan-interval", type=int, default=10)
    parser.add_argument("--seed",            type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.exists(EXISTING_CSV):
        raise FileNotFoundError(f"Existing summary CSV not found: {EXISTING_CSV}")
    existing_df = pd.read_csv(EXISTING_CSV)

    if args.noise_levels.strip():
        noise_levels = [float(x) for x in args.noise_levels.split(",")]
    else:
        noise_levels = sorted(existing_df["state_noise"].unique().tolist())
    if any(n < 0 or n > 1 for n in noise_levels):
        raise ValueError("noise levels must be in [0, 1]")

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
        f"Event-Driven TW contextual noise sensitivity | noise_levels={noise_levels}\n"
        f"rounds={cfg.n_rounds}  sims={cfg.n_sims}  seed={cfg.random_state}"
    )
    print("=" * 72)

    new_rows = []
    for noise in noise_levels:
        print(f"\n--- state_noise={noise:.2f} ---", flush=True)
        avg_r, avg_reg = run_noise_edtw(raw_dfs=raw_dfs, cfg=cfg, state_noise=noise)
        print(f"  {POLICY_NAME:<25} avg_reward={avg_r:.4f}  avg_regret={avg_reg:.4f}")
        new_rows.append({
            "state_noise": noise,
            "policy": POLICY_NAME,
            "avg_reward": avg_r,
            "avg_regret_vs_oracle": avg_reg,
        })

    # Drop stale ED-TW rows, append new, sort
    merged_df = existing_df[existing_df["policy"] != POLICY_NAME].copy()
    merged_df = pd.concat([merged_df, pd.DataFrame(new_rows)], ignore_index=True)
    merged_df = merged_df.sort_values(["state_noise", "policy"]).reset_index(drop=True)
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
