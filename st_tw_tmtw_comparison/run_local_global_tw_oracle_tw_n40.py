"""
Compare Local + Global + TW against Oracle Whittle and TW only.

Default setting:
  - n_jobs_sample = 40
  - seeds = 42, 43
  - n_dc = 3, 5, 8, 10
  - budget map = 3:1, 5:2, 8:3, 10:4

For each (seed, n_dc), datacenter files are sampled reproducibly from the VM
dataset. Rewards are saved per round and summarized against Oracle Whittle.

Outputs
-------
event_driven_TW/local_global_tw_oracle_tw_n40_summary.csv
event_driven_TW/local_global_tw_oracle_tw_n40_round_rewards.csv
event_driven_TW/plots/local_global_tw_oracle_tw_n40_running_avg.png
event_driven_TW/plots/local_global_tw_oracle_tw_n40_reward.png
event_driven_TW/plots/local_global_tw_oracle_tw_n40_oracle_ratio.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Event_driven_TW_varying_data_center_jobs import (  # noqa: E402
    BudgetOracleWhittlePolicy,
    BudgetTrustMixedTWPolicy,
    run_policy_budget,
)
from event_driven_TW.run_local_global_tw_varying_selected_dc import (  # noqa: E402
    choose_datacenter_files,
    list_datacenter_files,
    parse_budget_map,
    parse_int_list,
)
from multi_armed_bandits_mdp_thompson_whittle_greedy import (  # noqa: E402
    Config,
    GaussianPosterior,
    RMABEnvironment,
    build_reward_table,
    build_true_transitions,
)


POLICIES = [
    "Oracle Whittle",
    "TW only",
    "State Thompson",
    "Local + Global + TW",
]
DISPLAY_LABELS = {
    "Oracle Whittle": "Oracle",
    "TW only": "TW",
    "State Thompson": "ST",
    "Local + Global + TW": "TM-TW",
}
POLICY_SEED_OFFSETS = {
    "TW only": (201, 202),
    "State Thompson": (301, 302),
    "Local + Global + TW": (401, 402),
}


def make_cfg(n_dc: int, n_jobs: int, rounds: int, seed: int, policy: str) -> Config:
    cfg = Config(
        n_dc=n_dc,
        n_jobs_sample=n_jobs,
        n_rounds=rounds,
        n_sims=1,
        random_state=seed,
    )
    if policy == "TW only":
        cfg.mix_trust_min = 1.0
        cfg.mix_trust_max = 1.0
    return cfg


class BudgetStateThompsonPolicy:
    """State Thompson policy extended to select a top-k active set."""

    def __init__(self, n_dc: int, n_states: int, cfg: Config, rng_seed: int) -> None:
        self.n_dc = n_dc
        self.arm_ids = np.arange(n_dc, dtype=int)
        self.rng = np.random.default_rng(rng_seed)
        self.reward_post = GaussianPosterior(
            shape=(n_dc, n_states),
            prior_mean=cfg.reward_prior_mean,
            prior_var=cfg.reward_prior_var,
            obs_var=cfg.reward_obs_var,
        )

    def select_arms(self, states: np.ndarray, _t: int, budget: int) -> list[int]:
        sampled = self.reward_post.sample(self.rng)
        scores = sampled[self.arm_ids, states]
        return list(np.argsort(scores)[-budget:][::-1])

    def update_multi(
        self,
        states: np.ndarray,
        chosen_arms: list[int],
        per_arm_rewards: np.ndarray,
        _next_states: np.ndarray,
    ) -> None:
        for arm in chosen_arms:
            self.reward_post.update((arm, int(states[arm])), float(per_arm_rewards[arm]))


def run_policy_once(
    policy_name: str,
    reward_table: np.ndarray,
    p_active: np.ndarray,
    p_passive: np.ndarray,
    cfg: Config,
    seed: int,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if policy_name == "Oracle Whittle":
        policy = BudgetOracleWhittlePolicy(reward_table, p_active, p_passive, cfg)
        policy_seed = seed + 101
    elif policy_name == "State Thompson":
        rng_offset, env_offset = POLICY_SEED_OFFSETS[policy_name]
        policy = BudgetStateThompsonPolicy(
            n_dc=cfg.n_dc,
            n_states=cfg.n_jobs_sample,
            cfg=cfg,
            rng_seed=seed + rng_offset,
        )
        policy_seed = seed + env_offset
    else:
        rng_offset, env_offset = POLICY_SEED_OFFSETS[policy_name]
        policy = BudgetTrustMixedTWPolicy(
            n_dc=cfg.n_dc,
            n_states=cfg.n_jobs_sample,
            batch_size=cfg.batch_size,
            cfg=cfg,
            rng_seed=seed + rng_offset,
        )
        policy_seed = seed + env_offset

    env = RMABEnvironment(
        reward_table=reward_table,
        p_active=p_active,
        p_passive=p_passive,
        reward_noise_std=0.0,
        rng_seed=policy_seed,
    )
    t0 = time.perf_counter()
    rewards, chosen = run_policy_budget(env, policy, cfg.n_rounds, budget)
    return rewards, chosen, time.perf_counter() - t0


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_dc_values = parse_int_list(args.n_dc_values)
    seeds = parse_int_list(args.seeds)
    budget_by_n_dc = parse_budget_map(args.budget_map)
    all_files = list_datacenter_files()

    summary_rows: list[dict] = []
    round_rows: list[dict] = []

    print(
        f"Comparing {', '.join(POLICIES)} | n_jobs={args.n_jobs} | "
        f"rounds={args.rounds} | seeds={seeds}"
    )
    for n_dc in n_dc_values:
        if n_dc not in budget_by_n_dc:
            raise ValueError(f"No budget configured for n_dc={n_dc}")
        budget = budget_by_n_dc[n_dc]

        for seed in seeds:
            raw_dfs, dc_ids, _ = choose_datacenter_files(
                files=all_files,
                n_dc=n_dc,
                seed=seed + n_dc * 1009,
            )
            reward_table = build_reward_table(
                raw_dfs=raw_dfs,
                random_state=seed,
                n_jobs_sample=args.n_jobs,
                n_dc=n_dc,
                batch_size=Config().batch_size,
                lookahead_size=Config().lookahead_size,
            )
            p_active, p_passive = build_true_transitions(
                n_states=args.n_jobs,
                batch_size=Config().batch_size,
                n_dc=n_dc,
            )

            run_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
            for policy_name in POLICIES:
                cfg = make_cfg(n_dc=n_dc, n_jobs=args.n_jobs, rounds=args.rounds, seed=seed, policy=policy_name)
                rewards, chosen, wall_time = run_policy_once(
                    policy_name=policy_name,
                    reward_table=reward_table,
                    p_active=p_active,
                    p_passive=p_passive,
                    cfg=cfg,
                    seed=seed,
                    budget=budget,
                )
                run_cache[policy_name] = (rewards, chosen, wall_time)

            oracle_rewards = run_cache["Oracle Whittle"][0]
            oracle_avg = float(np.mean(oracle_rewards))

            for policy_name in POLICIES:
                rewards, chosen, wall_time = run_cache[policy_name]
                avg_reward = float(np.mean(rewards))
                summary_rows.append({
                    "seed": seed,
                    "n_dc": n_dc,
                    "n_jobs": args.n_jobs,
                    "budget": budget,
                    "datacenter_ids": "|".join(map(str, dc_ids)),
                    "policy": policy_name,
                    "avg_reward": avg_reward,
                    "cum_reward": float(np.sum(rewards)),
                    "oracle_avg_reward": oracle_avg,
                    "cum_reward_over_oracle_pct": 100.0 * avg_reward / max(oracle_avg, 1e-12),
                    "avg_wall_time_s": wall_time,
                })

                running_avg = np.cumsum(rewards) / np.arange(1, args.rounds + 1)
                for t, (reward, run_avg) in enumerate(zip(rewards, running_avg), start=1):
                    row = {
                        "seed": seed,
                        "n_dc": n_dc,
                        "n_jobs": args.n_jobs,
                        "budget": budget,
                        "policy": policy_name,
                        "round": t,
                        "reward": float(reward),
                        "running_avg_reward": float(run_avg),
                    }
                    for slot in range(budget):
                        local_idx = int(chosen[t - 1, slot])
                        row[f"selected_datacenter_id_{slot + 1}"] = int(dc_ids[local_idx])
                    round_rows.append(row)

            print(
                f"seed={seed} n_dc={n_dc:>2} budget={budget} "
                f"oracle={oracle_avg:.3f} "
                f"tw={np.mean(run_cache['TW only'][0]):.3f} "
                f"state={np.mean(run_cache['State Thompson'][0]):.3f} "
                f"local_global={np.mean(run_cache['Local + Global + TW'][0]):.3f}",
                flush=True,
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(round_rows)


def plot_results(summary_df: pd.DataFrame, output_dir: str) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    agg = (
        summary_df
        .groupby(["n_dc", "policy"], as_index=False)
        .agg(
            avg_reward=("avg_reward", "mean"),
            sem_reward=("avg_reward", "sem"),
            oracle_ratio=("cum_reward_over_oracle_pct", "mean"),
            sem_ratio=("cum_reward_over_oracle_pct", "sem"),
        )
    )
    oracle_mean_by_n_dc = (
        agg[agg["policy"] == "Oracle Whittle"]
        .set_index("n_dc")["avg_reward"]
    )
    agg["oracle_ratio"] = agg.apply(
        lambda row: 100.0 * row["avg_reward"] / max(float(oracle_mean_by_n_dc.loc[row["n_dc"]]), 1e-12),
        axis=1,
    )

    n_dc_values = sorted(summary_df["n_dc"].unique())
    x = np.arange(len(n_dc_values))
    width = 0.24
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    reward_policies = [policy for policy in POLICIES if policy != "Oracle Whittle"]
    x_reward = np.arange(len(n_dc_values))
    reward_width = min(0.8 / max(len(reward_policies), 1), 0.22)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, policy in enumerate(reward_policies):
        sub = agg[agg["policy"] == policy].set_index("n_dc").loc[n_dc_values]
        ax.bar(
            x_reward + (i - (len(reward_policies) - 1) / 2.0) * reward_width,
            sub["avg_reward"],
            width=reward_width,
            yerr=sub["sem_reward"].fillna(0.0),
            capsize=3,
            label=DISPLAY_LABELS[policy],
            color=colors[(i + 1) % len(colors)],
        )
    ax.set_xticks(x_reward)
    ax.set_xticklabels([str(v) for v in n_dc_values])
    ax.set_xlabel("Number of datacenters", fontsize=12)
    ax.set_ylabel("Average reward", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = os.path.join(plot_dir, "local_global_tw_oracle_tw_n40_reward.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    ratio_policies = [policy for policy in POLICIES if policy != "Oracle Whittle"]
    x_ratio = np.arange(len(n_dc_values))
    ratio_width = min(0.8 / max(len(ratio_policies), 1), 0.22)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, policy in enumerate(ratio_policies):
        sub = agg[agg["policy"] == policy].set_index("n_dc").loc[n_dc_values]
        ax.bar(
            x_ratio + (i - (len(ratio_policies) - 1) / 2.0) * ratio_width,
            sub["oracle_ratio"],
            width=ratio_width,
            yerr=sub["sem_ratio"].fillna(0.0),
            capsize=3,
            label=DISPLAY_LABELS[policy],
            color=colors[(i + 1) % len(colors)],
        )
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x_ratio)
    ax.set_xticklabels([str(v) for v in n_dc_values])
    ax.set_xlabel("Number of datacenters", fontsize=12)
    ax.set_ylabel("Cumulative reward / Oracle (%)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    out = os.path.join(plot_dir, "local_global_tw_oracle_tw_n40_oracle_ratio.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, policy in enumerate(reward_policies):
        sub = agg[agg["policy"] == policy].set_index("n_dc").loc[n_dc_values]
        x_pos = x_reward + (i - (len(reward_policies) - 1) / 2.0) * reward_width
        axes[0].bar(
            x_pos,
            sub["avg_reward"],
            width=reward_width,
            yerr=sub["sem_reward"].fillna(0.0),
            capsize=3,
            label=DISPLAY_LABELS[policy],
            color=colors[(i + 1) % len(colors)],
        )
        axes[1].bar(
            x_pos,
            sub["oracle_ratio"],
            width=reward_width,
            yerr=sub["sem_ratio"].fillna(0.0),
            capsize=3,
            label=DISPLAY_LABELS[policy],
            color=colors[(i + 1) % len(colors)],
        )

    axes[0].set_xticks(x_reward)
    axes[0].set_xticklabels([str(v) for v in n_dc_values])
    axes[0].set_xlabel("Number of datacenters", fontsize=15)
    axes[0].set_ylabel("Average reward", fontsize=15)
    axes[0].set_title("(a) Average reward", fontsize=16)
    axes[0].tick_params(axis="both", labelsize=13)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].axhline(100.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x_reward)
    axes[1].set_xticklabels([str(v) for v in n_dc_values])
    axes[1].set_xlabel("Number of datacenters", fontsize=15)
    axes[1].set_ylabel("Cumulative reward / Oracle (%)", fontsize=15)
    axes[1].set_title("(b) Relative to Oracle", fontsize=16)
    axes[1].tick_params(axis="both", labelsize=13)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    out = os.path.join(plot_dir, "local_global_tw_oracle_tw_n40_combined.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_running_average_rewards(round_df: pd.DataFrame, output_dir: str) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    mean_rewards = (
        round_df
        .groupby(["n_dc", "budget", "policy", "round"], as_index=False)
        .agg(reward=("reward", "mean"))
        .sort_values(["n_dc", "policy", "round"])
    )
    mean_rewards["running_avg_reward"] = (
        mean_rewards
        .groupby(["n_dc", "policy"])["reward"]
        .transform(lambda s: s.cumsum() / np.arange(1, len(s) + 1))
    )

    n_dc_values = sorted(mean_rewards["n_dc"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_by_policy = {
        "Oracle Whittle": "black",
        "TW only": colors[0],
        "State Thompson": colors[1],
        "Local + Global + TW": colors[2],
    }
    linestyle_by_policy = {
        "Oracle Whittle": "--",
        "TW only": "-",
        "State Thompson": "-",
        "Local + Global + TW": "-",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.0), sharex=True)
    axes = axes.ravel()
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    for panel_idx, (ax, n_dc) in enumerate(zip(axes, n_dc_values)):
        dc_df = mean_rewards[mean_rewards["n_dc"] == n_dc]
        budget = int(dc_df["budget"].iloc[0])
        for policy in POLICIES:
            sub = dc_df[dc_df["policy"] == policy]
            if sub.empty:
                continue
            ax.plot(
                sub["round"],
                sub["running_avg_reward"],
                label=DISPLAY_LABELS[policy],
                linewidth=2.1,
                color=color_by_policy.get(policy),
                linestyle=linestyle_by_policy.get(policy, "-"),
            )
        ax.set_title(
            f"{subplot_labels[panel_idx]} $n_{{\\mathrm{{dc}}}}={int(n_dc)}$, budget={budget}",
            fontsize=19,
        )
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if int(n_dc) == 3:
            ax.set_ylim(top=20.0)

    for ax in axes[len(n_dc_values):]:
        ax.axis("off")

    for ax in axes[::2]:
        ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T r_t$", fontsize=18)
    for ax in axes[-2:]:
        ax.set_xlabel("Round", fontsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=18)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    out = os.path.join(plot_dir, "local_global_tw_oracle_tw_n40_running_avg.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def write_comparison_table(summary_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    agg = (
        summary_df
        .groupby(["n_dc", "budget", "policy"], as_index=False)
        .agg(
            avg_reward=("avg_reward", "mean"),
            std_reward=("avg_reward", "std"),
            oracle_ratio=("cum_reward_over_oracle_pct", "mean"),
            std_ratio=("cum_reward_over_oracle_pct", "std"),
        )
    )

    rows = []
    non_oracle_policies = [policy for policy in POLICIES if policy != "Oracle Whittle"]
    for (n_dc, budget), group in agg.groupby(["n_dc", "budget"], sort=True):
        by_policy = group.set_index("policy")
        oracle = by_policy.loc["Oracle Whittle"]
        tw = by_policy.loc["TW only"]
        oracle_avg = float(oracle["avg_reward"])
        tw_avg = float(tw["avg_reward"])
        for policy in non_oracle_policies:
            policy_avg = float(by_policy.loc[policy, "avg_reward"])
            rows.append({
                "n_dc": int(n_dc),
                "budget": int(budget),
                "policy": DISPLAY_LABELS[policy],
                "oracle_avg_reward": oracle_avg,
                "avg_reward": policy_avg,
                "oracle_pct": 100.0 * policy_avg / max(oracle_avg, 1e-12),
                "delta_vs_tw_oracle_pct": 100.0 * (policy_avg - tw_avg) / max(oracle_avg, 1e-12),
            })

    table_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "local_global_tw_oracle_tw_n40_table.csv")
    tex_path = os.path.join(output_dir, "local_global_tw_oracle_tw_n40_table.tex")
    table_df.to_csv(csv_path, index=False)

    latex_rows = []
    for row in table_df.itertuples(index=False):
        latex_rows.append(
            f"{row.n_dc} & {row.budget} & "
            f"{row.policy} & "
            f"{row.oracle_avg_reward:.3f} & "
            f"{row.avg_reward:.3f} & {row.oracle_pct:.2f} & "
            f"{row.delta_vs_tw_oracle_pct:+.2f} \\\\"
        )

    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Reward comparison for Thompson-Whittle variants with 40 sampled VM jobs per datacenter.}",
        r"\label{tab:local_global_tw_n40}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{cc l cccc}",
        r"\toprule",
        r"$n_{\mathrm{dc}}$ & Budget & Policy & Oracle & Avg. reward & Reward / Oracle (\%) & $\Delta$ vs TW / Oracle (\%) \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    return table_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Local+Global+TW, TW only, and Oracle at n_jobs=40.")
    parser.add_argument("--n-dc-values", default="3,5,8,10")
    parser.add_argument("--n-jobs", type=int, default=40)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--budget-map", default="3:1,5:2,8:3,10:4")
    parser.add_argument("--output-dir", default="event_driven_TW")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    summary_df, round_df = run_experiment(args)

    summary_path = os.path.join(args.output_dir, "local_global_tw_oracle_tw_n40_summary.csv")
    round_path = os.path.join(args.output_dir, "local_global_tw_oracle_tw_n40_round_rewards.csv")
    summary_df.to_csv(summary_path, index=False)
    round_df.to_csv(round_path, index=False)
    print(f"Saved: {summary_path}")
    print(f"Saved: {round_path}")

    plot_results(summary_df, args.output_dir)
    plot_running_average_rewards(round_df, args.output_dir)
    write_comparison_table(summary_df, args.output_dir)


if __name__ == "__main__":
    main()
