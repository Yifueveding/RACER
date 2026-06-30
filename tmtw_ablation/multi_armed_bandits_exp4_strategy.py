"""
EXP4-Strategy: EXP4 bandit where experts are arm-selection strategies.

Experts
-------
  E0 – Thompson-Whittle (TW)   : sample reward/transition posteriors → Whittle indices
  E1 – Global UCB              : empirical arm mean + sqrt(log t / n_pulls) bonus (state-agnostic)
  E2 – Local UCB               : per-(arm, state) Gaussian posterior mean + std bonus

All three experts share the same Gaussian reward posterior and Dirichlet transition
posterior (updated after every observed reward).  EXP4 maintains a weight vector over
the 3 experts and at each round:
  1. Each expert greedily recommends an arm → one-hot ξ_e ∈ {0,1}^K
  2. Mix:  p = (1-γ) · Σ_e w_e ξ_e / Σw  +  γ/K
  3. Sample an arm from p
  4. Observe reward, compute importance-weighted estimate r̂_e, update log-weights (Hedge)

Baselines (standalone policies for comparison)
-----------------------------------------------
  Oracle Whittle     – true transitions and rewards are known
  TW standalone      – Thompson-Whittle with trust=1 (pure Whittle)
  Global UCB         – global empirical UCB only
  Local UCB          – per-(arm,state) Gaussian posterior UCB only

Outputs
-------
  results/plots/exp4_strategy_reward.png
  results/plots/exp4_strategy_regret.png
  results/plots/exp4_strategy_expert_weights.png
  results/plots/exp4_strategy_arm_selection.png
  results/exp4_strategy_summary.csv
  results/exp4_strategy_plot_data.npz
"""

from __future__ import annotations

import glob
import os
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from multi_armed_bandits_ablation_trust_mixed import (
    Config,
    build_reward_table,
    build_true_transitions,
    RMABEnvironment,
    compute_whittle_table,
    GaussianPosterior,
    TransitionPosterior,
    OracleWhittlePolicy,

    run_policy,
    make_base_cfg,
)


# ---------------------------------------------------------------------------
# Standalone baseline policies
# ---------------------------------------------------------------------------

class GlobalUCBPolicy:
    """Global UCB: arm score = empirical mean + UCB bonus (state-agnostic)."""

    def __init__(self, n_dc: int, cfg: Config, rng_seed: int):
        self.n_dc = n_dc
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)
        self.arm_pull_counts = np.zeros(n_dc, dtype=float)
        self.arm_reward_sums = np.zeros(n_dc, dtype=float)
        self.step_n = 0

    def select_arm(self, states: np.ndarray, t: int) -> int:
        if t < self.n_dc:
            return int(t % self.n_dc)
        prior_count = self.cfg.mix_global_prior_count
        pulls = self.arm_pull_counts + prior_count
        means = (self.arm_reward_sums + self.cfg.reward_prior_mean * prior_count) / pulls
        bonus = self.cfg.mix_global_ucb_coef * np.sqrt(np.log(t + 2.0) / pulls)
        return int(np.argmax(means + bonus))

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray):
        self.arm_pull_counts[chosen_arm] += 1.0
        self.arm_reward_sums[chosen_arm] += reward
        self.step_n += 1


class LocalUCBPolicy:
    """Local UCB: per-(arm, state) Gaussian posterior mean + std bonus."""

    def __init__(self, n_dc: int, n_states: int, batch_size: int, cfg: Config, rng_seed: int):
        self.n_dc = n_dc
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)
        self.arm_ids = np.arange(n_dc, dtype=int)
        self.reward_post = GaussianPosterior(
            shape=(n_dc, n_states),
            prior_mean=cfg.reward_prior_mean,
            prior_var=cfg.reward_prior_var,
            obs_var=cfg.reward_obs_var,
        )
        self.step_n = 0

    def select_arm(self, states: np.ndarray, t: int) -> int:
        if t < self.n_dc:
            return int(t % self.n_dc)
        mu = self.reward_post.mu[self.arm_ids, states]
        std = np.sqrt(1.0 / self.reward_post.precision[self.arm_ids, states])
        return int(np.argmax(mu + self.cfg.mix_greedy_bonus_coef * std))

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray):
        self.reward_post.update((chosen_arm, int(states[chosen_arm])), reward)
        self.step_n += 1


# ---------------------------------------------------------------------------
# EXP4-Strategy Policy
# ---------------------------------------------------------------------------

class EXP4StrategyPolicy:
    """
    EXP4 bandit with three arm-selection strategies as experts.

    All experts share the same posteriors (GaussianPosterior for rewards,
    DirichletPosterior for transitions).  At each round EXP4 mixes the
    experts' one-hot arm recommendations into a distribution p over arms,
    samples, observes the reward, and updates expert log-weights via Hedge.

    Expert indices
    --------------
      0 – TW        : argmax Whittle index (replanned every replan_interval rounds)
      1 – GlobalUCB : argmax empirical mean + UCB bonus
      2 – LocalUCB  : argmax Gaussian posterior mean + std bonus (per state)
    """

    EXPERT_NAMES: List[str] = ["TW", "GlobalUCB", "LocalUCB"]

    def __init__(
        self,
        n_dc: int,
        n_states: int,
        batch_size: int,
        cfg: Config,
        rng_seed: int,
    ):
        self.n_dc = n_dc
        self.n_states = n_states
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)
        self.n_experts = len(self.EXPERT_NAMES)
        self.arm_ids = np.arange(n_dc, dtype=int)

        # ---- Shared posteriors ----
        self.reward_post = GaussianPosterior(
            shape=(n_dc, n_states),
            prior_mean=cfg.reward_prior_mean,
            prior_var=cfg.reward_prior_var,
            obs_var=cfg.reward_obs_var,
        )
        self.trans_post = TransitionPosterior(
            n_dc=n_dc, n_states=n_states, batch_size=batch_size,
            prior_alpha=cfg.trans_prior_alpha,
            structural_bias=cfg.trans_structural_bias,
        )

        # ---- EXP4 state ----
        self.log_weights = np.zeros(self.n_experts, dtype=float)
        self.weights = np.ones(self.n_experts, dtype=float)
        # Initialised to uniform; overwritten by select_arm after warmup
        self.last_p = np.ones(n_dc, dtype=float) / n_dc
        self.last_xi = np.zeros((self.n_experts, n_dc), dtype=float)

        # ---- TW expert state ----
        self.cached_W = np.zeros((n_states, n_dc), dtype=float)
        self.last_replan_t: int = -(10 ** 9)

        # ---- Global UCB expert state ----
        self.arm_pull_counts = np.zeros(n_dc, dtype=float)
        self.arm_reward_sums = np.zeros(n_dc, dtype=float)

        # ---- Reward normalisation for importance weighting ----
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        self.reward_scale = 1.0

        # ---- Logging ----
        # One normalised weight vector per round (length = n_rounds after run)
        self.weight_history: List[np.ndarray] = []

        self._actions_buf = np.zeros(n_dc, dtype=int)
        self.step_n = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_replan(self, t: int) -> None:
        if (t - self.last_replan_t) >= self.cfg.replan_interval:
            r_samp = self.reward_post.sample(self.rng)
            pa_samp, pp_samp = self.trans_post.sample(self.rng)
            self.cached_W = compute_whittle_table(r_samp, pa_samp, pp_samp, self.cfg)
            self.last_replan_t = t

    def _update_reward_stats(self, reward: float) -> None:
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2
        if self.reward_count > 1:
            var = self.reward_m2 / (self.reward_count - 1)
            self.reward_scale = max(float(np.sqrt(var)), 1e-3)

    def _reward_to_unit(self, reward: float) -> float:
        """Soft-clip reward to [0, 1] via tanh, for stable importance weighting."""
        scale = max(self.reward_scale, 1e-3)
        z = (reward - self.reward_mean) / scale
        z = float(np.tanh(z / 5.0) * 5.0)
        return 0.5 * (z / 5.0 + 1.0)

    # ------------------------------------------------------------------
    # Expert arm distributions  (one-hot)
    # ------------------------------------------------------------------

    def _tw_xi(self, states: np.ndarray) -> np.ndarray:
        scores = self.cached_W[states, self.arm_ids]
        xi = np.zeros(self.n_dc)
        xi[int(np.argmax(scores))] = 1.0
        return xi

    def _global_ucb_xi(self, t: int) -> np.ndarray:
        prior_count = self.cfg.mix_global_prior_count
        pulls = self.arm_pull_counts + prior_count
        means = (self.arm_reward_sums + self.cfg.reward_prior_mean * prior_count) / pulls
        bonus = self.cfg.mix_global_ucb_coef * np.sqrt(np.log(t + 2.0) / pulls)
        xi = np.zeros(self.n_dc)
        xi[int(np.argmax(means + bonus))] = 1.0
        return xi

    def _local_ucb_xi(self, states: np.ndarray) -> np.ndarray:
        mu = self.reward_post.mu[self.arm_ids, states]
        std = np.sqrt(1.0 / self.reward_post.precision[self.arm_ids, states])
        xi = np.zeros(self.n_dc)
        xi[int(np.argmax(mu + self.cfg.mix_greedy_bonus_coef * std))] = 1.0
        return xi

    def _expert_distributions(self, states: np.ndarray, t: int) -> np.ndarray:
        """Returns xi of shape (n_experts, n_dc) — one-hot per expert."""
        xi = np.zeros((self.n_experts, self.n_dc))
        xi[0] = self._tw_xi(states)
        xi[1] = self._global_ucb_xi(t)
        xi[2] = self._local_ucb_xi(states)
        return xi

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def select_arm(self, states: np.ndarray, t: int) -> int:
        # Warmup: cycle through all arms once before EXP4 kicks in
        if t < self.n_dc:
            return int(t % self.n_dc)

        self._maybe_replan(t)

        xi = self._expert_distributions(states, t)

        # EXP4 mixture: weighted combination of expert distributions + γ-uniform
        w_sum = max(float(np.sum(self.weights)), 1e-12)
        mix = (self.weights[:, None] * xi).sum(axis=0) / w_sum
        p = np.maximum(mix, 1e-12)
        p = p / p.sum()

        arm = int(self.rng.choice(self.n_dc, p=p))
        self.last_p = p
        self.last_xi = xi
        return arm

    def update(self, states: np.ndarray, chosen_arm: int, reward: float, next_states: np.ndarray):
        # ---- Update shared posteriors ----
        self.reward_post.update((chosen_arm, int(states[chosen_arm])), reward)
        self._actions_buf.fill(0)
        self._actions_buf[chosen_arm] = 1
        self.trans_post.alpha[self.arm_ids, states, self._actions_buf, next_states] += 1.0
        self.trans_post.visit_counts[self.arm_ids, states, self._actions_buf] += 1.0
        self.arm_pull_counts[chosen_arm] += 1.0
        self.arm_reward_sums[chosen_arm] += reward

        # ---- Hedge weight update (skip during warmup) ----
        if self.step_n >= self.n_dc:
            self._update_reward_stats(reward)
            r_unit = self._reward_to_unit(reward)
            p_a = max(float(self.last_p[chosen_arm]), 1e-12)
            # Importance-weighted reward estimate per expert
            r_hat = self.last_xi[:, chosen_arm] * r_unit / p_a
            eta_t = float(np.sqrt(np.log(self.n_experts) / (self.n_dc * max(1, self.step_n))))
            self.log_weights += eta_t * r_hat
            self.log_weights -= np.max(self.log_weights)   # numerical stability
            self.weights = np.exp(self.log_weights)

        # ---- Record normalised weights ----
        w_norm = self.weights / max(float(np.sum(self.weights)), 1e-12)
        self.weight_history.append(w_norm.copy())
        self.step_n += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, window: int = 50) -> np.ndarray:
    """Simple moving-average smoothing (valid mode, centred via 'same' padding)."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def main() -> None:
    os.makedirs("results/plots", exist_ok=True)

    datacenter_files = sorted(
        glob.glob("datacenter_with_metrics/datacenter_*_with_metrics.csv")
    )[:5]
    if len(datacenter_files) < 5:
        raise RuntimeError(f"Expected 5 datacenter files, found {len(datacenter_files)}")
    raw_dfs = [pd.read_csv(fp) for fp in datacenter_files]
    print(f"Loaded {len(datacenter_files)} datacenters.")

    base_cfg = Config()
    n_rounds = base_cfg.n_rounds
    n_sims = base_cfg.n_sims
    n_dc = base_cfg.n_dc

    policy_names = ["Oracle Whittle", "EXP4-Strategy"]

    cum_rewards: Dict[str, np.ndarray] = {n: np.zeros(n_rounds) for n in policy_names}
    cum_regrets: Dict[str, np.ndarray] = {
        n: np.zeros(n_rounds) for n in policy_names if n != "Oracle Whittle"
    }
    avg_rewards: Dict[str, list] = {n: [] for n in policy_names}
    sel_counts: Dict[str, np.ndarray] = {n: np.zeros(n_dc, dtype=int) for n in policy_names}
    wall_times: Dict[str, list] = {n: [] for n in policy_names}
    all_weight_histories: List[np.ndarray] = []

    print("=" * 72)
    print(f"EXP4-Strategy experiment | {n_sims} sims | {n_rounds} rounds")
    print("=" * 72)

    for sim in range(n_sims):
        sim_seed = base_cfg.random_state + sim * 13

        R = build_reward_table(
            raw_dfs=raw_dfs,
            random_state=sim_seed,
            n_jobs_sample=base_cfg.n_jobs_sample,
            n_dc=n_dc,
            batch_size=base_cfg.batch_size,
            lookahead_size=base_cfg.lookahead_size,
            lmp=base_cfg.lmp,
        )
        p_active_true, p_passive_true = build_true_transitions(
            n_states=base_cfg.n_jobs_sample,
            batch_size=base_cfg.batch_size,
            n_dc=n_dc,
        )

        policies = {
            "Oracle Whittle": OracleWhittlePolicy(R, p_active_true, p_passive_true, base_cfg),
            "EXP4-Strategy": EXP4StrategyPolicy(
                n_dc=n_dc,
                n_states=base_cfg.n_jobs_sample,
                batch_size=base_cfg.batch_size,
                cfg=base_cfg,
                rng_seed=sim_seed + 300,
            ),
        }
        envs = {
            name: RMABEnvironment(R, p_active_true, p_passive_true, 0.0, sim_seed + 400 + i)
            for i, name in enumerate(policy_names)
        }

        oracle_cum = np.zeros(n_rounds)
        msg_parts = [f"sim {sim + 1}/{n_sims}"]

        for name in policy_names:
            t0 = time.perf_counter()
            rewards, chosen = run_policy(
                envs[name], policies[name], n_rounds, label=f"sim{sim + 1} {name}"
            )
            wall_times[name].append(time.perf_counter() - t0)
            cum_rewards[name] += rewards
            avg_rewards[name].append(float(np.mean(rewards)))
            for dc in range(n_dc):
                sel_counts[name][dc] += int(np.sum(chosen == dc))
            if name == "Oracle Whittle":
                oracle_cum = np.cumsum(rewards)
            else:
                cum_regrets[name] += oracle_cum - np.cumsum(rewards)
            msg_parts.append(f"{name}={np.mean(rewards):.3f}")

        all_weight_histories.append(np.array(policies["EXP4-Strategy"].weight_history))
        print("  " + "  |  ".join(msg_parts))

    mean_rewards = {n: cum_rewards[n] / n_sims for n in policy_names}
    mean_regrets = {n: cum_regrets[n] / n_sims for n in policy_names if n != "Oracle Whittle"}

    # -----------------------------------------------------------------------
    # Save raw plot data
    # -----------------------------------------------------------------------
    save_kwargs: dict = {
        "policy_names": np.array(policy_names),
        "n_rounds": np.array(n_rounds),
        "n_sims": np.array(n_sims),
        "n_dc": np.array(n_dc),
        "expert_names": np.array(EXP4StrategyPolicy.EXPERT_NAMES),
    }
    for name in policy_names:
        key = name.replace(" ", "_")
        save_kwargs[f"mean_rewards__{key}"] = mean_rewards[name]
        save_kwargs[f"avg_rewards__{key}"] = np.array(avg_rewards[name])
        save_kwargs[f"sel_counts__{key}"] = sel_counts[name]
        if name != "Oracle Whittle":
            save_kwargs[f"mean_regrets__{key}"] = mean_regrets[name]
    min_hist_len = min(len(h) for h in all_weight_histories)
    avg_weights = np.mean([h[:min_hist_len] for h in all_weight_histories], axis=0)
    save_kwargs["exp4_avg_weights"] = avg_weights
    np.savez("results/exp4_strategy_plot_data.npz", **save_kwargs)
    print("Saved: results/exp4_strategy_plot_data.npz")

    print("\nAverage reward over simulations:")
    for name in policy_names:
        print(f"  {name:<22} {np.mean(avg_rewards[name]):.4f}")

    rounds_axis = np.arange(1, n_rounds + 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # -----------------------------------------------------------------------
    # Load ablation results and build combined plot entries
    # ablation keys: Oracle_Whittle, TW_only, Local_UCB_plus_TW,
    #                Global_UCB_plus_TW, Local_plus_Global_plus_TW
    # -----------------------------------------------------------------------
    abl = np.load("results/ablation_trust_mixed_plot_data.npz", allow_pickle=True)
    abl_entries = [
        ("Oracle Whittle",       "Oracle_Whittle",             "black",    "--"),
        ("EXP4-Strategy",        None,                          None,       "-"),
        ("(Local+Global+TW)",    "Local_plus_Global_plus_TW",  None,       "-"),
        ("Global UCB + TW",      "Global_UCB_plus_TW",         None,       "-"),
        ("Local UCB + TW",       "Local_UCB_plus_TW",          None,       "-"),
        ("TW only",              "TW_only",                    None,       "-"),
    ]
    color_idx = 0

    # -----------------------------------------------------------------------
    # Plot 1: combined per-round average reward
    # -----------------------------------------------------------------------
    line_series = []
    for label, abl_key, forced_color, ls in abl_entries:
        if forced_color is None:
            color = colors[color_idx % len(colors)]
            color_idx += 1
        else:
            color = forced_color

        if abl_key is None:
            # EXP4 — from this run
            r = mean_rewards["EXP4-Strategy"]
            avg = float(np.mean(avg_rewards["EXP4-Strategy"]))
        else:
            r = abl[f"mean_rewards__{abl_key}"]
            avg = float(np.mean(abl[f"avg_rewards__{abl_key}"]))

        running_avg = np.cumsum(r) / rounds_axis
        line_series.append((label, running_avg, avg, color, ls))

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.2])
    ax = fig.add_subplot(gs[0, :])
    for label, running_avg, avg, color, ls in line_series:
        ax.plot(
            rounds_axis,
            running_avg,
            label=f"{label} (avg={avg:.3f})",
            linewidth=1.5,
            linestyle=ls,
            color=color,
        )
    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T r_t$", fontsize=14)
    ax.set_xlim(0, n_rounds)
    ax.set_ylim(top=25.0)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=11, ncol=2)

    checkpoints = [100, n_rounds]
    bar_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for bar_ax, checkpoint in zip(bar_axes, checkpoints):
        idx = checkpoint - 1
        labels = [entry[0] for entry in line_series]
        values = [entry[1][idx] for entry in line_series]
        bar_colors = [entry[3] for entry in line_series]
        bar_ax.bar(labels, values, color=bar_colors, alpha=0.9)
        bar_ax.set_title(f"Round {checkpoint}", fontsize=13)
        bar_ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T r_t$", fontsize=11)
        bar_ax.set_ylim(0.0, max(values) * 1.12)
        bar_ax.tick_params(axis="x", labelrotation=25, labelsize=9)
        bar_ax.tick_params(axis="y", labelsize=10)
        for tick in bar_ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        for i, value in enumerate(values):
            bar_ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out1 = "results/plots/exp4_vs_ablation_reward.png"
    plt.savefig(out1, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {out1}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 2: combined per-round regret vs Oracle
    # -----------------------------------------------------------------------
    oracle_cum_mean = np.cumsum(mean_rewards["Oracle Whittle"])
    fig, ax = plt.subplots(figsize=(11, 5))
    color_idx = 0
    for label, abl_key, forced_color, ls in abl_entries:
        if label == "Oracle Whittle":
            continue
        if forced_color is None:
            color = colors[color_idx % len(colors)]
            color_idx += 1
        else:
            color = forced_color

        if abl_key is None:
            r = mean_rewards["EXP4-Strategy"]
        else:
            r = abl[f"mean_rewards__{abl_key}"]

        regret = (oracle_cum_mean - np.cumsum(r)) / rounds_axis
        ax.plot(rounds_axis, regret, label=label, linewidth=1.5,
                linestyle=ls, color=color)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T (r_t^* - r_t)$", fontsize=14)
    ax.set_xlim(0, n_rounds)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=11)
    plt.tight_layout()
    out2 = "results/plots/exp4_vs_ablation_regret.png"
    plt.savefig(out2, bbox_inches="tight", dpi=150)
    print(f"Saved: {out2}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 3: EXP4 expert weight evolution (averaged + smoothed)
    # -----------------------------------------------------------------------
    smooth_w = 50
    fig, ax = plt.subplots(figsize=(10, 4))
    weight_rounds = np.arange(1, min_hist_len + 1)
    for e, ename in enumerate(EXP4StrategyPolicy.EXPERT_NAMES):
        smoothed = _smooth(avg_weights[:, e], smooth_w)
        ax.plot(weight_rounds, smoothed, label=ename, linewidth=1.5,
                color=colors[e % len(colors)])
    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel("Normalised expert weight", fontsize=14)
    ax.set_title(f"EXP4 expert weight evolution (avg over {n_sims} sims, smooth={smooth_w})")
    ax.set_xlim(0, min_hist_len)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=12)
    plt.tight_layout()
    out3 = "results/plots/exp4_strategy_expert_weights.png"
    plt.savefig(out3, bbox_inches="tight", dpi=150)
    print(f"Saved: {out3}")
    plt.close()

    # -----------------------------------------------------------------------
    # CSV summary (EXP4 only)
    # -----------------------------------------------------------------------
    rows = []
    for name in policy_names:
        row = {
            "policy": name,
            "avg_reward": float(np.mean(avg_rewards[name])),
            "first100_avg_reward": float(np.mean(mean_rewards[name][:100])),
            "avg_regret_vs_oracle": (
                0.0 if name == "Oracle Whittle" else float(np.mean(mean_regrets[name]))
            ),
            "avg_wall_time_s": float(np.mean(wall_times[name])),
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv("results/exp4_strategy_summary.csv", index=False)
    print("Saved: results/exp4_strategy_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
