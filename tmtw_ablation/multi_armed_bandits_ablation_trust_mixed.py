"""
Ablation study: TrustMixedThompsonWhittlePolicy component decomposition.

Three scenarios (600 rounds each):
  (1) Local UCB + Thompson-Whittle  — global greedy weight = 0
  (2) Global UCB + Thompson-Whittle — local greedy weight = 0
  (3) All three: Local + Global + Thompson-Whittle (default blend)

Oracle Whittle is included as an upper-bound reference.

Outputs
-------
- results/plots/ablation_trust_mixed_reward.png
- results/plots/ablation_trust_mixed_regret.png
- results/ablation_trust_mixed_summary.csv
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    n_dc: int = 5
    n_jobs_sample: int = 40
    batch_size: int = 5
    lookahead_size: int = 10
    gamma: float = 0.9
    vi_theta: float = 1e-4
    vi_max_iters: int = 120
    binary_iters: int = 10
    whittle_tol: float = 5e-3
    n_rounds: int = 1000
    n_sims: int = 4
    replan_interval: int = 10
    transition_noise: float = 0.0
    reward_noise_std: float = 0.0
    reward_prior_mean: float = 0.0
    reward_prior_var: float = 25.0
    reward_obs_var: float = 4.0
    trans_prior_alpha: float = 0.1
    trans_structural_bias: float = 0.9
    # Trust-Mixed hyper-params (shared across scenarios unless overridden)
    mix_warmup_rounds: int = 5
    mix_trust_warmup_rounds: int = 10
    mix_trust_ramp_rounds: int = 450
    mix_trust_min: float = 0.01
    mix_trust_max: float = 0.98
    mix_trust_conf_weight: float = 0.45
    mix_trust_progress_weight: float = 0.25
    mix_trust_stability_weight: float = 0.30
    mix_whittle_change_scale: float = 0.20
    mix_trust_switch_round: int = 120
    mix_trust_switch_temp: float = 18.0
    mix_reward_conf_scale: float = 6.0
    mix_trans_conf_scale: float = 8.0
    mix_global_conf_scale: float = 10.0
    mix_global_ucb_coef: float = 4.0
    mix_global_prior_count: float = 1.0
    # Scenario-specific: greedy component blend
    mix_greedy_global_weight: float = 0.95
    mix_greedy_global_decay_rounds: int = 180
    mix_eps_start: float = 0.00
    mix_eps_end: float = 0.00
    mix_eps_decay_rounds: int = 1
    mix_greedy_bonus_coef: float = 0.8
    random_state: int = 42


# ---------------------------------------------------------------------------
# Reward / transition helpers
# ---------------------------------------------------------------------------

def build_reward_table(
    raw_dfs: List[pd.DataFrame],
    random_state: int,
    n_jobs_sample: int,
    n_dc: int,
    batch_size: int,
    lookahead_size: int,
) -> np.ndarray:
    powers, corehrs, ch_norms, is_ints = [], [], [], []
    for dc, df in enumerate(raw_dfs[:n_dc]):
        df_s = df.sample(
            n=min(n_jobs_sample, len(df)),
            random_state=random_state + dc + 1,
        ).reset_index(drop=True)
        corehour = df_s["corehour"].values
        ch_norm = (corehour - corehour.min()) / max(corehour.max() - corehour.min(), 1e-8)
        power = df_s["avgcpu"].values * ch_norm
        is_int = df_s["vmcategory"].values == "Interactive"
        powers.append(power); corehrs.append(corehour)
        ch_norms.append(ch_norm); is_ints.append(is_int)

    def idxs(s: int, size: int) -> List[int]:
        return [(s + i) % n_jobs_sample for i in range(size)]

    def reward(s: int, dc: int) -> float:
        default_batch = idxs(s, batch_size)
        pool = idxs(s, lookahead_size)
        sorted_pool = sorted(pool, key=lambda i: corehrs[dc][i])
        selected = sorted_pool[:batch_size]
        delayed = sorted_pool[batch_size:]
        saving = sum(powers[dc][i] for i in default_batch) - sum(powers[dc][i] for i in selected)
        penalty = sum(10.0 * ch_norms[dc][i] for i in delayed if is_ints[dc][i])
        return float(saving - penalty)

    return np.array([[reward(s, dc) for dc in range(n_dc)] for s in range(n_jobs_sample)], dtype=float)


def build_true_transitions(
    n_states: int,
    batch_size: int,
    n_dc: int,
    noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    p_active = np.zeros((n_dc, n_states, n_states), dtype=float)
    p_passive = np.zeros((n_dc, n_states, n_states), dtype=float)
    for dc in range(n_dc):
        for s in range(n_states):
            s_next = (s + batch_size) % n_states
            p_active[dc, s, s_next] = 1.0
            p_passive[dc, s, s] = 1.0
    if noise > 0.0:
        uniform = np.ones((n_states,), dtype=float) / n_states
        p_active = (1.0 - noise) * p_active + noise * uniform[np.newaxis, np.newaxis, :]
        p_passive = (1.0 - noise) * p_passive + noise * uniform[np.newaxis, np.newaxis, :]
    return p_active, p_passive


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RMABEnvironment:
    def __init__(self, reward_table, p_active, p_passive, reward_noise_std, rng_seed):
        self.R = reward_table
        self.Pa = p_active
        self.Pp = p_passive
        self.n_states, self.n_dc = self.R.shape
        self.reward_noise_std = reward_noise_std
        self.rng = np.random.default_rng(rng_seed)

    def reset(self) -> np.ndarray:
        return np.zeros((self.n_dc,), dtype=int)

    def step(self, states: np.ndarray, chosen_arm: int) -> Tuple[np.ndarray, float]:
        next_states = np.zeros_like(states)
        reward = 0.0
        for dc in range(self.n_dc):
            s = int(states[dc])
            is_active = int(dc == chosen_arm)
            p = self.Pa[dc, s] if is_active else self.Pp[dc, s]
            next_states[dc] = int(self.rng.choice(self.n_states, p=p))
            if is_active:
                mean_r = self.R[s, dc]
                noise = self.rng.normal(0.0, self.reward_noise_std) if self.reward_noise_std > 0 else 0.0
                reward = float(mean_r + noise)
        return next_states, reward


# ---------------------------------------------------------------------------
# Whittle index solver
# ---------------------------------------------------------------------------

def solve_subsidy_mdp(r_active, p_active, p_passive, lam, gamma, theta, max_iters, v_init=None):
    n_states = r_active.shape[0]
    V = np.zeros((n_states,), dtype=float) if v_init is None else v_init.copy()
    for _ in range(max_iters):
        q_active = r_active + gamma * (p_active @ V)
        q_passive = lam + gamma * (p_passive @ V)
        new_V = np.maximum(q_active, q_passive)
        if np.max(np.abs(new_V - V)) < theta:
            return new_V
        V = new_V
    return V


def compute_whittle_index_for_state(s, r_active, p_active, p_passive, gamma, vi_theta,
                                     vi_max_iters, binary_iters, whittle_tol, lam_lo, lam_hi):
    lo, hi = lam_lo, lam_hi
    V = np.zeros_like(r_active)
    for _ in range(binary_iters):
        if hi - lo < whittle_tol:
            break
        mid = 0.5 * (lo + hi)
        V = solve_subsidy_mdp(r_active, p_active, p_passive, mid, gamma, vi_theta, vi_max_iters, V)
        q_active = r_active[s] + gamma * float(np.dot(p_active[s], V))
        q_passive = mid + gamma * float(np.dot(p_passive[s], V))
        if q_active >= q_passive:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_whittle_table(r_active_dc_state, p_active, p_passive, cfg):
    n_dc, n_states = r_active_dc_state.shape
    W = np.zeros((n_states, n_dc), dtype=float)
    for dc in range(n_dc):
        r_dc = r_active_dc_state[dc]
        lam_lo = float(r_dc.min() - abs(r_dc.min()) - 1.0)
        lam_hi = float(r_dc.max() / max(1.0 - cfg.gamma, 1e-6) + 1.0)
        for s in range(n_states):
            W[s, dc] = compute_whittle_index_for_state(
                s=s, r_active=r_dc, p_active=p_active[dc], p_passive=p_passive[dc],
                gamma=cfg.gamma, vi_theta=cfg.vi_theta, vi_max_iters=cfg.vi_max_iters,
                binary_iters=cfg.binary_iters, whittle_tol=cfg.whittle_tol,
                lam_lo=lam_lo, lam_hi=lam_hi,
            )
    return W


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------

class GaussianPosterior:
    def __init__(self, shape, prior_mean, prior_var, obs_var):
        self.mu = np.full(shape, prior_mean, dtype=float)
        self.precision = np.full(shape, 1.0 / prior_var, dtype=float)
        self.obs_precision = 1.0 / obs_var

    def sample(self, rng):
        return rng.normal(loc=self.mu, scale=np.sqrt(1.0 / self.precision))

    def update(self, index, reward):
        old_prec = self.precision[index]
        old_mu = self.mu[index]
        new_prec = old_prec + self.obs_precision
        self.precision[index] = new_prec
        self.mu[index] = (old_prec * old_mu + self.obs_precision * reward) / new_prec


class TransitionPosterior:
    def __init__(self, n_dc, n_states, batch_size, prior_alpha, structural_bias):
        self.alpha = np.full((n_dc, n_states, 2, n_states), prior_alpha, dtype=float)
        self.visit_counts = np.zeros((n_dc, n_states, 2), dtype=float)
        for dc in range(n_dc):
            for s in range(n_states):
                s_active_next = (s + batch_size) % n_states
                self.alpha[dc, s, 1, s_active_next] += structural_bias
                self.alpha[dc, s, 0, s] += structural_bias

    def sample(self, rng):
        n_dc, n_states = self.alpha.shape[:2]
        p_active = np.zeros((n_dc, n_states, n_states), dtype=float)
        p_passive = np.zeros((n_dc, n_states, n_states), dtype=float)
        for dc in range(n_dc):
            for s in range(n_states):
                p_passive[dc, s] = rng.dirichlet(self.alpha[dc, s, 0])
                p_active[dc, s] = rng.dirichlet(self.alpha[dc, s, 1])
        return p_active, p_passive

    def update(self, dc, state, action, next_state):
        self.alpha[dc, state, action, next_state] += 1.0
        self.visit_counts[dc, state, action] += 1.0


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def normalize_scores_01(scores):
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if s_max - s_min < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - s_min) / (s_max - s_min)


class OracleWhittlePolicy:
    def __init__(self, reward_table, p_active, p_passive, cfg):
        self.W = compute_whittle_table(reward_table.T, p_active, p_passive, cfg)
        self.n_dc = reward_table.shape[1]

    def select_arm(self, states, t):
        return int(np.argmax([self.W[states[dc], dc] for dc in range(self.n_dc)]))

    def update(self, states, chosen_arm, reward, next_states):
        pass


class ThompsonWhittleBase:
    """Shared Thompson-Whittle logic used by all TrustMixed scenarios."""

    def __init__(self, n_dc, n_states, batch_size, cfg, rng_seed):
        self.cfg = cfg
        self.n_dc = n_dc
        self.n_states = n_states
        self.arm_ids = np.arange(n_dc, dtype=int)
        self.rng = np.random.default_rng(rng_seed)
        self.reward_post = GaussianPosterior(
            shape=(n_dc, n_states),
            prior_mean=cfg.reward_prior_mean,
            prior_var=cfg.reward_prior_var,
            obs_var=cfg.reward_obs_var,
        )
        self.trans_post = TransitionPosterior(
            n_dc=n_dc, n_states=n_states, batch_size=batch_size,
            prior_alpha=cfg.trans_prior_alpha, structural_bias=cfg.trans_structural_bias,
        )
        self._actions_buf = np.zeros((n_dc,), dtype=int)
        self.cached_W = np.zeros((n_states, n_dc), dtype=float)
        self.whittle_change = 1.0
        self.last_replan_t = -(10 ** 9)
        self._prior_precision = 1.0 / max(cfg.reward_prior_var, 1e-8)
        self._obs_precision = 1.0 / max(cfg.reward_obs_var, 1e-8)
        self.arm_pull_counts = np.zeros((n_dc,), dtype=float)
        self.arm_reward_sums = np.zeros((n_dc,), dtype=float)
        w_conf = max(cfg.mix_trust_conf_weight, 0.0)
        w_prog = max(cfg.mix_trust_progress_weight, 0.0)
        w_stab = max(cfg.mix_trust_stability_weight, 0.0)
        w_sum = max(w_conf + w_prog + w_stab, 1e-8)
        self._w_conf = w_conf / w_sum
        self._w_prog = w_prog / w_sum
        self._w_stab = w_stab / w_sum

    @staticmethod
    def _linear_decay(t, start, end, horizon):
        decay_frac = max(0.0, 1.0 - (t / max(horizon, 1)))
        return end + (start - end) * decay_frac

    def _maybe_replan(self, t):
        if (t - self.last_replan_t) >= self.cfg.replan_interval:
            sampled_rewards = self.reward_post.sample(self.rng)
            sampled_pa, sampled_pp = self.trans_post.sample(self.rng)
            new_W = compute_whittle_table(sampled_rewards, sampled_pa, sampled_pp, self.cfg)
            if self.last_replan_t > -(10 ** 8):
                denom = max(float(np.mean(np.abs(self.cached_W))), 1e-8)
                self.whittle_change = float(np.mean(np.abs(new_W - self.cached_W)) / denom)
            else:
                self.whittle_change = 1.0
            self.cached_W = new_W
            self.last_replan_t = t

    def _whittle_scores(self, states):
        return self.cached_W[states, self.arm_ids]

    def _global_ucb_score(self, t):
        pulls = self.arm_pull_counts + self.cfg.mix_global_prior_count
        prior_sum = self.cfg.reward_prior_mean * self.cfg.mix_global_prior_count
        means = (self.arm_reward_sums + prior_sum) / pulls
        bonus = self.cfg.mix_global_ucb_coef * np.sqrt(np.log(t + 2.0) / pulls)
        return means + bonus

    def _local_ucb_score(self, states):
        mu = self.reward_post.mu[self.arm_ids, states]
        std = np.sqrt(1.0 / self.reward_post.precision[self.arm_ids, states])
        return mu + self.cfg.mix_greedy_bonus_coef * std

    def _greedy_score(self, states, t):
        local_ucb = self._local_ucb_score(states)
        global_ucb = self._global_ucb_score(t)
        w_global = self.cfg.mix_greedy_global_weight * max(
            0.0, 1.0 - (t / max(self.cfg.mix_greedy_global_decay_rounds, 1)),
        )
        w_global = float(np.clip(w_global, 0.0, 1.0))
        return w_global * global_ucb + (1.0 - w_global) * local_ucb

    def _compute_model_confidence(self, states):
        prec = self.reward_post.precision[self.arm_ids, states]
        n_eff = np.maximum((prec - self._prior_precision) / self._obs_precision, 0.0)
        reward_conf = 1.0 - np.exp(-n_eff / max(self.cfg.mix_reward_conf_scale, 1e-8))
        active_visits = self.trans_post.visit_counts[self.arm_ids, states, 1]
        trans_conf = 1.0 - np.exp(-active_visits / max(self.cfg.mix_trans_conf_scale, 1e-8))
        state_conf = 0.5 * float(np.mean(reward_conf)) + 0.5 * float(np.mean(trans_conf))
        avg_pulls = float(np.mean(self.arm_pull_counts))
        global_conf = 1.0 - np.exp(-avg_pulls / max(self.cfg.mix_global_conf_scale, 1e-8))
        return 0.5 * state_conf + 0.5 * global_conf

    def _compute_trust_index(self, states, t):
        if t < self.cfg.mix_trust_warmup_rounds:
            progress = 0.0
        else:
            progress = min(1.0, (t - self.cfg.mix_trust_warmup_rounds) / max(self.cfg.mix_trust_ramp_rounds, 1))
        model_conf = self._compute_model_confidence(states)
        whittle_stability = float(np.exp(-self.whittle_change / max(self.cfg.mix_whittle_change_scale, 1e-8)))
        avg_pulls = float(np.mean(self.arm_pull_counts))
        active_cov = 1.0 - np.exp(-avg_pulls / max(self.cfg.mix_global_conf_scale, 1e-8))
        trust_raw = self._w_conf * model_conf + self._w_prog * progress + self._w_stab * whittle_stability
        trust_gate = 0.25 + 0.75 * active_cov
        trust_t = trust_gate * trust_raw
        switch_floor = 1.0 / (1.0 + np.exp(-(t - self.cfg.mix_trust_switch_round) / max(self.cfg.mix_trust_switch_temp, 1e-8)))
        trust_t = max(trust_t, self.cfg.mix_trust_max * switch_floor)
        return float(np.clip(trust_t, self.cfg.mix_trust_min, self.cfg.mix_trust_max))

    def select_arm(self, states, t):
        if t < self.cfg.mix_warmup_rounds:
            return int(t % self.n_dc)
        eps_t = self._linear_decay(t, self.cfg.mix_eps_start, self.cfg.mix_eps_end, self.cfg.mix_eps_decay_rounds)
        if self.rng.random() < eps_t:
            return int(self.rng.integers(0, self.n_dc))
        self._maybe_replan(t)
        whittle_raw = self._whittle_scores(states)
        trust_t = self._compute_trust_index(states, t)
        greedy_raw = self._greedy_score(states, t)
        whittle_score = normalize_scores_01(whittle_raw)
        greedy_score = normalize_scores_01(greedy_raw)
        mixed_score = trust_t * whittle_score + (1.0 - trust_t) * greedy_score
        return int(np.argmax(mixed_score))

    def update(self, states, chosen_arm, reward, next_states):
        s_sel = int(states[chosen_arm])
        self.reward_post.update((chosen_arm, s_sel), reward)
        self._actions_buf.fill(0)
        self._actions_buf[chosen_arm] = 1
        self.trans_post.alpha[self.arm_ids, states, self._actions_buf, next_states] += 1.0
        self.trans_post.visit_counts[self.arm_ids, states, self._actions_buf] += 1.0
        self.arm_pull_counts[chosen_arm] += 1.0
        self.arm_reward_sums[chosen_arm] += reward


# ---------------------------------------------------------------------------
# Scenario configs
# ---------------------------------------------------------------------------

def make_base_cfg(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


SCENARIO_CONFIGS = {
    "TW only": make_base_cfg(
        mix_trust_min=1.0,                     # force trust = 1 → pure Whittle, no greedy
        mix_trust_max=1.0,
    ),
    "Local UCB + TW": make_base_cfg(
        mix_greedy_global_weight=0.0,          # w_global = 0 always → pure local UCB
    ),
    "Global UCB + TW": make_base_cfg(
        mix_greedy_global_weight=1.0,          # w_global = 1 always
        mix_greedy_global_decay_rounds=10_000, # no decay over 600 rounds
    ),
    "(Local + Global + TW)": make_base_cfg(
        mix_greedy_global_weight=0.95,         # starts mostly global, decays to local
        mix_greedy_global_decay_rounds=180,
    ),
}


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_policy(env, policy, n_rounds, label="", print_every=100):
    states = env.reset()
    rewards = np.zeros((n_rounds,), dtype=float)
    chosen = np.zeros((n_rounds,), dtype=int)
    for t in range(n_rounds):
        arm = policy.select_arm(states, t)
        next_states, reward = env.step(states, arm)
        policy.update(states, arm, reward, next_states)
        rewards[t] = reward
        chosen[t] = arm
        states = next_states
        if print_every > 0 and (t + 1) % print_every == 0:
            avg_r = float(np.mean(rewards[:t + 1]))
            print(f"    [{label}] round {t+1:>{len(str(n_rounds))}}/{n_rounds}  avg_reward={avg_r:.4f}")
    return rewards, chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("results/plots", exist_ok=True)

    datacenter_files = sorted(glob.glob("datacenter_with_metrics/datacenter_*_with_metrics.csv"))[:5]
    if len(datacenter_files) < 5:
        raise RuntimeError(f"Expected 5 datacenter files, found {len(datacenter_files)}")
    raw_dfs = [pd.read_csv(fp) for fp in datacenter_files]
    print(f"Loaded {len(datacenter_files)} datacenters.")

    # Use a shared base config for environment/Oracle parameters.
    base_cfg = Config()
    n_rounds = base_cfg.n_rounds
    n_sims = base_cfg.n_sims
    n_dc = base_cfg.n_dc

    scenario_names = list(SCENARIO_CONFIGS.keys())
    all_names = ["Oracle Whittle"] + scenario_names

    cum_rewards: Dict[str, np.ndarray] = {n: np.zeros(n_rounds) for n in all_names}
    cum_regrets: Dict[str, np.ndarray] = {n: np.zeros(n_rounds) for n in scenario_names}
    avg_rewards: Dict[str, list] = {n: [] for n in all_names}
    sel_counts: Dict[str, np.ndarray] = {n: np.zeros(n_dc, dtype=int) for n in all_names}
    wall_times: Dict[str, list] = {n: [] for n in all_names}

    print("=" * 72)
    print(f"Ablation: TrustMixed greedy component | {n_sims} sims | {n_rounds} rounds")
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
        )
        p_active_true, p_passive_true = build_true_transitions(
            n_states=base_cfg.n_jobs_sample,
            batch_size=base_cfg.batch_size,
            n_dc=n_dc,
        )

        # Oracle
        oracle = OracleWhittlePolicy(R, p_active_true, p_passive_true, base_cfg)
        oracle_env = RMABEnvironment(R, p_active_true, p_passive_true, 0.0, sim_seed + 101)
        t0 = time.perf_counter()
        oracle_rewards, oracle_chosen = run_policy(oracle_env, oracle, n_rounds, label=f"sim{sim+1} Oracle Whittle")
        wall_times["Oracle Whittle"].append(time.perf_counter() - t0)
        oracle_cum = np.cumsum(oracle_rewards)
        cum_rewards["Oracle Whittle"] += oracle_rewards
        avg_rewards["Oracle Whittle"].append(float(np.mean(oracle_rewards)))
        for dc in range(n_dc):
            sel_counts["Oracle Whittle"][dc] += int(np.sum(oracle_chosen == dc))

        msg_parts = [f"sim {sim+1}/{n_sims}", f"Oracle={np.mean(oracle_rewards):.3f}"]

        for i, (name, cfg) in enumerate(SCENARIO_CONFIGS.items()):
            policy = ThompsonWhittleBase(
                n_dc=n_dc,
                n_states=base_cfg.n_jobs_sample,
                batch_size=base_cfg.batch_size,
                cfg=cfg,
                rng_seed=sim_seed + 200 + i * 50,
            )
            env = RMABEnvironment(R, p_active_true, p_passive_true, 0.0, sim_seed + 201 + i * 50)
            t0 = time.perf_counter()
            rewards, chosen = run_policy(env, policy, n_rounds, label=f"sim{sim+1} {name}")
            wall_times[name].append(time.perf_counter() - t0)
            cum_rewards[name] += rewards
            cum_regrets[name] += (oracle_cum - np.cumsum(rewards))
            avg_rewards[name].append(float(np.mean(rewards)))
            for dc in range(n_dc):
                sel_counts[name][dc] += int(np.sum(chosen == dc))
            msg_parts.append(f"{name}={np.mean(rewards):.3f}")

        print("  " + "  |  ".join(msg_parts))

    mean_rewards = {n: cum_rewards[n] / n_sims for n in all_names}
    mean_regrets = {n: cum_regrets[n] / n_sims for n in scenario_names}

    # --- Save raw plot data ---
    plot_data_path = "results/ablation_trust_mixed_plot_data.npz"
    save_kwargs: dict = {
        "all_names": np.array(all_names),
        "scenario_names": np.array(scenario_names),
        "n_rounds": np.array(n_rounds),
        "n_sims": np.array(n_sims),
        "n_dc": np.array(n_dc),
    }
    for name in all_names:
        key = name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        save_kwargs[f"mean_rewards__{key}"] = mean_rewards[name]
        save_kwargs[f"avg_rewards__{key}"] = np.array(avg_rewards[name])
        save_kwargs[f"sel_counts__{key}"] = sel_counts[name]
        if name in scenario_names:
            save_kwargs[f"mean_regrets__{key}"] = mean_regrets[name]
    np.savez(plot_data_path, **save_kwargs)
    print(f"Saved plot data: {plot_data_path}")

    print("\nAverage reward over simulations:")
    for name in all_names:
        print(f"  {name:<38} {np.mean(avg_rewards[name]):.4f}")

    rounds_axis = np.arange(1, n_rounds + 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Plot 1: per-round average reward ---
    _label_map = {
        "All three (Local + Global + TW)": "Local + Global + TW",
        "(Local + Global + TW)": "Local + Global + TW",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    non_oracle_idx = 0
    for i, name in enumerate(all_names):
        if name == "Oracle Whittle":
            _color = "black"
        else:
            _color = colors[non_oracle_idx % len(colors)]
            non_oracle_idx += 1
        ax.plot(
            rounds_axis,
            np.cumsum(mean_rewards[name]) / rounds_axis,
            label=f"{_label_map.get(name, name)} (avg={np.mean(avg_rewards[name]):.3f})",
            linewidth=2.0,
            linestyle="--" if name == "Oracle Whittle" else "-",
            color=_color,
        )
    ax.set_xlabel("Round", fontsize=14)
    ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T r_t$", fontsize=14)
    ax.set_xlim(0, 1000)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=13)
    plt.tight_layout()
    out1 = "results/plots/ablation_trust_mixed_reward.png"
    plt.savefig(out1, bbox_inches="tight", dpi=150)
    print(f"\nSaved: {out1}")
    plt.close()

    # --- Plot 2: per-round average regret ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(scenario_names):
        ax.plot(
            rounds_axis,
            mean_regrets[name] / rounds_axis,
            label=name,
            linewidth=2.0,
            color=colors[(i + 1) % len(colors)],
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Round")
    ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T (r_t^* - r_t)$")
    ax.set_title(
        "Ablation: Per-round average regret vs Oracle Whittle\n"
        f"({n_dc} DCs, {base_cfg.n_jobs_sample} states, {n_rounds} rounds, {n_sims} sims)"
    )
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out2 = "results/plots/ablation_trust_mixed_regret.png"
    plt.savefig(out2, bbox_inches="tight", dpi=150)
    print(f"Saved: {out2}")
    plt.close()

    # --- Plot 3: arm selection frequency ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_dc)
    width = 0.18
    non_oracle_idx = 0
    for i, name in enumerate(all_names):
        if name == "Oracle Whittle":
            _bar_color, _hatch, _ec, _lw = "white", "////", "black", 1.0
        else:
            _bar_color = colors[non_oracle_idx % len(colors)]
            _hatch, _ec, _lw = "", "black", 0.5
            non_oracle_idx += 1
        ax.bar(
            x + (i - (len(all_names) - 1) / 2.0) * width,
            100.0 * sel_counts[name] / (n_sims * n_rounds),
            width=width,
            label=_label_map.get(name, name),
            color=_bar_color,
            hatch=_hatch,
            edgecolor=_ec,
            linewidth=_lw,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"DC{dc}" for dc in range(n_dc)], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Selection frequency (%)", fontsize=14)
    ax.legend(loc="best", fontsize=13)
    plt.tight_layout()
    out3 = "results/plots/ablation_trust_mixed_arm_selection.png"
    plt.savefig(out3, bbox_inches="tight", dpi=150)
    print(f"Saved: {out3}")
    plt.close()

    # --- CSV summary ---
    first_k = min(100, n_rounds)
    rows = []
    for name in all_names:
        row = {
            "policy": name,
            "avg_reward": float(np.mean(avg_rewards[name])),
            "cum_reward": float(np.sum(mean_rewards[name])),
            "first100_avg_reward": float(np.mean(mean_rewards[name][:first_k])),
            "avg_regret_vs_oracle": 0.0 if name == "Oracle Whittle"
                                    else float(np.mean(mean_regrets[name])),
            "avg_wall_time_s": float(np.mean(wall_times[name])),
            "total_wall_time_s": float(np.sum(wall_times[name])),
        }
        for dc in range(n_dc):
            row[f"sel_pct_dc{dc}"] = round(100.0 * sel_counts[name][dc] / (n_sims * n_rounds), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv("results/ablation_trust_mixed_summary.csv", index=False)
    print("Saved: results/ablation_trust_mixed_summary.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
