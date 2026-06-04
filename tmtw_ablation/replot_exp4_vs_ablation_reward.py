import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
exp4 = np.load(SCRIPT_DIR / "exp4_strategy_plot_data.npz", allow_pickle=True)
abl = np.load(SCRIPT_DIR / "ablation_trust_mixed_plot_data.npz", allow_pickle=True)
tw = np.load(SCRIPT_DIR / "thompson_whittle_plot_data.npz", allow_pickle=True)

n_rounds = int(exp4["n_rounds"])
rounds_axis = np.arange(1, n_rounds + 1)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FS_AXIS_LABEL = 18
FS_TICK = 15
FS_LEGEND = 12
FS_BAR_TITLE = 17
FS_BAR_LABEL = 14
FS_BAR_XTICK = 12
FS_BAR_TICK = 13
FS_VALUE = 11

entries = [
    ("Oracle Whittle", "Oracle_Whittle", "black", "--"),
    ("EXP4-Strategy", None, None, "-"),
    ("(Local+Global+TW)", "Local_plus_Global_plus_TW", None, "-"),
    ("Global UCB + TW", "Global_UCB_plus_TW", None, "-"),
    ("Local UCB + TW", "Local_UCB_plus_TW", None, "-"),
    ("TW only", "TW_only", None, "-"),
    ("State Thompson", "State_Thompson", None, "-"),
]

line_series = []
color_idx = 0
for label, abl_key, forced_color, linestyle in entries:
    if forced_color is None:
        color = colors[color_idx % len(colors)]
        color_idx += 1
    else:
        color = forced_color

    if abl_key is None:
        rewards = exp4["mean_rewards__EXP4-Strategy"]
        avg = float(np.mean(exp4["avg_rewards__EXP4-Strategy"]))
    elif abl_key == "State_Thompson":
        if int(tw["n_rounds"]) != n_rounds:
            raise ValueError("State Thompson cache has a different number of rounds")
        rewards = tw["mean_rewards__State_Thompson"]
        avg = float(np.mean(tw["avg_rewards__State_Thompson"]))
    else:
        rewards = abl[f"mean_rewards__{abl_key}"]
        avg = float(np.mean(abl[f"avg_rewards__{abl_key}"]))

    line_series.append(
        (label, np.cumsum(rewards) / rounds_axis, avg, color, linestyle)
    )

fig, ax = plt.subplots(figsize=(13, 5.8))
for label, running_avg, avg, color, linestyle in line_series:
    ax.plot(
        rounds_axis,
        running_avg,
        label=f"{label} (avg={avg:.3f})",
        linewidth=2.0,
        linestyle=linestyle,
        color=color,
    )

ax.set_xlabel("Round", fontsize=FS_AXIS_LABEL)
ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T r_t$", fontsize=FS_AXIS_LABEL)
ax.set_xlim(0, n_rounds)
ax.set_ylim(top=25.0)
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.97),
    fontsize=FS_LEGEND,
    ncol=2,
)

plt.tight_layout()
out = SCRIPT_DIR / "exp4_vs_ablation_reward_line.png"
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved: {out}")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for bar_ax, checkpoint in zip(
    axes,
    [100, n_rounds],
):
    idx = checkpoint - 1
    labels = [entry[0] for entry in line_series]
    values = [entry[1][idx] for entry in line_series]
    bar_colors = [entry[3] for entry in line_series]

    bar_ax.bar(labels, values, color=bar_colors, alpha=0.9)
    bar_ax.set_title(f"Round {checkpoint}", fontsize=FS_BAR_TITLE)
    bar_ax.set_ylabel(r"$T^{-1}\sum_{t=1}^T r_t$", fontsize=FS_BAR_LABEL)
    bar_ax.set_ylim(0.0, max(values) * 1.12)
    bar_ax.tick_params(axis="x", labelrotation=25, labelsize=FS_BAR_XTICK)
    bar_ax.tick_params(axis="y", labelsize=FS_BAR_TICK)
    for tick in bar_ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    for i, value in enumerate(values):
        bar_ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=FS_VALUE)

plt.tight_layout()
out = SCRIPT_DIR / "exp4_vs_ablation_reward_bars.png"
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"Saved: {out}")
plt.close()
