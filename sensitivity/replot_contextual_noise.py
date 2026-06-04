"""Regenerate sensitivity_contextual_noise_combined.png from saved CSV.

Reads sensitivity_contextual_noise_summary_new.csv when available (includes
Event-Driven TW), falls back to sensitivity_contextual_noise_summary.csv.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_new_csv = SCRIPT_DIR / "sensitivity_contextual_noise_summary_new.csv"
_old_csv = SCRIPT_DIR / "sensitivity_contextual_noise_summary.csv"
csv_path = _new_csv if os.path.exists(_new_csv) else _old_csv
df = pd.read_csv(csv_path)
print(f"Reading: {csv_path}")

_base_policies = ["Oracle Whittle", "Thompson-Whittle", "Trust-Mixed TW", "State Thompson"]
_extra         = [p for p in df["policy"].unique() if p not in _base_policies]
policy_names   = [p for p in _base_policies + _extra if p in df["policy"].values]
scenario_names = [p for p in policy_names if p != "Oracle Whittle"]

noise_levels = sorted(df["state_noise"].unique())

avg_reward = {
    name: [df[(df.policy == name) & (df.state_noise == n)]["avg_reward"].values[0]
           for n in noise_levels]
    for name in policy_names
}
avg_regret = {
    name: [df[(df.policy == name) & (df.state_noise == n)]["avg_regret_vs_oracle"].values[0]
           for n in noise_levels]
    for name in scenario_names
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

FS_LABEL  = 24
FS_TITLE  = 25
FS_TICK   = 22
FS_LEGEND = 22
LINE_STYLES = ["--", "-", "--", "-.", ":"]
MARKERS     = ["o", "s", "^", "D", "v"]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# --- Panel (a): average reward ---
ax = axes[0]
non_oracle_idx = 0
for name in policy_names:
    if name == "Oracle Whittle":
        _color, _ls, _marker = "black", "--", "o"
    else:
        _color  = colors[non_oracle_idx % len(colors)]
        _ls     = LINE_STYLES[1 + non_oracle_idx % (len(LINE_STYLES) - 1)]
        _marker = MARKERS[1 + non_oracle_idx % (len(MARKERS) - 1)]
        non_oracle_idx += 1
    ax.plot(noise_levels, avg_reward[name],
            marker=_marker, markersize=10, linewidth=3.5, linestyle=_ls, color=_color, label=name)
ax.set_xlabel("State-observation noise probability", fontsize=FS_LABEL)
ax.set_ylabel("Average reward", fontsize=FS_LABEL)
ax.set_title("(a) Average reward", fontsize=FS_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)

# --- Panel (b): average cumulative regret vs Oracle ---
ax = axes[1]
ax.axhline(0.0, color="black", linewidth=3.5, linestyle="--", label="Oracle Whittle")
for i, name in enumerate(scenario_names):
    ax.plot(noise_levels, avg_regret[name],
            marker=MARKERS[1 + i % (len(MARKERS) - 1)],
            markersize=10, linewidth=3.5,
            linestyle=LINE_STYLES[1 + i % (len(LINE_STYLES) - 1)],
            color=colors[i % len(colors)], label=name)
ax.set_xlabel("State-observation noise probability", fontsize=FS_LABEL)
ax.set_ylabel("Average cumulative\nregret vs Oracle", fontsize=FS_LABEL)
ax.set_title("(b) Average cumulative regret vs Oracle", fontsize=FS_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)

# --- Single shared legend at the bottom ---
handles, labels = axes[0].get_legend_handles_labels()  # has all 4 policies incl. Oracle
fig.legend(handles, labels, loc="lower center", fontsize=FS_LEGEND,
           ncol=min(len(policy_names), 5), bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
out = SCRIPT_DIR / "sensitivity_contextual_noise_combined_new.png"
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {out}")
