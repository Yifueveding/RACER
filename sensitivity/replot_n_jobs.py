"""Regenerate sensitivity_n_jobs_combined.png from saved CSV.

Reads sensitivity_n_jobs_summary_new.csv when available (includes Event-Driven TW),
falls back to sensitivity_n_jobs_summary.csv otherwise.
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
_new_csv = SCRIPT_DIR / "sensitivity_n_jobs_summary_new.csv"
_old_csv = SCRIPT_DIR / "sensitivity_n_jobs_summary.csv"
csv_path = _new_csv if os.path.exists(_new_csv) else _old_csv
df = pd.read_csv(csv_path)
print(f"Reading: {csv_path}")

_base_policies = ["Oracle Whittle", "Thompson-Whittle", "Trust-Mixed TW", "State Thompson"]
_extra = [p for p in df["policy"].unique() if p not in _base_policies]
policy_names = _base_policies + _extra

n_jobs_levels = sorted(df["n_jobs_sample"].unique())

# Only include policies actually present in the CSV
policy_names = [p for p in policy_names if p in df["policy"].values]

avg_reward = {
    name: [df[(df.policy == name) & (df.n_jobs_sample == n)]["avg_reward"].values[0]
           for n in n_jobs_levels]
    for name in policy_names
}
avg_time = {
    name: [df[(df.policy == name) & (df.n_jobs_sample == n)]["avg_sim_time_s"].values[0]
           for n in n_jobs_levels]
    for name in policy_names
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
for idx, name in enumerate(policy_names):
    if name == "Oracle Whittle":
        _color, _ls, _marker = "black", "--", "o"
    else:
        _color = colors[non_oracle_idx % len(colors)]
        _ls    = LINE_STYLES[1 + non_oracle_idx % (len(LINE_STYLES) - 1)]
        _marker = MARKERS[1 + non_oracle_idx % (len(MARKERS) - 1)]
        non_oracle_idx += 1
    ax.plot(n_jobs_levels, avg_reward[name],
            marker=_marker, linewidth=3.5, markersize=10, linestyle=_ls, color=_color, label=name)
ax.set_xlabel("Number of job samples", fontsize=FS_LABEL)
ax.set_ylabel("Average reward", fontsize=FS_LABEL)
ax.set_title("(a) Average reward", fontsize=FS_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)

# --- Panel (b): average simulation time ---
ax = axes[1]
non_oracle_idx = 0
for idx, name in enumerate(policy_names):
    if name == "Oracle Whittle":
        _color, _ls, _marker = "black", "--", "o"
    else:
        _color = colors[non_oracle_idx % len(colors)]
        _ls    = LINE_STYLES[1 + non_oracle_idx % (len(LINE_STYLES) - 1)]
        _marker = MARKERS[1 + non_oracle_idx % (len(MARKERS) - 1)]
        non_oracle_idx += 1
    ax.plot(n_jobs_levels, avg_time[name],
            marker=_marker, linewidth=3.5, markersize=10, linestyle=_ls, color=_color, label=name)
ax.set_xlabel("Number of job samples", fontsize=FS_LABEL)
ax.set_ylabel("Average simulation time (s)", fontsize=FS_LABEL)
ax.set_title("(b) Average simulation time", fontsize=FS_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)

# --- Single shared legend at the bottom ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", fontsize=FS_LEGEND,
           ncol=min(len(policy_names), 5), bbox_to_anchor=(0.5, -0.08))

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
out = SCRIPT_DIR / "sensitivity_n_jobs_combined_new.png"
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {out}")
