"""
Plot CAISO 5-minute LMP traces for the five nearest nodes at each data center.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "racer_cdc_mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "caiso_25nodelmp_at5dc.csv"
OUTPUT_PATH = SCRIPT_DIR / "caiso_5node_lmp_by_datacenter.png"

META_COLUMNS = ["Date", "Data Center", "Latitude", "Longitude", "NODE"]


def main() -> None:
    df = pd.read_csv(INPUT_PATH, skiprows=1)
    time_columns = [col for col in df.columns if col not in META_COLUMNS]
    data_centers = sorted(df["Data Center"].unique())

    fig, axes = plt.subplots(
        len(data_centers),
        1,
        figsize=(13, 12),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if len(data_centers) == 1:
        axes = [axes]

    x = range(len(time_columns))
    tick_positions = list(range(0, len(time_columns), 24))
    tick_labels = [time_columns[i] for i in tick_positions]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for ax, data_center in zip(axes, data_centers):
        sub = df[df["Data Center"] == data_center].sort_values("NODE")
        for color, (_, row) in zip(colors, sub.iterrows()):
            values = pd.to_numeric(row[time_columns], errors="coerce")
            ax.plot(x, values, linewidth=1.4, color=color, label=row["NODE"])

        date = sub["Date"].iloc[0]
        ax.set_title(data_center, fontsize=13, loc="left")
        ax.set_ylabel("LMP ($/MWh)", fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.legend(ncol=5, fontsize=8, loc="upper right", frameon=False)
        ax.text(
            0.0,
            0.92,
            str(date),
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            va="top",
        )

    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Time of day", fontsize=11)

    fig.suptitle("CAISO 5-Minute LMP at Five Nearest Nodes per Data Center", fontsize=16)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
