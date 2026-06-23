"""
Plot LMP traces for ERCOT, ISONE, and PJM in separate figures.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "racer_cdc_mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

MARKETS = [
    {
        "name": "ERCOT",
        "input": SCRIPT_DIR / "combined_ERCOT_lmp.csv",
        "output": SCRIPT_DIR / "ercot_lmp_at_datacenters.png",
        "date_col": "deliveryDate",
        "data_center_col": "Data Center",
        "meta_cols": [
            "deliveryDate",
            "Data Center",
            "Latitude",
            "Longitude",
            "settlementPoint",
            "settlementPointType",
        ],
    },
    {
        "name": "ISONE",
        "input": SCRIPT_DIR / "combined_ISONE_5min_lmp.csv",
        "output": SCRIPT_DIR / "isone_lmp_at_datacenters.png",
        "date_col": "Date",
        "data_center_col": "Data Center",
        "meta_cols": [
            "Date",
            "Data Center",
            "Latitude",
            "Longitude",
            "location_id",
            "Load Zone",
        ],
    },
    {
        "name": "PJM",
        "input": SCRIPT_DIR / "combined_PJM_5min_lmp.csv",
        "output": SCRIPT_DIR / "pjm_lmp_at_datacenters.png",
        "date_col": "date",
        "data_center_col": "Data center",
        "meta_cols": [
            "date",
            "Data center",
            "Latitude",
            "Longitude",
            "pnode_id",
            "pnode_name",
            "type",
            "zone",
        ],
    },
]


def short_time_label(label: str) -> str:
    return label[:5]


def plot_market(config: dict[str, object]) -> None:
    df = pd.read_csv(config["input"], skiprows=1)
    meta_cols = set(config["meta_cols"])
    time_columns = [col for col in df.columns if col not in meta_cols]

    fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    x = range(len(time_columns))
    tick_step = 4 if len(time_columns) <= 96 else 24
    tick_positions = list(range(0, len(time_columns), tick_step))
    tick_labels = [short_time_label(time_columns[i]) for i in tick_positions]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    data_center_col = config["data_center_col"]
    for color, (_, row) in zip(colors, df.sort_values(data_center_col).iterrows()):
        values = pd.to_numeric(row[time_columns], errors="coerce")
        ax.plot(x, values, linewidth=1.7, color=color, label=row[data_center_col])

    date = df[config["date_col"]].iloc[0]
    ax.set_title(f"{config['name']} LMP at Data Centers ({date})", fontsize=15)
    ax.set_xlabel("Time of day", fontsize=12)
    ax.set_ylabel("LMP ($/MWh)", fontsize=12)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)

    fig.savefig(config["output"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {config['output']}")


def main() -> None:
    for config in MARKETS:
        plot_market(config)


if __name__ == "__main__":
    main()
