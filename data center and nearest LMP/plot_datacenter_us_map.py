"""
Plot data center locations from the market LMP CSVs on a contiguous US map.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "racer_cdc_mplconfig"))

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "datacenter_us_map.png"
LABEL_PATH = SCRIPT_DIR / "datacenter_us_map_labels.csv"

MARKETS = [
    {
        "market": "CAISO",
        "file": "caiso_finalized.csv",
        "skiprows": 0,
        "data_center_col": "Data Center",
    },
    {
        "market": "ERCOT",
        "file": "ercot_load_zone_15min.csv",
        "skiprows": 1,
        "data_center_col": "Data Center",
    },
    {
        "market": "ISONE",
        "file": "combined_ISONE_5min_lmp.csv",
        "skiprows": 1,
        "data_center_col": "Data Center",
    },
    {
        "market": "PJM",
        "file": "combined_PJM_5min_lmp.csv",
        "skiprows": 1,
        "data_center_col": "Data center",
    },
    {
        "market": "NYISO",
        "file": "NYISO_zone_lmp_5min.csv",
        "skiprows": 1,
        "data_center_col": "Data center",
    },
]

COLORS = {
    "CAISO": "#1f77b4",
    "ERCOT": "#d62728",
    "ISONE": "#2ca02c",
    "PJM": "#9467bd",
    "NYISO": "#ff7f0e",
}


def load_locations() -> pd.DataFrame:
    frames = []
    for config in MARKETS:
        df = pd.read_csv(SCRIPT_DIR / config["file"], skiprows=config["skiprows"])
        subset = df[[config["data_center_col"], "Latitude", "Longitude"]].copy()
        subset = subset.rename(columns={config["data_center_col"]: "Data Center"})
        subset["Market"] = config["market"]
        subset["Source File"] = config["file"]
        frames.append(subset)

    locations = pd.concat(frames, ignore_index=True)
    locations["Latitude"] = pd.to_numeric(locations["Latitude"], errors="coerce")
    locations["Longitude"] = pd.to_numeric(locations["Longitude"], errors="coerce")
    locations = locations.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    # A few source rows use positive west longitudes. Keep CSVs untouched, but
    # normalize for map placement within the contiguous US.
    locations.loc[locations["Longitude"] > 0, "Longitude"] *= -1

    locations["Map Label"] = (
        locations["Market"] + "-" + (locations.groupby("Market").cumcount() + 1).astype(str)
    )
    return locations


def main() -> None:
    locations = load_locations()

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    usa = world[world["name"] == "United States of America"]

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    usa.plot(ax=ax, color="#f3f0e8", edgecolor="#8f8f8f", linewidth=0.8)

    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_aspect("equal", adjustable="box")

    label_offsets = {
        "CAISO": [(8, 8), (8, -3), (8, 18), (8, -15), (8, -26)],
        "ERCOT": [(8, 0), (8, 10), (8, -6), (8, 0), (8, 6), (8, -10)],
        "ISONE": [(8, 12), (8, -2), (8, -16)],
        "PJM": [(8, 2), (8, 12), (8, -8), (8, 6), (8, -8)],
        "NYISO": [(8, 10), (8, 6), (8, -8), (-32, 8), (-32, -8)],
    }

    for market, sub in locations.groupby("Market", sort=False):
        ax.scatter(
            sub["Longitude"],
            sub["Latitude"],
            s=72,
            color=COLORS[market],
            edgecolor="white",
            linewidth=0.9,
            label=f"{market} ({len(sub)})",
            zorder=3,
        )
        for offset, (_, row) in zip(label_offsets[market], sub.iterrows()):
            ax.annotate(
                row["Map Label"],
                (row["Longitude"], row["Latitude"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color="#202020",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#808080",
                    "linewidth": 0.4,
                    "shrinkA": 0,
                    "shrinkB": 3,
                },
                zorder=4,
            )

    ax.set_title("Data Center Locations by Electricity Market", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, color="#d7d7d7", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=10)

    locations[
        ["Map Label", "Market", "Data Center", "Latitude", "Longitude", "Source File"]
    ].to_csv(LABEL_PATH, index=False)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {LABEL_PATH}")


if __name__ == "__main__":
    main()
