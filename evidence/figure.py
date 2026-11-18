"""Render Figure 4 from the published S8 results and the weighted follow-up."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for arg in ("archived", "followup", "pdf"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(args.archived), pd.read_csv(args.followup).query("interval=='weighted'")])
    order = ["clean", "missing", "delayed", "noisy", "disturbance", "fdi_like", "bias", "slow_ramp"]
    labels = ["Clean", "Missing", "Delay", "Noise", "Disturb.", "Abrupt", "Bias", "Ramp"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.5), layout="constrained")
    styles = [("split","Split","o","#24588d"), ("weighted","Weighted","D","#a44221"),
              ("block_max","Block maxima","^","#16715a"), ("adaptive","Adaptive","s","#755392")]
    for ax, metric, scale, ylabel in zip(axes, ["load_coverage","width_kw"], [100,1],
                                        ["Load coverage (%)","Combined interval width (kW)"]):
        for i, (key, label, marker, color) in enumerate(styles):
            series = data.query("interval==@key").set_index("scenario").loc[order, metric]
            ax.scatter(np.arange(8)+(i-1.5)*.14, series*scale, s=21, c=color, marker=marker, label=label, zorder=3)
        ax.set_xticks(range(8),labels, rotation=30, ha="right",fontsize=8)
        ax.set_ylabel(ylabel,fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#dedede", lw=.6)
        ax.tick_params(axis="y", labelsize=8)
    axes[0].axhline(90, color="#555555", lw=.7, ls="--")
    axes[0].set_ylim(35,104)
    axes[1].legend(loc="center left", ncol=2, fontsize=7, frameon=True, columnspacing=.4, handletextpad=.25)
    fig.savefig(args.pdf,metadata={"Title":"Empirical interval coverage and width across eight conditions","Creator":"Matplotlib"})
