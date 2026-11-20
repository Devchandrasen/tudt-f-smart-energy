"""Render the README chart from the archived split-conformal scenario means."""

import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


DOCS = Path(__file__).resolve().parent
SOURCE = DOCS.parent / "evidence/results/interval_summary.csv"
TARGET = DOCS / "assets/dispatch-conditions.png"
CONDITIONS = {
    "clean": "Clean input",
    "delayed": "Two-hour delay",
    "missing": "Missing readings",
    "noisy": "Noisy readings",
    "disturbance": "Grid disturbance",
    "fdi_like": "Injected spikes",
    "bias": "Fixed bias",
    "slow_ramp": "Slow ramp",
}


def render():
    with SOURCE.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["interval"] == "split"]
    results = {row["scenario"]: row for row in rows}
    if results.keys() != CONDITIONS.keys() or len(rows) != len(CONDITIONS):
        raise ValueError("Expected one split-interval result per displayed condition.")

    ink = "#142f39"
    muted = "#455e67"
    fig = Figure(figsize=(9.5, 5.35), facecolor="#fafbf9")
    FigureCanvasAgg(fig)
    fig.add_artist(Rectangle((0, 0.765), 1, 0.235, transform=fig.transFigure,
                             facecolor=ink, edgecolor="none", zorder=0))
    fig.text(0.038, 0.899, "When does battery dispatch pause?", fontsize=23,
             color="white", weight="bold", va="center")
    fig.text(0.04, 0.817, "Germany–Luxembourg   /   Nine 168-hour windows   /   Split-conformal intervals",
             fontsize=12, color="#d6e8e8", va="center")

    # Each panel spans 0–100%. A separate numeric column keeps small rates legible.
    panels = [
        ("load_coverage", "Load interval coverage", 0.285, "#176d75"),
        ("abstention", "Dispatch abstention", 0.662, "#9b4b27"),
    ]
    row_positions = range(len(CONDITIONS))
    fig.text(0.04, 0.68, "Input condition", fontsize=12, color=muted, weight="bold")
    for metric, title, left, color in panels:
        fig.text(left, 0.68, title, fontsize=12, color=ink, weight="bold")
        ax = fig.add_axes((left, 0.095, 0.25, 0.54), facecolor="none")
        rates = [100 * float(results[name][metric]) for name in CONDITIONS]
        ax.set(xlim=(0, 100), ylim=(len(CONDITIONS) - 0.5, -0.5))
        ax.barh(row_positions, [100] * len(CONDITIONS), height=0.27, color="#dce5e4")
        ax.barh(row_positions, rates, height=0.27, color=color)
        ax.set_axis_off()
        for y, (label, rate) in enumerate(zip(CONDITIONS.values(), rates)):
            row_transform = ax.get_yaxis_transform()
            ax.text(1.035, y, f"{rate:.1f}%", transform=row_transform, fontsize=11,
                    va="center", color=ink)
            if metric == "load_coverage":
                ax.text((0.04 - left) / ax.get_position().width, y, label, transform=row_transform,
                        fontsize=12, va="center", color=ink)

    fig.text(0.04, 0.038, "Both columns show the percentage of evaluated hours, averaged across windows.",
             fontsize=10, color=muted)
    TARGET.parent.mkdir(exist_ok=True)
    fig.savefig(TARGET, dpi=160, metadata={"Title": "Load interval coverage and dispatch abstention",
                                          "Source": "evidence/results/interval_summary.csv (split rows)"})


if __name__ == "__main__":
    render()
