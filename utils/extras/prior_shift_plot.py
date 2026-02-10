#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt

# Metric -> CSV column name
METRIC_COLS = OrderedDict([
    ("AV", "av"),
    ("AC", "ac"),
    ("PR", "pr"),
    ("UI", "ui"),
    ("S",  "s"),
    ("C",  "c"),
    ("I",  "i"),
    ("A",  "a"),
])

# Per-metric class order (so bars are consistent)
CLASS_ORDER = {
    "AV": ["NETWORK", "ADJACENT_NETWORK", "LOCAL", "PHYSICAL"],
    "AC": ["LOW", "HIGH"],
    "PR": ["NONE", "LOW", "HIGH"],
    "UI": ["NONE", "REQUIRED"],
    "S":  ["UNCHANGED", "CHANGED"],
    "C":  ["NONE", "LOW", "HIGH"],
    "I":  ["NONE", "LOW", "HIGH"],
    "A":  ["NONE", "LOW", "HIGH"],
}

def compute_priors(df: pd.DataFrame):
    """
    Compute class counts and percentages per metric.

    Returns:
        counts: DataFrame indexed by (Metric, Class) with a 'count' column.
        perc:   DataFrame indexed by (Metric, Class) with a 'percent' column.
    """
    rows = []
    for metric_label, col in METRIC_COLS.items():
        series = df[col].astype(str).str.upper().str.strip()
        total = len(series)
        vc = series.value_counts()
        for cls in CLASS_ORDER[metric_label]:
            n = vc.get(cls, 0)
            pct = 100.0 * n / total if total > 0 else 0.0
            rows.append({
                "Metric": metric_label,
                "Class": cls,
                "count": int(n),
                "percent": pct,
            })

    priors_df = pd.DataFrame(rows)
    counts = priors_df.set_index(["Metric", "Class"])[["count"]]
    perc = priors_df.set_index(["Metric", "Class"])[["percent"]]
    return counts, perc


def print_prior_table(perc: pd.DataFrame, label: str):
    """
    Print a compact text table for sanity checking and LaTeX copy-paste help.
    """
    print(f"\n=== Class priors (%): {label} ===")
    df = perc.reset_index()
    for metric in METRIC_COLS.keys():
        sub = df[df["Metric"] == metric].copy()
        if sub.empty:
            continue
        print(f"\n[{metric}]")
        for _, row in sub.iterrows():
            print(f"  {row['Class']:>18}: {row['percent']:6.2f}%")


def plot_stacked_bars(perc: pd.DataFrame, title: str, out_path: Path):
    """
    Make a stacked horizontal bar chart with:
      - Left panel: AV, AC, PR, UI
      - Right panel: S, C, I, A
    Each bar is stacked by class proportion.
    """
    # Convert back to a friendlier dict: metric -> {class: proportion 0-1}
    df = perc.reset_index()
    prior_dict = {}
    for metric in METRIC_COLS.keys():
        sub = df[df["Metric"] == metric]
        total_pct = sub["percent"].sum()  # should be ~100
        # Use fractions, not percents, for plotting
        prior_dict[metric] = {
            row["Class"]: float(row["percent"] / 100.0)
            for _, row in sub.iterrows()
        }

    exploit_metrics = ["AV", "AC", "PR", "UI"]
    impact_metrics = ["S", "C", "I", "A"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4.5),
        sharex=True,
        gridspec_kw={"width_ratios": [1, 1]}
    )

    def plot_panel(ax, metrics, panel_title):
        y_pos = range(len(metrics))
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(metrics)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Class proportion")
        ax.set_title(panel_title)

        # We will reuse the same colour cycle across metrics and classes,
        # so legend refers to classes, not metrics.
        all_handles = OrderedDict()

        for i, metric in enumerate(metrics):
            left = 0.0
            for cls in CLASS_ORDER[metric]:
                frac = prior_dict.get(metric, {}).get(cls, 0.0)
                if frac <= 0:
                    continue
                bar = ax.barh(i, frac, left=left, label=cls)
                left += frac
                # Store handle for legend (one per class)
                if cls not in all_handles:
                    all_handles[cls] = bar[0]

        # Only build legend from the first panel;
        # we will move it outside in the main function.
        return all_handles

    handles_left = plot_panel(axes[0], exploit_metrics, "Exploitability metrics")
    handles_right = plot_panel(axes[1], impact_metrics, "Scope and impact metrics")

    # Build a single legend using the union of class labels
    handles = handles_left.copy()
    handles.update(handles_right)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0.0, 1, 0.90])

    # Place legend below
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.20)
    print(f"Saving figure to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV with CVSS base metrics")
    ap.add_argument("--title", default="Class priors by metric",
                    help="Title for the plot")
    ap.add_argument("--out", default=None,
                    help="Output PNG path (default: same stem + '_priors.png')")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    counts, perc = compute_priors(df)
    print_prior_table(perc, label=csv_path.name)

    if args.out is None:
        out_path = csv_path.with_name(csv_path.stem + "_priors.png")
    else:
        out_path = Path(args.out)

    plot_stacked_bars(perc, title=args.title, out_path=out_path)

    # Optionally: save priors to CSV for LaTeX tables
    priors_out = csv_path.with_name(csv_path.stem + "_priors_table.csv")
    print(f"Saving priors table to {priors_out}")
    perc.reset_index().to_csv(priors_out, index=False)


if __name__ == "__main__":
    main()
