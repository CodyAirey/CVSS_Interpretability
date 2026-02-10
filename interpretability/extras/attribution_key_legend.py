#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def signed_rgba_bar(n: int = 800) -> np.ndarray:
    w = np.linspace(-1.0, 1.0, n)
    alpha = (30.0 + 200.0 * np.abs(w)) / 255.0

    rgba = np.zeros((1, n, 4), dtype=float)
    pos = w >= 0

    # negative -> blue
    rgba[0, ~pos, 2] = 1.0
    rgba[0, ~pos, 3] = alpha[~pos]

    # positive -> red
    rgba[0, pos, 0] = 1.0
    rgba[0, pos, 3] = alpha[pos]

    return rgba


def lrp_rgba_bar(n: int = 800) -> np.ndarray:
    w = np.linspace(0.0, 1.0, n)
    alpha = (30.0 + 200.0 * w) / 255.0

    rgba = np.zeros((1, n, 4), dtype=float)
    rgba[..., 0] = 1.0
    rgba[..., 1] = 165 / 255.0
    rgba[..., 2] = 0.0
    rgba[..., 3] = alpha
    return rgba


def make_key(out_path: str = "figures/attribution_colour_key.svg",
             width: float = 7.2,
             height: float = 2.7) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    title_fs = 11
    label_fs = 10

    fig = plt.figure(figsize=(width, height), dpi=150)

    # Two blocks, each with: title / bar / labels
    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, 1.0],
        hspace=0.55
    )

    def add_block(gs, title: str, img: np.ndarray, labels: tuple[str, ...]) -> None:
        sub = gs.subgridspec(
            nrows=3, ncols=1,
            height_ratios=[0.28, 0.52, 0.20],  # title close to bar; labels have their own row
            hspace=0.15
        )

        ax_title = fig.add_subplot(sub[0, 0])
        ax_title.axis("off")
        ax_title.text(
            0.0, 0.5, title,
            ha="left", va="center",
            fontsize=title_fs, fontweight="semibold"
        )

        ax_bar = fig.add_subplot(sub[1, 0])
        ax_bar.set_axis_off()
        ax_bar.imshow(img, aspect="auto", interpolation="nearest")

        ax_lab = fig.add_subplot(sub[2, 0])
        ax_lab.axis("off")

        if len(labels) == 3:
            ax_lab.text(0.00, 0.35, labels[0], ha="left",   va="center", fontsize=label_fs)
            ax_lab.text(0.50, 0.35, labels[1], ha="center", va="center", fontsize=label_fs)
            ax_lab.text(1.00, 0.35, labels[2], ha="right",  va="center", fontsize=label_fs)
        elif len(labels) == 2:
            ax_lab.text(0.00, 0.35, labels[0], ha="left",  va="center", fontsize=label_fs)
            ax_lab.text(1.00, 0.35, labels[1], ha="right", va="center", fontsize=label_fs)
        else:
            # fall back: evenly spaced
            xs = np.linspace(0, 1, len(labels))
            for x, lab in zip(xs, labels):
                ax_lab.text(float(x), 0.35, lab, ha="center", va="center", fontsize=label_fs)

    add_block(
        outer[0, 0],
        "Signed attribution (IG, LIME, SHAP)",
        signed_rgba_bar(),
        ("Negative", "Neutral", "Positive"),
    )

    add_block(
        outer[1, 0],
        "Non-negative relevance (LRP)",
        lrp_rgba_bar(),
        ("Low", "High"),
    )

    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_key()
