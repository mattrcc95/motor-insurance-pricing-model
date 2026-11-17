"""Shared, consistent plotting style for all notebooks.

Colours follow a fixed categorical order (validated for colour-blind
separation): the same entity always gets the same colour across notebooks.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import FIGURES_DIR

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"

# Fixed entity -> colour mapping used everywhere
MODEL_COLORS = {
    "Observed": INK_2,
    "Null": "#a8a7a2",
    "GLM": SERIES[0],
    "GBM": SERIES[1],
}


def set_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (9, 4.5),
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "font.size": 10,
    })


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure to reports/figures (git-ignored) at publication resolution."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
