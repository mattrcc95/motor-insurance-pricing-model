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


def plot_relativities(model, factors: list[str], ncols: int = 3, title: str = "", name: str | None = None):
    """Plot exp(beta) with 95% CI for each level of the categorical GLM factors.

    The reference level is drawn at 1.0 (hollow marker) - every other level is a
    multiplier relative to it.
    """
    import numpy as np

    rel = model.relativities()
    design = model.design_
    nrows = int(np.ceil(len(factors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), factors):
        levels = design.levels_[f]
        vals, lo, hi = [], [], []
        for lev in levels:
            key = f"{f}[{lev}]"
            if key in rel.index:
                vals.append(rel.loc[key, "relativity"]); lo.append(rel.loc[key, "ci_low"]); hi.append(rel.loc[key, "ci_high"])
            else:
                vals.append(1.0); lo.append(1.0); hi.append(1.0)
        x = np.arange(len(levels))
        vals, lo, hi = map(np.array, (vals, lo, hi))
        ax.axhline(1, color=INK_2, lw=1, ls="--")
        ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="o", color=SERIES[0], ms=5, capsize=2, lw=1.2)
        ref = levels.index(design.reference_[f])
        ax.plot(x[ref], 1, "o", mfc=SURFACE, mec=SERIES[0], ms=7)
        ax.set_xticks(x); ax.set_xticklabels(levels, rotation=45 if len(levels) > 6 else 0, ha="right" if len(levels) > 6 else "center", fontsize=8)
        ax.set_title(f.replace("GLM", ""))
    for ax in axes.ravel()[len(factors):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    if name:
        save(fig, name)
    plt.show()


def plot_lorenz(y, preds: dict, weight, title: str, name: str | None = None, ax=None):
    """Ordered Lorenz curves for several models plus the oracle and random lines."""
    from .metrics import gini_index, lorenz_curve

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, p in preds.items():
        x, c = lorenz_curve(y, p, weight)
        ax.plot(x, c, color=MODEL_COLORS.get(label, None), label=f"{label} (Gini {gini_index(y, p, weight):.3f})")
    x, c = lorenz_curve(y, y, weight)
    ax.plot(x, c, color=INK_2, ls=":", lw=1.2, label=f"Oracle (Gini {gini_index(y, y, weight):.3f})")
    ax.plot([0, 1], [0, 1], color="#a8a7a2", ls="--", lw=1, label="Random (Gini 0)")
    ax.set(xlabel="cumulative share of exposure\n(sorted from safest to riskiest predicted)",
           ylabel="cumulative share of claims", title=title)
    ax.legend(loc="upper left", fontsize=8)
    if own:
        plt.tight_layout()
        if name:
            save(fig, name)
        plt.show()


def plot_lift(tables: dict, title: str, ylabel: str, name: str | None = None):
    """Side-by-side lift charts: observed vs predicted mean per decile of prediction."""
    import numpy as np

    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 4), sharey=True, squeeze=False)
    for ax, (label, t) in zip(axes.ravel(), tables.items()):
        x = np.arange(1, len(t) + 1)
        ax.bar(x, t["observed"], color=GRID, edgecolor=INK_2, lw=0.4, label="Observed")
        ax.plot(x, t["predicted"], color=MODEL_COLORS.get(label, SERIES[0]), marker="o", ms=5, label=f"{label} predicted")
        ax.set(title=label, xlabel="decile of predicted risk (equal exposure)", xticks=x)
        ax.legend(fontsize=8, loc="upper left")
    axes[0, 0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    plt.tight_layout()
    if name:
        save(fig, name)
    plt.show()


def plot_double_lift(t, label_a: str, label_b: str, title: str, name: str | None = None):
    """Double lift chart (relative to the overall observed mean)."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(1, len(t) + 1)
    ax.bar(x, t["observed"], color=GRID, edgecolor=INK_2, lw=0.4, label="Observed")
    ax.plot(x, t["model_a"], color=MODEL_COLORS.get(label_a, SERIES[0]), marker="o", ms=5, label=label_a)
    ax.plot(x, t["model_b"], color=MODEL_COLORS.get(label_b, SERIES[1]), marker="s", ms=5, label=label_b)
    ax.axhline(1, color=INK_2, lw=1, ls="--")
    ax.set(xticks=x, xlabel=f"decile of the ratio {label_a} / {label_b} prediction",
           ylabel="relative to portfolio average", title=title)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    if name:
        save(fig, name)
    plt.show()


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure to reports/figures (git-ignored) at publication resolution."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
