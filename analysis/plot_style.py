"""Shared paper-facing plotting helpers for CALE figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd


PAPER_DPI = 320

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "cale_seq",
    ["#F8FAFC", "#D9E8F5", "#9CC8DF", "#4D93B7", "#1F5F8B", "#0B304F"],
)

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "cale_div",
    ["#2B6CB0", "#C7DCEF", "#F8FAFC", "#F3C6B8", "#B23A2E"],
)

NEUTRAL_CMAP = LinearSegmentedColormap.from_list(
    "cale_neutral",
    ["#F9FAFB", "#E5E7EB", "#9CA3AF", "#4B5563", "#111827"],
)

CATEGORICAL_CMAP = ListedColormap(
    ["#8DD3C7", "#BEBADA", "#FDB462", "#80B1D3", "#FB8072", "#B3DE69"]
)


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.titleweight": "medium",
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def pretty_label(name: object) -> str:
    return str(name).replace("_proxy", "").replace("_", " ").title()


def _as_labels(labels: Iterable[object], *, pretty: bool) -> list[str]:
    if pretty:
        return [pretty_label(label) for label in labels]
    return [str(label) for label in labels]


def _text_color(value: float, vmin: float, vmax: float, diverging: bool) -> str:
    if not np.isfinite(value):
        return "#6B7280"
    if diverging:
        midpoint = (vmin + vmax) / 2
        strength = abs(value - midpoint) / max((vmax - vmin) / 2, 1e-9)
        return "white" if strength > 0.62 else "#111827"
    return "white" if value > vmin + (vmax - vmin) * 0.58 else "#111827"


def save_paper_heatmap(
    df: pd.DataFrame,
    path: Path,
    title: str,
    cbar_label: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
    diverging: bool = False,
    categorical: bool = False,
    fmt: str = ".2f",
    annotate: bool = True,
    pretty_columns: bool = False,
    pretty_index: bool = False,
    figsize: tuple[float, float] | None = None,
    xrotation: float = 30,
    cbar_ticks: list[float] | None = None,
    cbar_ticklabels: list[str] | None = None,
    annotation_labels: pd.DataFrame | None = None,
) -> None:
    """Save a clean, print-friendly heatmap with consistent CALE styling."""
    if df.empty:
        return
    apply_paper_style()
    path.parent.mkdir(parents=True, exist_ok=True)

    values = df.astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    if diverging:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound

    active_cmap = cmap or (CATEGORICAL_CMAP if categorical else DIVERGING_CMAP if diverging else SEQUENTIAL_CMAP)
    active_cmap = active_cmap.copy()
    active_cmap.set_bad("#F3F4F6")

    if figsize is None:
        fig_width = max(6.8, 0.72 * len(df.columns) + 2.2)
        fig_height = max(3.9, 0.42 * len(df.index) + 1.8)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(values, aspect="auto", cmap=active_cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=10)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(_as_labels(df.columns, pretty=pretty_columns), rotation=xrotation, ha="right", fontsize=8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(_as_labels(df.index, pretty=pretty_index), fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if annotation_labels is not None:
                    label = str(annotation_labels.iloc[i, j])
                else:
                    label = "NA" if np.isnan(value) else format(value, fmt)
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color=_text_color(value, vmin, vmax, diverging),
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=cbar_ticks)
    if cbar_ticklabels is not None:
        cbar.ax.set_yticklabels(cbar_ticklabels)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=PAPER_DPI)
    plt.close(fig)
