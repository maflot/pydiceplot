"""
Preprocessing for dice plots.

Produces a backend-agnostic `DicePlotData` record that the matplotlib and
plotly backends both consume. Supports three mutually-exclusive input modes
(matching kuva's `DicePlot::with_records`, `with_dot_data`, `with_points`):

- **Categorical**: `cat_c_colors` supplied → each pip is either present with
  a fixed color or absent. One row per (x, y, cat_c) in `data`.
- **Per-dot continuous**: `fill_col` and/or `size_col` given → each pip
  encodes continuous fill and/or size. One row per (x, y, cat_c).
- **Tile-level** (less common): a single continuous value per (x, y) tile
  with no per-pip encoding. Not auto-detected; request via `tile_fill_col`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DicePoint:
    x_cat: str
    y_cat: str
    dot_colors: List[Optional[str]] = field(default_factory=list)   # categorical
    dot_fills:  List[Optional[float]] = field(default_factory=list) # per-pip continuous
    dot_sizes:  List[Optional[float]] = field(default_factory=list) # per-pip continuous
    tile_fill:  Optional[float] = None                              # tile-level


@dataclass
class DicePlotData:
    points: List[DicePoint]
    x_categories: List[str]
    y_categories: List[str]
    category_labels: List[str]   # cat_c labels (pip slot labels)
    cat_c_colors: Optional[dict] = None
    ndots: int = 0
    mode: str = "categorical"    # "categorical" | "per_dot" | "tile"
    fill_extent: Optional[tuple] = None
    size_extent: Optional[tuple] = None

    @property
    def n_x(self) -> int: return len(self.x_categories)

    @property
    def n_y(self) -> int: return len(self.y_categories)


def _sorted_unique(series: pd.Series) -> List[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return sorted(series.dropna().unique().tolist())


def preprocess_dice_plot(
    data: pd.DataFrame,
    cat_a: str,
    cat_b: str,
    cat_c: str,
    cat_c_colors: Optional[dict] = None,
    fill_col: Optional[str] = None,
    size_col: Optional[str] = None,
    cat_a_order: Optional[Sequence[str]] = None,
    cat_b_order: Optional[Sequence[str]] = None,
    max_dice_sides: int = 6,
) -> DicePlotData:
    """Turn long-format `data` into a `DicePlotData` ready for rendering.

    Parameters
    ----------
    data : DataFrame with columns `cat_a`, `cat_b`, `cat_c` (+ optional
        `fill_col`, `size_col`). One row per present pip slot.
    cat_c_colors : dict mapping cat_c label → hex color. If provided, the
        plot is categorical (pips coloured by `cat_c`). The dict key order
        determines the pip slot order.
    fill_col, size_col : column names for continuous per-pip encoding.
        Providing either switches the plot to per-dot mode.
    """
    missing = [c for c in (cat_a, cat_b, cat_c) if c not in data.columns]
    if missing:
        raise KeyError(f"dice_plot: columns missing from data: {missing}")

    if fill_col is not None and fill_col not in data.columns:
        raise KeyError(f"dice_plot: fill_col '{fill_col}' not in data")
    if size_col is not None and size_col not in data.columns:
        raise KeyError(f"dice_plot: size_col '{size_col}' not in data")

    per_dot_mode = fill_col is not None or size_col is not None
    mode = "per_dot" if per_dot_mode else "categorical"

    # ── Determine category orders ──────────────────────────────────────────
    if cat_a_order is None:
        cat_a_order = _sorted_unique(data[cat_a])
    if cat_b_order is None:
        cat_b_order = _sorted_unique(data[cat_b])

    if cat_c_colors is not None:
        category_labels = list(cat_c_colors.keys())
    else:
        category_labels = _sorted_unique(data[cat_c])

    ndots = len(category_labels)
    if ndots < 1 or ndots > max_dice_sides:
        raise ValueError(
            f"dice_plot: number of cat_c categories ({ndots}) must be in "
            f"1..{max_dice_sides}"
        )

    # Auto-generate categorical colors when not supplied (and in categorical mode)
    if mode == "categorical" and cat_c_colors is None:
        cat_c_colors = dict(zip(category_labels, generate_automatic_colors(ndots)))

    # ── Build per-tile DicePoint records ───────────────────────────────────
    slot_index = {label: i for i, label in enumerate(category_labels)}

    # Warn (don't crash) on rows whose cat_c isn't in the slot list
    valid = data[cat_c].isin(category_labels)
    if not valid.all():
        dropped = data.loc[~valid, cat_c].unique().tolist()
        import warnings
        warnings.warn(
            f"dice_plot: dropping rows with cat_c values not in category_labels: {dropped}"
        )
        data = data.loc[valid]

    # Aggregate rows into points keyed by (x_cat, y_cat)
    points_map: dict[tuple, DicePoint] = {}
    for _, row in data.iterrows():
        key = (row[cat_a], row[cat_b])
        pt = points_map.get(key)
        if pt is None:
            pt = DicePoint(
                x_cat=row[cat_a],
                y_cat=row[cat_b],
                dot_colors=[None] * ndots,
                dot_fills=[None] * ndots,
                dot_sizes=[None] * ndots,
            )
            points_map[key] = pt
        slot = slot_index[row[cat_c]]
        if mode == "categorical":
            pt.dot_colors[slot] = cat_c_colors[row[cat_c]]
        else:
            if fill_col is not None:
                v = row[fill_col]
                pt.dot_fills[slot] = float(v) if pd.notna(v) else None
            if size_col is not None:
                v = row[size_col]
                pt.dot_sizes[slot] = float(v) if pd.notna(v) else None

    points = list(points_map.values())

    # ── Compute extents for continuous encodings ──────────────────────────
    fill_extent = None
    size_extent = None
    if mode == "per_dot":
        all_fills = [v for p in points for v in p.dot_fills if v is not None]
        all_sizes = [v for p in points for v in p.dot_sizes if v is not None]
        if all_fills:
            fill_extent = (float(min(all_fills)), float(max(all_fills)))
        if all_sizes:
            size_extent = (float(min(all_sizes)), float(max(all_sizes)))

    return DicePlotData(
        points=points,
        x_categories=list(cat_a_order),
        y_categories=list(cat_b_order),
        category_labels=category_labels,
        cat_c_colors=cat_c_colors,
        ndots=ndots,
        mode=mode,
        fill_extent=fill_extent,
        size_extent=size_extent,
    )


# ── Sample data helpers (preserved from old module) ────────────────────────

def generate_automatic_colors(n_colors: int) -> List[str]:
    """Generate `n_colors` distinct hex colors."""
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = list(prop_cycle.by_key()["color"])
    if n_colors > len(colors):
        extra = plt.cm.Set3(np.linspace(0, 1, n_colors - len(colors)))
        colors.extend([mcolors.to_hex(c) for c in extra])
    return [mcolors.to_hex(c) for c in colors[:n_colors]]


def get_diceplot_example_data(n: int) -> pd.DataFrame:
    """Toy data for a dice plot with `n` pathology variables (1..6)."""
    if n < 1 or n > 6:
        raise ValueError("n must be between 1 and 6")

    cell_types = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", "Endothelial"]
    pathways = [
        "Apoptosis", "Inflammation", "Metabolism", "Signal Transduction",
        "Synaptic Transmission", "Cell Cycle", "DNA Repair", "Protein Synthesis",
        "Lipid Metabolism", "Neurotransmitter Release", "Oxidative Stress",
        "Energy Production", "Calcium Signaling", "Synaptic Plasticity",
        "Immune Response",
    ]
    pathology_vars = ["Alzheimer's disease", "Cancer", "Flu", "ADHD", "Age", "Weight"][:n]

    rng = np.random.default_rng(123)
    rows = []
    for ct in cell_types:
        for pw in pathways:
            k = int(rng.integers(1, n + 1))
            picks = rng.choice(pathology_vars, size=k, replace=False)
            for var in picks:
                rows.append({"CellType": ct, "Pathway": pw, "PathologyVariable": var})
    return pd.DataFrame(rows)


def get_example_group_colors() -> dict:
    return {"Group1": "#1f77b4", "Group2": "#ff7f0e", "Group3": "#2ca02c"}


def get_example_cat_c_colors() -> dict:
    return {
        "Alzheimer's disease": "#1f77b4",
        "Cancer": "#ff7f0e",
        "Flu": "#2ca02c",
        "ADHD": "#d62728",
        "Age": "#9467bd",
        "Weight": "#8c564b",
    }
