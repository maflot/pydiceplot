"""
Preprocessing for dice plots.

Produces a backend-agnostic `DicePlotData` record that the matplotlib and
plotly backends both consume. Supports three input modes:

- **Categorical**: `pip_colors={label: hex, ...}` supplied → each pip is
  either present (filled in its category colour) or absent.
- **Per-dot continuous**: `fill` / `size` are numeric column names →
  each pip encodes continuous fill and/or size.
- **Per-dot discrete fill**: `fill` + `fill_palette={value: hex, ...}` →
  pip slot is selected by `pips`, pip colour comes from a separate
  categorical column via the palette.
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
    pip_colors: List[Optional[str]] = field(default_factory=list)
    pip_fills:  List[Optional[float]] = field(default_factory=list)
    pip_sizes:  List[Optional[float]] = field(default_factory=list)
    tile_fill:  Optional[float] = None


@dataclass
class DicePlotData:
    points: List[DicePoint]
    x_categories: List[str]
    y_categories: List[str]
    pip_labels: List[str]          # one label per pip slot, in slot order
    pip_colors: Optional[dict] = None  # legend palette (discrete)
    npips: int = 0
    mode: str = "categorical"          # "categorical" | "per_dot"
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
    x: str,
    y: str,
    pips: str,
    *,
    pip_colors: Optional[dict] = None,
    fill: Optional[str] = None,
    fill_palette: Optional[dict] = None,
    size: Optional[str] = None,
    x_order: Optional[Sequence[str]] = None,
    y_order: Optional[Sequence[str]] = None,
    pips_order: Optional[Sequence[str]] = None,
    max_pips: int = 9,
) -> DicePlotData:
    """Turn long-format `data` into a `DicePlotData` ready for rendering.

    Parameters
    ----------
    data : DataFrame
        One row per present pip.
    x, y, pips : column names.
        `x` maps to the plot x-axis category, `y` to the y-axis category,
        `pips` to the pip slot category (1..npips).
    pip_colors : dict {pips value → hex}, optional.
        When set, each pip is coloured by its `pips` value. Key order sets
        the pip slot order.
    fill : str, optional.
        Column name for per-pip fill. Continuous if numeric, discrete if
        paired with `fill_palette`.
    fill_palette : dict {fill value → hex}, optional.
        Discrete colour per `fill` value. Use together with `pips_order` (or a
        factor on `pips`) to control pip slot order.
    size : str, optional.
        Numeric column name for per-pip size.
    """
    missing = [c for c in (x, y, pips) if c not in data.columns]
    if missing:
        raise KeyError(f"dice_plot: columns missing from data: {missing}")

    if fill is not None and fill not in data.columns:
        raise KeyError(f"dice_plot: fill '{fill}' not in data")
    if size is not None and size not in data.columns:
        raise KeyError(f"dice_plot: size '{size}' not in data")

    if pip_colors is not None and fill_palette is not None:
        raise ValueError(
            "dice_plot: pass either `pip_colors` or `fill_palette`, not both"
        )

    discrete_fill = fill_palette is not None
    continuous_per_dot = not discrete_fill and (fill is not None or size is not None)
    mode = "per_dot" if continuous_per_dot else "categorical"

    # ── Category orders ────────────────────────────────────────────────────
    if x_order is None:
        x_order = _sorted_unique(data[x])
    if y_order is None:
        y_order = _sorted_unique(data[y])

    if pip_colors is not None:
        pip_labels = list(pip_colors.keys())
    elif pips_order is not None:
        pip_labels = list(pips_order)
    else:
        pip_labels = _sorted_unique(data[pips])

    npips = len(pip_labels)
    if npips < 1 or npips > max_pips:
        raise ValueError(
            f"dice_plot: number of `pips` categories ({npips}) must be in "
            f"1..{max_pips}"
        )

    if mode == "categorical" and pip_colors is None and fill_palette is None:
        pip_colors = dict(zip(pip_labels, generate_automatic_colors(npips)))

    # ── Build per-tile points ──────────────────────────────────────────────
    slot_index = {label: i for i, label in enumerate(pip_labels)}

    valid = data[pips].isin(pip_labels)
    if not valid.all():
        dropped = data.loc[~valid, pips].unique().tolist()
        import warnings
        warnings.warn(
            f"dice_plot: dropping rows with `pips` values not in pip_labels: {dropped}"
        )
        data = data.loc[valid]

    points_map: dict[tuple, DicePoint] = {}
    for _, row in data.iterrows():
        key = (row[x], row[y])
        pt = points_map.get(key)
        if pt is None:
            pt = DicePoint(
                x_cat=row[x],
                y_cat=row[y],
                pip_colors=[None] * npips,
                pip_fills=[None] * npips,
                pip_sizes=[None] * npips,
            )
            points_map[key] = pt
        slot = slot_index[row[pips]]
        if mode == "categorical":
            if discrete_fill:
                fv = row[fill]
                if pd.notna(fv):
                    pt.pip_colors[slot] = fill_palette.get(fv)
            else:
                pt.pip_colors[slot] = pip_colors[row[pips]]
        else:
            if fill is not None:
                v = row[fill]
                pt.pip_fills[slot] = float(v) if pd.notna(v) else None
            if size is not None:
                v = row[size]
                pt.pip_sizes[slot] = float(v) if pd.notna(v) else None

    points = list(points_map.values())

    fill_extent = None
    size_extent = None
    if mode == "per_dot":
        all_fills = [v for p in points for v in p.pip_fills if v is not None]
        all_sizes = [v for p in points for v in p.pip_sizes if v is not None]
        if all_fills:
            fill_extent = (float(min(all_fills)), float(max(all_fills)))
        if all_sizes:
            size_extent = (float(min(all_sizes)), float(max(all_sizes)))

    legend_colors = pip_colors if pip_colors is not None else fill_palette

    return DicePlotData(
        points=points,
        x_categories=list(x_order),
        y_categories=list(y_order),
        pip_labels=pip_labels,
        pip_colors=legend_colors,
        npips=npips,
        mode=mode,
        fill_extent=fill_extent,
        size_extent=size_extent,
    )


# ── Sample data helpers ────────────────────────────────────────────────────

def generate_automatic_colors(n_colors: int) -> List[str]:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = list(prop_cycle.by_key()["color"])
    if n_colors > len(colors):
        extra = plt.cm.Set3(np.linspace(0, 1, n_colors - len(colors)))
        colors.extend([mcolors.to_hex(c) for c in extra])
    return [mcolors.to_hex(c) for c in colors[:n_colors]]


def get_diceplot_example_data(n: int) -> pd.DataFrame:
    """Toy data for a dice plot with `n` pathology variables (1..9)."""
    if n < 1 or n > 9:
        raise ValueError("n must be between 1 and 9")

    cell_types = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", "Endothelial"]
    pathways = [
        "Apoptosis", "Inflammation", "Metabolism", "Signal Transduction",
        "Synaptic Transmission", "Cell Cycle", "DNA Repair", "Protein Synthesis",
        "Lipid Metabolism", "Neurotransmitter Release", "Oxidative Stress",
        "Energy Production", "Calcium Signaling", "Synaptic Plasticity",
        "Immune Response",
    ]
    pathology_vars = [
        "Alzheimer's disease", "Cancer", "Flu", "ADHD", "Age", "Weight",
        "Diabetes", "Obesity", "Hypertension",
    ][:n]

    rng = np.random.default_rng(123)
    rows = []
    for ct in cell_types:
        for pw in pathways:
            k = int(rng.integers(1, n + 1))
            picks = rng.choice(pathology_vars, size=k, replace=False)
            for var in picks:
                rows.append({"CellType": ct, "Pathway": pw, "PathologyVariable": var})
    return pd.DataFrame(rows)


def get_example_cat_c_colors() -> dict:
    return {
        "Alzheimer's disease": "#1f77b4",
        "Cancer": "#ff7f0e",
        "Flu": "#2ca02c",
        "ADHD": "#d62728",
        "Age": "#9467bd",
        "Weight": "#8c564b",
        "Diabetes": "#e377c2",
        "Obesity": "#17becf",
        "Hypertension": "#bcbd22",
    }


def get_example_group_colors() -> dict:
    return {"Group1": "#1f77b4", "Group2": "#ff7f0e", "Group3": "#2ca02c"}
