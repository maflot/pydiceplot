"""Preprocessing and shared geometry for domino plots.

Domino plots encode one continuous fill and one continuous size value for up
to two contrast slots inside each `(feature, celltype)` tile. This module
builds a backend-agnostic `DominoPlotData` record so both matplotlib and
plotly render from the same validated geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence
import warnings

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


GROUP_SPACING = 3.0
BOX_HALF = 0.4
GROUP_CENTER = 1.5
CONTRAST_BOX_CENTERS = (1.0, 2.0)
CONTRAST_POINT_X_OFFSETS = (-0.2, 0.2)
CONTRAST_POINT_Y_OFFSETS = (0.2, -0.2)
MPL_MARKER_AREA_RANGE = (45.0, 220.0)
PLOTLY_MARKER_SIZE_RANGE = (10.0, 24.0)


@dataclass
class DominoPoint:
    x: float
    y: float
    fill_value: Optional[float]
    size_value: Optional[float]
    feature_value: str
    celltype_value: str
    contrast_value: str
    contrast_label: str
    label_value: Optional[str] = None


@dataclass
class DominoBox:
    x0: float
    x1: float
    y0: float
    y1: float
    feature_value: str
    celltype_value: str
    contrast_value: str
    contrast_label: str


@dataclass
class DominoPlotData:
    points: List[DominoPoint]
    boxes: List[DominoBox]
    x_tickvals: List[float]
    x_ticktext: List[str]
    y_tickvals: List[float]
    y_ticktext: List[str]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_axis_name: str
    y_axis_name: str
    features: List[str]
    celltypes: List[str]
    contrast_order: List[str]
    contrast_labels: List[str]
    fill_extent: Optional[tuple[float, float]] = None
    size_extent: Optional[tuple[float, float]] = None

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_celltypes(self) -> int:
        return len(self.celltypes)


def _sorted_unique(series: pd.Series) -> List[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return sorted(series.dropna().astype(str).unique().tolist())


def _coerce_numeric(series: pd.Series, column: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & numeric.isna()
    if bad.any():
        raise TypeError(
            f"domino_plot: column '{column}' must be numeric; failed on "
            f"{series.loc[bad].iloc[0]!r}"
        )
    return numeric


def _normalize_size(
    value: float | None, vmin: float, vmax: float, low: float, high: float
) -> float:
    if value is None:
        return low
    if vmax <= vmin:
        return (low + high) / 2.0
    norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    return low + (high - low) * norm


def scaled_domino_marker_area(
    value: float | None,
    vmin: float,
    vmax: float,
    min_area: float = MPL_MARKER_AREA_RANGE[0],
    max_area: float = MPL_MARKER_AREA_RANGE[1],
) -> float:
    """Map a size value to a matplotlib scatter area."""
    return _normalize_size(value, vmin, vmax, min_area, max_area)


def scaled_domino_marker_size(
    value: float | None,
    vmin: float,
    vmax: float,
    min_size: float = PLOTLY_MARKER_SIZE_RANGE[0],
    max_size: float = PLOTLY_MARKER_SIZE_RANGE[1],
) -> float:
    """Map a size value to a plotly marker diameter."""
    return _normalize_size(value, vmin, vmax, min_size, max_size)


def domino_plotly_colorscale(cmap: str) -> list[list[float | str]]:
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    return [[i / 10.0, mcolors.to_hex(cmap_obj(i / 10.0))] for i in range(11)]


def preprocess_domino_plot(
    data: pd.DataFrame,
    feature: str,
    celltype: str,
    contrast: str,
    *,
    features: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
    fill: str,
    size: str,
    feature_order: Optional[Sequence[str]] = None,
    celltype_order: Optional[Sequence[str]] = None,
    contrast_order: Optional[Sequence[str]] = None,
    contrast_labels: Optional[Sequence[str]] = None,
    switch_axis: bool = False,
) -> DominoPlotData:
    """Validate and preprocess long-format domino-plot input."""
    missing = [
        c for c in (feature, celltype, contrast, fill, size) if c not in data.columns
    ]
    if missing:
        raise KeyError(f"domino_plot: columns missing from data: {missing}")
    if label is not None and label not in data.columns:
        raise KeyError(f"domino_plot: label '{label}' not in data")

    df = data.copy()

    if feature_order is not None:
        feature_values = [str(v) for v in feature_order]
    elif features is not None:
        feature_values = [str(v) for v in features]
    else:
        feature_values = _sorted_unique(df[feature])
    if not feature_values:
        raise ValueError("domino_plot: feature order is empty")
    allowed_features = set(feature_values)
    dropped_feature_rows = ~df[feature].astype(str).isin(allowed_features)
    if dropped_feature_rows.any():
        warnings.warn(
            "domino_plot: dropping rows with feature values not in the requested "
            f"feature order: {sorted(df.loc[dropped_feature_rows, feature].astype(str).unique().tolist())}"
        )
        df = df.loc[~dropped_feature_rows].copy()
    if df.empty:
        raise ValueError("domino_plot: no rows remain after filtering features")

    if celltype_order is not None:
        celltype_values = [str(v) for v in celltype_order]
    else:
        celltype_values = _sorted_unique(df[celltype])
    if not celltype_values:
        raise ValueError("domino_plot: celltype order is empty")
    allowed_celltypes = set(celltype_values)
    dropped_celltype_rows = ~df[celltype].astype(str).isin(allowed_celltypes)
    if dropped_celltype_rows.any():
        warnings.warn(
            "domino_plot: dropping rows with celltype values not in the requested "
            f"celltype order: {sorted(df.loc[dropped_celltype_rows, celltype].astype(str).unique().tolist())}"
        )
        df = df.loc[~dropped_celltype_rows].copy()
    if df.empty:
        raise ValueError("domino_plot: no rows remain after filtering celltypes")

    if contrast_order is None:
        contrast_values = _sorted_unique(df[contrast])
    else:
        contrast_values = [str(v) for v in contrast_order]
    if len(contrast_values) != 2:
        raise ValueError(
            "domino_plot: domino plots currently require exactly two contrasts; "
            f"got {len(contrast_values)}"
        )
    observed_contrasts = set(df[contrast].astype(str).dropna().unique().tolist())
    extras = sorted(observed_contrasts - set(contrast_values))
    if extras:
        raise ValueError(
            "domino_plot: found contrast values outside `contrast_order`: "
            f"{extras}. Domino plots support exactly the two ordered contrasts."
        )

    if contrast_labels is None:
        contrast_label_values = list(contrast_values)
    else:
        contrast_label_values = [str(v) for v in contrast_labels]
        if len(contrast_label_values) != 2:
            raise ValueError(
                "domino_plot: `contrast_labels` must contain exactly two labels"
            )
    contrast_label_map = dict(zip(contrast_values, contrast_label_values))

    feature_index = {value: i for i, value in enumerate(feature_values)}
    celltype_index = {value: i for i, value in enumerate(celltype_values)}
    contrast_index = {value: i for i, value in enumerate(contrast_values)}

    fill_values = _coerce_numeric(df[fill], fill)
    size_values = _coerce_numeric(df[size], size)

    boxes: List[DominoBox] = []
    for feature_value, celltype_value, contrast_value in product(
        feature_values, celltype_values, contrast_values
    ):
        fi = feature_index[feature_value]
        ci = celltype_index[celltype_value]
        slot = contrast_index[contrast_value]
        x_center = fi * GROUP_SPACING + CONTRAST_BOX_CENTERS[slot]
        y_center = ci + 1.0
        boxes.append(
            DominoBox(
                x0=x_center - BOX_HALF,
                x1=x_center + BOX_HALF,
                y0=y_center - BOX_HALF,
                y1=y_center + BOX_HALF,
                feature_value=feature_value,
                celltype_value=celltype_value,
                contrast_value=contrast_value,
                contrast_label=contrast_label_map[contrast_value],
            )
        )

    points: List[DominoPoint] = []
    for idx, row in df.iterrows():
        feature_value = str(row[feature])
        celltype_value = str(row[celltype])
        contrast_value = str(row[contrast])
        fi = feature_index[feature_value]
        ci = celltype_index[celltype_value]
        slot = contrast_index[contrast_value]
        x = (
            fi * GROUP_SPACING
            + CONTRAST_BOX_CENTERS[slot]
            + CONTRAST_POINT_X_OFFSETS[slot]
        )
        y = ci + 1.0 + CONTRAST_POINT_Y_OFFSETS[slot]
        label_value = None
        if label is not None and pd.notna(row[label]):
            label_value = str(row[label])
        points.append(
            DominoPoint(
                x=x,
                y=y,
                fill_value=float(fill_values.loc[idx])
                if pd.notna(fill_values.loc[idx])
                else None,
                size_value=float(size_values.loc[idx])
                if pd.notna(size_values.loc[idx])
                else None,
                feature_value=feature_value,
                celltype_value=celltype_value,
                contrast_value=contrast_value,
                contrast_label=contrast_label_map[contrast_value],
                label_value=label_value,
            )
        )

    valid_fill_values = [
        point.fill_value for point in points if point.fill_value is not None
    ]
    valid_size_values = [
        point.size_value for point in points if point.size_value is not None
    ]
    fill_extent = None
    size_extent = None
    if valid_fill_values:
        fill_extent = (float(min(valid_fill_values)), float(max(valid_fill_values)))
    if valid_size_values:
        size_extent = (float(min(valid_size_values)), float(max(valid_size_values)))

    dp = DominoPlotData(
        points=points,
        boxes=boxes,
        x_tickvals=[
            i * GROUP_SPACING + GROUP_CENTER for i in range(len(feature_values))
        ],
        x_ticktext=list(feature_values),
        y_tickvals=[i + 1.0 for i in range(len(celltype_values))],
        y_ticktext=list(celltype_values),
        x_range=(
            min(box.x0 for box in boxes) - 0.2,
            max(box.x1 for box in boxes) + 0.2,
        ),
        y_range=(
            min(box.y0 for box in boxes) - 0.1,
            max(box.y1 for box in boxes) + 0.1,
        ),
        x_axis_name=feature,
        y_axis_name=celltype,
        features=list(feature_values),
        celltypes=list(celltype_values),
        contrast_order=list(contrast_values),
        contrast_labels=list(contrast_label_values),
        fill_extent=fill_extent,
        size_extent=size_extent,
    )
    if switch_axis:
        _swap_domino_axes(dp)
    return dp


def _swap_domino_axes(dp: DominoPlotData) -> None:
    for box in dp.boxes:
        box.x0, box.y0 = box.y0, box.x0
        box.x1, box.y1 = box.y1, box.x1
    for point in dp.points:
        point.x, point.y = point.y, point.x
    dp.x_tickvals, dp.y_tickvals = dp.y_tickvals, dp.x_tickvals
    dp.x_ticktext, dp.y_ticktext = dp.y_ticktext, dp.x_ticktext
    dp.x_range, dp.y_range = dp.y_range, dp.x_range
    dp.x_axis_name, dp.y_axis_name = dp.y_axis_name, dp.x_axis_name
    dp.features, dp.celltypes = dp.celltypes, dp.features


# Utility function to generate the example data
def get_domino_example_data() -> pd.DataFrame:
    """Generate example data for the refactored domino plot."""
    gene_list = ["GeneA", "GeneB", "GeneC"]
    cell_types = ["Neuron", "Astrocyte", "Microglia"]
    contrasts = ["Type1", "Type2"]

    data = pd.DataFrame(
        [
            (gene, cell_type, contrast)
            for gene in gene_list
            for cell_type in cell_types
            for contrast in contrasts
        ],
        columns=["gene", "Cell_Type", "Group"],
    )

    vars_type1 = ["MCI-NCI", "AD-MCI", "AD-NCI"]
    vars_type2 = ["Amyloid", "Plaq N", "Tangles", "NFT"]

    rng = np.random.default_rng(123)
    data_type1 = data[data["Group"] == "Type1"].copy()
    data_type1["var"] = rng.choice(vars_type1, size=len(data_type1), replace=True)
    data_type2 = data[data["Group"] == "Type2"].copy()
    data_type2["var"] = rng.choice(vars_type2, size=len(data_type2), replace=True)

    data_combined = pd.concat([data_type1, data_type2], ignore_index=True)
    data_combined["logFC"] = rng.uniform(-2, 2, size=len(data_combined))
    data_combined["adj_p_value"] = rng.uniform(0.0001, 0.05, size=len(data_combined))
    data_combined["neg_log10_adj_p"] = -np.log10(data_combined["adj_p_value"])
    return data_combined
