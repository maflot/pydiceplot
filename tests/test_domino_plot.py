"""Smoke and preprocessing tests for the public `domino_plot` API."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import pydiceplot
from pydiceplot import domino_plot
from pydiceplot.plots.backends._domino_utils import (
    get_domino_example_data,
    preprocess_domino_plot,
    scaled_domino_marker_area,
    scaled_domino_marker_size,
)


def test_preprocess_domino_builds_boxes_and_points():
    data = get_domino_example_data()
    dp = preprocess_domino_plot(
        data,
        "gene", "Cell_Type", "Group",
        features=["GeneA", "GeneB", "GeneC"],
        label="var",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        contrast_labels=["Type 1", "Type 2"],
    )
    assert dp.n_features == 3
    assert dp.n_celltypes == 3
    assert len(dp.boxes) == 18
    assert len(dp.points) == 18
    assert dp.x_ticktext == ["GeneA", "GeneB", "GeneC"]
    assert dp.contrast_labels == ["Type 1", "Type 2"]
    assert dp.fill_extent is not None
    assert dp.size_extent is not None


def test_preprocess_domino_rejects_more_than_two_contrasts():
    data = get_domino_example_data()
    extra = data.iloc[[0]].copy()
    extra["Group"] = "Type3"
    data = pd.concat([data, extra], ignore_index=True)
    with pytest.raises(ValueError, match="exactly two contrasts"):
        preprocess_domino_plot(
            data,
            "gene", "Cell_Type", "Group",
            fill="logFC",
            size="neg_log10_adj_p",
        )


def test_preprocess_domino_switch_axis_swaps_axes():
    data = get_domino_example_data()
    dp = preprocess_domino_plot(
        data,
        "gene", "Cell_Type", "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        switch_axis=True,
    )
    assert dp.x_axis_name == "Cell_Type"
    assert dp.y_axis_name == "gene"
    assert dp.x_ticktext == sorted(data["Cell_Type"].unique().tolist())


def test_scaled_domino_sizes_handle_flat_ranges():
    assert scaled_domino_marker_area(2.0, 2.0, 2.0) > 0
    assert scaled_domino_marker_size(2.0, 2.0, 2.0) > 0


def test_matplotlib_domino_returns_fig_ax():
    pydiceplot.set_backend("matplotlib")
    data = get_domino_example_data()
    fig, ax = domino_plot(
        data,
        "gene", "Cell_Type", "Group",
        features=["GeneA", "GeneB", "GeneC"],
        label="var",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        contrast_labels=["Type 1", "Type 2"],
        fill_label="Log2FC",
        size_label="-log10(adj p)",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_matplotlib_domino_with_existing_ax_returns_ax():
    pydiceplot.set_backend("matplotlib")
    data = get_domino_example_data()
    fig, user_ax = plt.subplots(figsize=(8, 4))
    result = domino_plot(
        data,
        "gene", "Cell_Type", "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        ax=user_ax,
    )
    assert result is user_ax
    plt.close(fig)


def test_matplotlib_domino_rejects_plotly_kwargs():
    pydiceplot.set_backend("matplotlib")
    data = get_domino_example_data()
    with pytest.raises(TypeError, match="plotly"):
        domino_plot(
            data,
            "gene", "Cell_Type", "Group",
            fill="logFC",
            size="neg_log10_adj_p",
            contrast_order=["Type1", "Type2"],
            width=900,
        )


def test_plotly_domino_returns_figure():
    import plotly.graph_objects as go

    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    fig = domino_plot(
        data,
        "gene", "Cell_Type", "Group",
        label="var",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        contrast_labels=["Type 1", "Type 2"],
        fill_label="Log2FC",
        size_label="-log10(adj p)",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) > 0
    assert any(trace.type == "scatter" for trace in fig.data)


def test_plotly_domino_with_existing_fig_reuses_figure():
    import plotly.graph_objects as go

    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    user_fig = go.Figure()
    result = domino_plot(
        data,
        "gene", "Cell_Type", "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        fig=user_fig,
    )
    assert result is user_fig
    assert len(result.data) == 1


def test_plotly_domino_rejects_matplotlib_kwargs():
    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    _, user_ax = plt.subplots(figsize=(8, 4))
    try:
        with pytest.raises(TypeError, match="matplotlib"):
            domino_plot(
                data,
                "gene", "Cell_Type", "Group",
                fill="logFC",
                size="neg_log10_adj_p",
                contrast_order=["Type1", "Type2"],
                ax=user_ax,
            )
    finally:
        plt.close(user_ax.figure)
