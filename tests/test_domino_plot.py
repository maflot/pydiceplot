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
        "gene",
        "Cell_Type",
        "Group",
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
            "gene",
            "Cell_Type",
            "Group",
            fill="logFC",
            size="neg_log10_adj_p",
        )


def test_preprocess_domino_switch_axis_swaps_axes():
    data = get_domino_example_data()
    dp = preprocess_domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
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
        "gene",
        "Cell_Type",
        "Group",
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


def test_matplotlib_domino_legend_has_position_dots_and_labels():
    pydiceplot.set_backend("matplotlib")
    data = get_domino_example_data()
    fig, _ = domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        contrast_labels=["Type 1", "Type 2"],
        fill_label="Log2FC",
        size_label="-log10(adj p)",
    )
    legend_ax = fig.axes[1]
    texts = [text.get_text() for text in legend_ax.texts]
    assert "Contrast" in texts
    assert "Type 1" in texts
    assert "Type 2" in texts
    assert len(legend_ax.collections) >= 2
    assert len(legend_ax.collections[0].get_offsets()) == 2
    assert len(legend_ax.lines) >= 4
    plt.close(fig)


def test_matplotlib_domino_with_existing_ax_returns_ax():
    pydiceplot.set_backend("matplotlib")
    data = get_domino_example_data()
    fig, user_ax = plt.subplots(figsize=(8, 4))
    result = domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
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
            "gene",
            "Cell_Type",
            "Group",
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
        "gene",
        "Cell_Type",
        "Group",
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


def test_plotly_domino_legend_has_position_dots_and_labels():
    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    fig = domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        contrast_labels=["Type 1", "Type 2"],
        fill_label="Log2FC",
        size_label="-log10(adj p)",
    )
    annotations = [annotation.text for annotation in fig.layout.annotations]
    assert "<b>Contrast</b>" in annotations
    assert "Type 1" in annotations
    assert "Type 2" in annotations
    top_legend_circles = [
        shape
        for shape in fig.layout.shapes
        if shape.type == "circle"
        and shape.xref == "paper"
        and shape.yref == "paper"
        and float(shape.y0) > 0.75
    ]
    assert len(top_legend_circles) == 2
    top_legend_lines = [
        shape
        for shape in fig.layout.shapes
        if shape.type == "line"
        and shape.xref == "paper"
        and shape.yref == "paper"
        and float(shape.y0) > 0.70
        and float(shape.y1) > 0.70
    ]
    assert len(top_legend_lines) >= 4


def test_plotly_domino_with_existing_fig_reuses_figure():
    import plotly.graph_objects as go

    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    user_fig = go.Figure()
    result = domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        fig=user_fig,
    )
    assert result is user_fig
    assert len(result.data) == 1


def test_plotly_domino_composition_on_fresh_fig_sets_axes_and_uses_full_domain():
    """Regression: a caller-supplied blank Figure must receive category
    ticks, a reversed y-axis, locked aspect, and a full-width x-domain."""
    import plotly.graph_objects as go

    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    user_fig = go.Figure()
    result = domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        fig=user_fig,
    )
    xaxis = result.layout.xaxis
    yaxis = result.layout.yaxis
    assert xaxis.range is not None and len(xaxis.range) == 2
    # y-axis reversed: first value > second
    assert yaxis.range[0] > yaxis.range[1]
    assert yaxis.scaleanchor == "x"
    assert yaxis.scaleratio == 1.0
    # No reserved legend strip when composing
    assert list(xaxis.domain) == [0.0, 1.0]
    assert xaxis.tickvals is not None
    assert xaxis.ticktext is not None
    # Rectangles for every (feature, celltype, contrast) combination must be present
    assert len(result.layout.shapes) > 0


def test_plotly_domino_owns_figure_reserves_legend_domain():
    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    fig = domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
    )
    assert list(fig.layout.xaxis.domain) == [0.0, 0.74]


def test_plotly_domino_hover_uses_raw_size_from_customdata():
    """Regression: hover must report the underlying size value, not the
    scaled marker diameter."""
    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    fig = domino_plot(
        data,
        "gene",
        "Cell_Type",
        "Group",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
    )
    scatter = next(
        trace
        for trace in fig.data
        if trace.type == "scatter"
        and trace.mode == "markers"
        and trace.customdata is not None
    )
    assert "%{customdata[4]" in scatter.hovertemplate
    assert "%{marker.size" not in scatter.hovertemplate
    # customdata rows: [feature, celltype, contrast_label, label, size_value]
    row = scatter.customdata[0]
    assert len(row) == 5
    # fifth entry should be a raw numeric size (not a display diameter in 10..24 range)
    raw_size = row[4]
    assert raw_size is None or isinstance(raw_size, (int, float))


def test_plotly_domino_rejects_matplotlib_kwargs():
    pydiceplot.set_backend("plotly")
    data = get_domino_example_data()
    _, user_ax = plt.subplots(figsize=(8, 4))
    try:
        with pytest.raises(TypeError, match="matplotlib"):
            domino_plot(
                data,
                "gene",
                "Cell_Type",
                "Group",
                fill="logFC",
                size="neg_log10_adj_p",
                contrast_order=["Type1", "Type2"],
                ax=user_ax,
            )
    finally:
        plt.close(user_ax.figure)
