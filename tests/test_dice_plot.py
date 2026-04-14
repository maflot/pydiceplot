"""End-to-end smoke tests for the public `dice_plot` API.

We render against both backends and assert that the figure was constructed
(non-empty shapes/patches, correct number of categories). Pixel comparisons
are out of scope; backend-native primitives differ.
"""

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pydiceplot
from pydiceplot import dice_plot
from pydiceplot.plots.backends._dice_utils import (
    DicePlotData,
    get_diceplot_example_data,
    get_example_cat_c_colors,
    preprocess_dice_plot,
)


# ── preprocess ─────────────────────────────────────────────────────────────

def test_preprocess_categorical_mode():
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    dp = preprocess_dice_plot(
        data, "CellType", "Pathway", "PathologyVariable",
        cat_c_colors=colors,
    )
    assert dp.mode == "categorical"
    assert dp.ndots == 4
    assert dp.n_x == 5
    assert dp.n_y == 15
    # Each point carries a 4-slot color vector
    pt = dp.points[0]
    assert len(pt.dot_colors) == 4
    assert all(c is None or isinstance(c, str) for c in pt.dot_colors)


def test_preprocess_per_dot_mode():
    rng = np.random.default_rng(0)
    data = get_diceplot_example_data(3)
    data["lfc"] = rng.normal(0, 1, len(data))
    data["nlq"] = rng.uniform(1, 5, len(data))
    dp = preprocess_dice_plot(
        data, "CellType", "Pathway", "PathologyVariable",
        fill_col="lfc", size_col="nlq",
    )
    assert dp.mode == "per_dot"
    assert dp.ndots == 3
    assert dp.fill_extent is not None
    assert dp.size_extent is not None
    fmin, fmax = dp.fill_extent
    assert fmin < fmax
    # Per-pip arrays sized to ndots
    pt = dp.points[0]
    assert len(pt.dot_fills) == 3
    assert len(pt.dot_sizes) == 3


def test_preprocess_fill_palette_discrete_mode():
    data = pd.DataFrame({
        "a": ["x", "x", "x", "y"],
        "b": ["p", "p", "q", "q"],
        "c": ["L", "R", "L", "R"],
        "direction": ["Up", "Down", "Up", "Unchanged"],
    })
    palette = {"Up": "#ff0000", "Down": "#0000ff", "Unchanged": "#888888"}
    dp = preprocess_dice_plot(
        data, "a", "b", "c",
        fill_col="direction", fill_palette=palette,
        cat_c_order=["L", "R"],
    )
    assert dp.mode == "categorical"
    assert dp.ndots == 2
    # Point (x,p): L→Up=#ff0000, R→Down=#0000ff
    xp = next(p for p in dp.points if (p.x_cat, p.y_cat) == ("x", "p"))
    assert xp.dot_colors == ["#ff0000", "#0000ff"]
    # The legend colors should come from the palette, not cat_c_colors
    assert dp.cat_c_colors == palette


def test_preprocess_rejects_mixing_cat_c_colors_and_fill_palette():
    data = pd.DataFrame({"a": ["x"], "b": ["y"], "c": ["L"], "f": ["Up"]})
    with pytest.raises(ValueError, match="either"):
        preprocess_dice_plot(
            data, "a", "b", "c",
            cat_c_colors={"L": "#ff0000"},
            fill_col="f", fill_palette={"Up": "#00ff00"},
        )


def test_preprocess_drops_unknown_categories():
    data = pd.DataFrame({
        "a": ["x", "x", "x"],
        "b": ["y", "y", "y"],
        "c": ["A", "B", "ZZZ"],  # ZZZ not in colors
    })
    colors = {"A": "#ff0000", "B": "#00ff00"}
    with pytest.warns(UserWarning, match="dropping rows"):
        dp = preprocess_dice_plot(data, "a", "b", "c", cat_c_colors=colors)
    assert dp.ndots == 2


def test_preprocess_rejects_too_many_categories():
    data = pd.DataFrame({
        "a": ["x"] * 10, "b": ["y"] * 10,
        "c": list("ABCDEFGHIJ"),
    })
    with pytest.raises(ValueError, match="must be in 1..9"):
        preprocess_dice_plot(data, "a", "b", "c")


# ── matplotlib smoke ───────────────────────────────────────────────────────

def test_matplotlib_categorical_renders(tmp_path):
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    fig = dice_plot(
        data=data, cat_a="CellType", cat_b="Pathway", cat_c="PathologyVariable",
        cat_c_colors=colors, title="cat",
    )
    fig.save(str(tmp_path), "out", ".png")
    assert (tmp_path / "out.png").exists()
    assert (tmp_path / "out.png").stat().st_size > 1000


def test_matplotlib_per_dot_renders(tmp_path):
    pydiceplot.set_backend("matplotlib")
    rng = np.random.default_rng(1)
    data = get_diceplot_example_data(3)
    data["lfc"] = rng.normal(0, 1.2, len(data))
    data["nlq"] = rng.uniform(0.5, 4, len(data))
    fig = dice_plot(
        data=data, cat_a="CellType", cat_b="Pathway", cat_c="PathologyVariable",
        fill_col="lfc", size_col="nlq",
        fill_legend_label="Log2FC", size_legend_label="-log10(q)",
        color_map="RdBu_r",
    )
    fig.save(str(tmp_path), "out", ".png")
    assert (tmp_path / "out.png").exists()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_matplotlib_renders_all_dice_sizes(n, tmp_path):
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(n)
    colors = dict(list(get_example_cat_c_colors().items())[:n])
    fig = dice_plot(
        data=data, cat_a="CellType", cat_b="Pathway", cat_c="PathologyVariable",
        cat_c_colors=colors, title=f"n={n}",
    )
    fig.save(str(tmp_path), f"out_{n}", ".png")
    assert (tmp_path / f"out_{n}.png").exists()


# ── plotly smoke ───────────────────────────────────────────────────────────

def test_plotly_categorical_renders(tmp_path):
    pydiceplot.set_backend("plotly")
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    fig = dice_plot(
        data=data, cat_a="CellType", cat_b="Pathway", cat_c="PathologyVariable",
        cat_c_colors=colors, title="cat",
    )
    # We have a Figure with shapes — assert structure rather than exporting
    plotly_fig = fig.fig
    assert len(plotly_fig.layout.shapes) > 0
    assert any(s.type == "circle" for s in plotly_fig.layout.shapes)


def test_plotly_per_dot_has_colorbar(tmp_path):
    pydiceplot.set_backend("plotly")
    rng = np.random.default_rng(1)
    data = get_diceplot_example_data(3)
    data["lfc"] = rng.normal(0, 1.2, len(data))
    data["nlq"] = rng.uniform(0.5, 4, len(data))
    fig = dice_plot(
        data=data, cat_a="CellType", cat_b="Pathway", cat_c="PathologyVariable",
        fill_col="lfc", size_col="nlq",
        fill_legend_label="Log2FC", size_legend_label="-log10(q)",
    )
    plotly_fig = fig.fig
    # The colorbar carrier trace exists
    assert any(
        getattr(t.marker, "colorbar", None) is not None and t.marker.colorbar.title.text == "Log2FC"
        for t in plotly_fig.data
    )
