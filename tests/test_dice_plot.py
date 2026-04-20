"""End-to-end smoke tests for the public `dice_plot` API.

Asserts structure: mpl returns `(Figure, Axes)` or `Axes`; plotly returns
a `go.Figure` with shapes. Pixel comparisons are out of scope.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import pydiceplot
from pydiceplot import dice_plot
from pydiceplot.plots.backends._dice_utils import (
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
        pip_colors=colors,
    )
    assert dp.mode == "categorical"
    assert dp.npips == 4
    assert dp.n_x == 5
    assert dp.n_y == 15
    pt = dp.points[0]
    assert len(pt.pip_colors) == 4


def test_preprocess_per_dot_mode():
    rng = np.random.default_rng(0)
    data = get_diceplot_example_data(3)
    data["lfc"] = rng.normal(0, 1, len(data))
    data["nlq"] = rng.uniform(1, 5, len(data))
    dp = preprocess_dice_plot(
        data, "CellType", "Pathway", "PathologyVariable",
        fill="lfc", size="nlq",
    )
    assert dp.mode == "per_dot"
    assert dp.fill_extent is not None
    assert dp.size_extent is not None
    pt = dp.points[0]
    assert len(pt.pip_fills) == 3
    assert len(pt.pip_sizes) == 3


def test_preprocess_fill_palette_discrete_mode():
    data = pd.DataFrame({
        "specimen": ["x", "x", "x", "y"],
        "taxon": ["p", "p", "q", "q"],
        "organ": ["L", "R", "L", "R"],
        "direction": ["Up", "Down", "Up", "Unchanged"],
    })
    palette = {"Up": "#ff0000", "Down": "#0000ff", "Unchanged": "#888888"}
    dp = preprocess_dice_plot(
        data, "specimen", "taxon", "organ",
        fill="direction", fill_palette=palette,
        pips_order=["L", "R"],
    )
    assert dp.mode == "categorical"
    assert dp.npips == 2
    xp = next(p for p in dp.points if (p.x_cat, p.y_cat) == ("x", "p"))
    assert xp.pip_colors == ["#ff0000", "#0000ff"]
    assert dp.pip_colors == palette


def test_preprocess_rejects_mixing_pip_colors_and_fill_palette():
    data = pd.DataFrame({"s": ["x"], "t": ["y"], "o": ["L"], "f": ["Up"]})
    with pytest.raises(ValueError, match="either"):
        preprocess_dice_plot(
            data, "s", "t", "o",
            pip_colors={"L": "#ff0000"},
            fill="f", fill_palette={"Up": "#00ff00"},
        )


def test_preprocess_drops_unknown_categories():
    data = pd.DataFrame({
        "s": ["x"] * 3, "t": ["y"] * 3,
        "o": ["A", "B", "ZZZ"],
    })
    colors = {"A": "#ff0000", "B": "#00ff00"}
    with pytest.warns(UserWarning, match="dropping rows"):
        dp = preprocess_dice_plot(data, "s", "t", "o", pip_colors=colors)
    assert dp.npips == 2


def test_preprocess_rejects_too_many_categories():
    data = pd.DataFrame({
        "s": ["x"] * 10, "t": ["y"] * 10,
        "o": list("ABCDEFGHIJ"),
    })
    with pytest.raises(ValueError, match="must be in 1..9"):
        preprocess_dice_plot(data, "s", "t", "o")


# ── matplotlib smoke ───────────────────────────────────────────────────────

def test_matplotlib_categorical_returns_fig_ax():
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    fig, ax = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors, title="cat",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_matplotlib_per_dot_returns_fig_ax():
    pydiceplot.set_backend("matplotlib")
    rng = np.random.default_rng(1)
    data = get_diceplot_example_data(3)
    data["lfc"] = rng.normal(0, 1.2, len(data))
    data["nlq"] = rng.uniform(0.5, 4, len(data))
    fig, _ = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        fill="lfc", size="nlq",
        fill_label="Log2FC", size_label="-log10(q)", cmap="RdBu_r",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_matplotlib_with_existing_ax_returns_ax_only():
    """When caller passes `ax=`, we return just the Axes (no legend stack)."""
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(3)
    colors = dict(list(get_example_cat_c_colors().items())[:3])
    fig, user_ax = plt.subplots(figsize=(6, 6))
    result = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors, ax=user_ax,
    )
    assert result is user_ax
    assert isinstance(result, plt.Axes)
    plt.close(fig)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_matplotlib_renders_all_dice_sizes(n):
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(n)
    colors = dict(list(get_example_cat_c_colors().items())[:n])
    fig, _ = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors, title=f"n={n}",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_matplotlib_rejects_plotly_kwargs():
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(3)
    colors = dict(list(get_example_cat_c_colors().items())[:3])
    with pytest.raises(TypeError, match="plotly"):
        dice_plot(data, "CellType", "Pathway", "PathologyVariable",
                  pip_colors=colors, width=800)


# ── plotly smoke ───────────────────────────────────────────────────────────

def test_plotly_categorical_returns_figure():
    import plotly.graph_objects as go
    pydiceplot.set_backend("plotly")
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    fig = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors, title="cat",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.shapes) > 0
    assert any(s.type == "circle" for s in fig.layout.shapes)


def test_plotly_per_dot_has_colorbar():
    pydiceplot.set_backend("plotly")
    rng = np.random.default_rng(1)
    data = get_diceplot_example_data(3)
    data["lfc"] = rng.normal(0, 1.2, len(data))
    data["nlq"] = rng.uniform(0.5, 4, len(data))
    fig = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        fill="lfc", size="nlq",
        fill_label="Log2FC", size_label="-log10(q)",
    )
    assert any(
        getattr(t.marker, "colorbar", None) is not None
        and t.marker.colorbar.title.text == "Log2FC"
        for t in fig.data
    )


def test_plotly_rejects_matplotlib_kwargs():
    pydiceplot.set_backend("plotly")
    data = get_diceplot_example_data(3)
    colors = dict(list(get_example_cat_c_colors().items())[:3])
    with pytest.raises(TypeError, match="matplotlib"):
        dice_plot(data, "CellType", "Pathway", "PathologyVariable",
                  pip_colors=colors, figsize=(8, 8))


def test_plotly_composition_on_fresh_fig_sets_axes_and_uses_full_domain():
    """fig=go.Figure() must get a usable layout: category ticks, reversed y,
    locked aspect, and a full [0,1] x-domain (no reserved legend strip)."""
    import plotly.graph_objects as go
    pydiceplot.set_backend("plotly")
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    user_fig = go.Figure()
    result = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors, fig=user_fig,
    )
    assert result is user_fig
    xaxis = result.layout.xaxis
    yaxis = result.layout.yaxis
    n_x = 5   # CellType categories
    n_y = 15  # Pathway categories
    assert list(xaxis.range) == [0.5, n_x + 0.5]
    assert list(yaxis.range) == [n_y + 0.5, 0.5]  # reversed
    assert yaxis.scaleanchor == "x"
    assert yaxis.scaleratio == 1.0
    assert list(xaxis.tickvals) == list(range(1, n_x + 1))
    assert tuple(xaxis.ticktext) == tuple(["Astrocyte", "Endothelial", "Microglia",
                                            "Neuron", "Oligodendrocyte"])
    # No legend strip reserved when composing
    assert list(xaxis.domain) == [0.0, 1.0]
    assert len(result.layout.shapes) > 0


def test_plotly_owns_figure_reserves_legend_domain():
    import plotly.graph_objects as go
    pydiceplot.set_backend("plotly")
    data = get_diceplot_example_data(3)
    colors = dict(list(get_example_cat_c_colors().items())[:3])
    fig = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors,
    )
    assert isinstance(fig, go.Figure)
    assert list(fig.layout.xaxis.domain) == [0.0, 0.72]


def test_plotly_categorical_pip_radius_matches_base_pip_r():
    """Regression: base_pip_r already folds pip_scale — the backend must not
    multiply by pip_scale a second time."""
    import plotly.graph_objects as go
    from pydiceplot.plots.backends._layout import compute_dice_layout

    pydiceplot.set_backend("plotly")
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    pip_scale = 0.7
    tile_size = 0.9
    fig = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors, pip_scale=pip_scale, tile_size=tile_size,
    )
    # Reproduce the layout the backend built
    n_x = 5
    n_y = 15
    layout = compute_dice_layout(
        n_x=n_x, n_y=n_y,
        plot_width=float(n_x), plot_height=float(n_y),
        plot_x0=0.5, plot_y0=0.5,
        tile_frac=tile_size, pip_scale=pip_scale, npips=4,
    )
    expected_r = layout.base_pip_r
    pip_circles = [s for s in fig.layout.shapes
                   if s.type == "circle" and getattr(s, "xref", None) is None]
    assert pip_circles, "expected at least one pip circle"
    sample = pip_circles[0]
    actual_r = (sample.x1 - sample.x0) / 2.0
    assert actual_r == pytest.approx(expected_r, rel=1e-9)


def test_matplotlib_categorical_pip_radius_matches_base_pip_r():
    import matplotlib
    from pydiceplot.plots.backends._layout import compute_dice_layout

    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(4)
    colors = dict(list(get_example_cat_c_colors().items())[:4])
    pip_scale = 0.6
    tile_size = 0.85
    fig, ax = dice_plot(
        data, x="CellType", y="Pathway", pips="PathologyVariable",
        pip_colors=colors, pip_scale=pip_scale, tile_size=tile_size,
    )
    layout = compute_dice_layout(
        n_x=5, n_y=15,
        plot_width=5.0, plot_height=15.0,
        plot_x0=0.5, plot_y0=0.5,
        tile_frac=tile_size, pip_scale=pip_scale, npips=4,
    )
    expected_r = layout.base_pip_r
    circles = [p for p in ax.patches
               if isinstance(p, matplotlib.patches.Circle)]
    assert circles, "expected at least one pip circle"
    assert circles[0].radius == pytest.approx(expected_r, rel=1e-9)
    plt.close(fig)


def test_dice_plot_does_not_accept_npips():
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(3)
    colors = dict(list(get_example_cat_c_colors().items())[:3])
    with pytest.raises(TypeError):
        dice_plot(data, "CellType", "Pathway", "PathologyVariable",
                  pip_colors=colors, npips=3)


def test_dice_plot_does_not_accept_tile_width_or_tile_height():
    pydiceplot.set_backend("matplotlib")
    data = get_diceplot_example_data(3)
    colors = dict(list(get_example_cat_c_colors().items())[:3])
    with pytest.raises(TypeError):
        dice_plot(data, "CellType", "Pathway", "PathologyVariable",
                  pip_colors=colors, tile_width=0.9)
    with pytest.raises(TypeError):
        dice_plot(data, "CellType", "Pathway", "PathologyVariable",
                  pip_colors=colors, tile_height=0.9)
