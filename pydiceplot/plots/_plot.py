"""Public dice / domino plot API.

`dice_plot` is a thin dispatch layer over the active backend — it forwards
every argument to the backend's `plot_dice` and returns whatever that
backend produces natively:

- matplotlib: `(Figure, Axes)` when we create the figure, or just `Axes`
  when the caller passes `ax=`
- plotly: a `plotly.graph_objects.Figure`

Select the backend with `pydiceplot.set_backend("matplotlib" | "plotly")`.
"""

from __future__ import annotations

import importlib
from typing import Optional, Sequence


def _active_backend():
    from pydiceplot._backend import _backend

    module_name = f"pydiceplot.plots.backends._{_backend}_backend"
    return importlib.import_module(module_name)


def dice_plot(
    data,
    x: str,
    y: str,
    pips: str,
    *,
    # pip encoding
    pip_colors: Optional[dict] = None,
    fill: Optional[str] = None,
    fill_palette: Optional[dict] = None,
    size: Optional[str] = None,
    # ordering
    x_order=None,
    y_order=None,
    pips_order=None,
    # dice geometry
    pip_scale: float = 0.85,
    tile_size: float = 0.85,
    grid_lines: bool = False,
    # color scales
    fill_range=None,
    size_range=None,
    cmap: str = "viridis",
    # labels
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fill_label: Optional[str] = None,
    size_label: Optional[str] = None,
    pips_label: Optional[str] = None,
    # plot target — backend-specific
    ax=None,  # matplotlib only
    fig=None,  # plotly only
    figsize=None,  # matplotlib only — (width_in, height_in)
    width: Optional[int] = None,  # plotly only — pixels
    height: Optional[int] = None,  # plotly only — pixels
    max_pips: int = 9,
):
    """Draw a dice plot.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format input. One row per present pip.
    x, y : str
        Column names that map to the x- and y-axis categories.
    pips : str
        Column name selecting which pip slot (1..npips) this row occupies.
    pip_colors : dict, optional
        `{pips value: hex}` — when set, each pip is coloured by its `pips`
        value. Key order sets the pip slot order.
    fill : str, optional
        Column name for per-pip fill. Continuous unless paired with
        `fill_palette`.
    fill_palette : dict, optional
        `{fill value: hex}` — enables discrete per-pip fill. The pip slot
        still comes from `pips`; only the colour comes from `fill`.
    size : str, optional
        Numeric column name for per-pip size.
    x_order, y_order, pips_order : sequence, optional
        Explicit category orderings. Default: sorted unique.
    pip_scale : float
        Fraction of the sub-cell the pip radius can fill. Default 0.85.
    tile_size : float
        Tile side as a fraction of the cell. Default 0.85.
    grid_lines : bool
        Draw a faint 3×3 sub-grid inside each tile.
    fill_range, size_range : tuple, optional
        `(vmin, vmax)` for continuous mappings. Default: data extents.
    cmap : str
        Matplotlib colormap name for continuous fill. Default "viridis".
    title, xlabel, ylabel : str, optional
        Plot labels.
    fill_label, size_label, pips_label : str, optional
        Legend section titles. Default: the corresponding column name.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. Skips the right-side legend stack
        (caller is composing a multi-panel figure). Matplotlib only.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add shapes/traces to. Skips the legend stack.
        Plotly only.
    figsize : tuple, optional
        `(width_in, height_in)` for matplotlib. Ignored on plotly.
    width, height : int, optional
        Pixel dimensions for plotly. Ignored on matplotlib.

    Returns
    -------
    matplotlib: `(Figure, Axes)` or `Axes` (when `ax=` was supplied).
    plotly: `plotly.graph_objects.Figure`.
    """
    backend = _active_backend()
    kwargs = dict(
        pip_colors=pip_colors,
        fill=fill,
        fill_palette=fill_palette,
        size=size,
        x_order=x_order,
        y_order=y_order,
        pips_order=pips_order,
        pip_scale=pip_scale,
        tile_size=tile_size,
        grid_lines=grid_lines,
        fill_range=fill_range,
        size_range=size_range,
        cmap=cmap,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        fill_label=fill_label,
        size_label=size_label,
        pips_label=pips_label,
        max_pips=max_pips,
    )
    # Backend-specific plot targets
    if backend.__name__.endswith("_matplotlib_backend"):
        if fig is not None:
            raise TypeError("dice_plot: `fig=` is a plotly-only argument")
        if width is not None or height is not None:
            raise TypeError(
                "dice_plot: `width`/`height` are plotly-only; use `figsize=`"
            )
        kwargs.update(ax=ax, figsize=figsize)
    else:  # plotly
        if ax is not None:
            raise TypeError("dice_plot: `ax=` is a matplotlib-only argument")
        if figsize is not None:
            raise TypeError(
                "dice_plot: `figsize=` is matplotlib-only; use `width`/`height`"
            )
        kwargs.update(fig=fig, width=width, height=height)

    return backend.plot_dice(data, x, y, pips, **kwargs)


def domino_plot(
    data,
    feature: str,
    celltype: str,
    contrast: str,
    *,
    features: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
    fill: str,
    size: str,
    feature_order=None,
    celltype_order=None,
    contrast_order=None,
    contrast_labels=None,
    switch_axis: bool = False,
    fill_range=None,
    size_range=None,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fill_label: Optional[str] = None,
    size_label: Optional[str] = None,
    ax=None,  # matplotlib only
    fig=None,  # plotly only
    figsize=None,  # matplotlib only — (width_in, height_in)
    width: Optional[int] = None,  # plotly only — pixels
    height: Optional[int] = None,  # plotly only — pixels
):
    """Draw a domino plot.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format input. One row per point.
    feature, celltype, contrast : str
        Column names mapping the feature groups, y-axis categories, and the
        two contrast slots inside each tile.
    features : sequence, optional
        Optional feature filter. When `feature_order` is omitted, this also
        defines the displayed feature order.
    label : str, optional
        Optional label column used for plotly hover text.
    fill, size : str
        Numeric column names for point colour and point size.
    feature_order, celltype_order, contrast_order : sequence, optional
        Explicit orderings. Domino plots currently support exactly two
        contrast slots.
    contrast_labels : sequence, optional
        Human-readable labels for the two contrast slots.
    switch_axis : bool
        Rotate the plot so cell types move to the x-axis and features to the
        y-axis.
    fill_range, size_range : tuple, optional
        `(vmin, vmax)` overrides for the continuous mappings.
    cmap : str
        Matplotlib colormap name used by both backends. Default "RdBu_r".
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. Skips the built-in legend panel.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to draw into. Skips the built-in legend panel.
    figsize : tuple, optional
        `(width_in, height_in)` for matplotlib. Ignored on plotly.
    width, height : int, optional
        Pixel dimensions for plotly. Ignored on matplotlib.

    Returns
    -------
    matplotlib: `(Figure, Axes)` or `Axes` (when `ax=` was supplied).
    plotly: `plotly.graph_objects.Figure`.
    """
    backend = _active_backend()
    kwargs = dict(
        features=features,
        label=label,
        fill=fill,
        size=size,
        feature_order=feature_order,
        celltype_order=celltype_order,
        contrast_order=contrast_order,
        contrast_labels=contrast_labels,
        switch_axis=switch_axis,
        fill_range=fill_range,
        size_range=size_range,
        cmap=cmap,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        fill_label=fill_label,
        size_label=size_label,
    )
    if backend.__name__.endswith("_matplotlib_backend"):
        if fig is not None:
            raise TypeError("domino_plot: `fig=` is a plotly-only argument")
        if width is not None or height is not None:
            raise TypeError(
                "domino_plot: `width`/`height` are plotly-only; use `figsize=`"
            )
        kwargs.update(ax=ax, figsize=figsize)
    else:  # plotly
        if ax is not None:
            raise TypeError("domino_plot: `ax=` is a matplotlib-only argument")
        if figsize is not None:
            raise TypeError(
                "domino_plot: `figsize=` is matplotlib-only; use `width`/`height`"
            )
        kwargs.update(fig=fig, width=width, height=height)
    return backend.plot_domino(data, feature, celltype, contrast, **kwargs)
