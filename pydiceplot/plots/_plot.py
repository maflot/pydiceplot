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
from typing import Optional


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
    npips: Optional[int] = None,
    pip_scale: float = 0.85,
    tile_width: float = 0.85,
    tile_height: float = 0.85,
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
    ax=None,                          # matplotlib only
    fig=None,                         # plotly only
    figsize=None,                     # matplotlib only — (width_in, height_in)
    width: Optional[int] = None,      # plotly only — pixels
    height: Optional[int] = None,     # plotly only — pixels
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
    npips : int, optional
        Force a specific pip count (1..9). Default: `len(unique(pips))`.
    pip_scale : float
        Fraction of the sub-cell the pip radius can fill. Default 0.85.
    tile_width, tile_height : float
        Tile size as a fraction of the cell. Default 0.85.
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
        pip_colors=pip_colors, fill=fill, fill_palette=fill_palette, size=size,
        x_order=x_order, y_order=y_order, pips_order=pips_order,
        npips=npips, pip_scale=pip_scale,
        tile_width=tile_width, tile_height=tile_height, grid_lines=grid_lines,
        fill_range=fill_range, size_range=size_range, cmap=cmap,
        title=title, xlabel=xlabel, ylabel=ylabel,
        fill_label=fill_label, size_label=size_label, pips_label=pips_label,
        max_pips=max_pips,
    )
    # Backend-specific plot targets
    if backend.__name__.endswith("_matplotlib_backend"):
        if fig is not None:
            raise TypeError("dice_plot: `fig=` is a plotly-only argument")
        if width is not None or height is not None:
            raise TypeError("dice_plot: `width`/`height` are plotly-only; use `figsize=`")
        kwargs.update(ax=ax, figsize=figsize)
    else:  # plotly
        if ax is not None:
            raise TypeError("dice_plot: `ax=` is a matplotlib-only argument")
        if figsize is not None:
            raise TypeError("dice_plot: `figsize=` is matplotlib-only; use `width`/`height`")
        kwargs.update(fig=fig, width=width, height=height)

    return backend.plot_dice(data, x, y, pips, **kwargs)


def domino_plot(
    data,
    gene_list,
    *,
    switch_axis: bool = False,
    min_dot_size: float = 1,
    max_dot_size: float = 5,
    spacing_factor: float = 3,
    var_id: str = "var",
    feature_col: str = "gene",
    celltype_col: str = "CellType",
    contrast_col: str = "Contrast",
    contrast_levels=("Clinical", "Pathological"),
    contrast_labels=("Clinical", "Pathological"),
    logfc_col: str = "avg_log2FC",
    pval_col: str = "p_val_adj",
    logfc_limits=(-1.5, 1.5),
    logfc_colors=None,
    color_scale_name: str = "Log2 Fold Change",
    axis_text_size: float = 8,
    aspect_ratio=None,
    base_width: float = 5,
    base_height: float = 4,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Legacy domino plot (untouched by the dice_plot rewrite)."""
    if logfc_colors is None:
        logfc_colors = {"low": "blue", "mid": "white", "high": "red"}
    backend = _active_backend()
    return backend.plot_domino(
        data=data, gene_list=gene_list, switch_axis=switch_axis,
        min_dot_size=min_dot_size, max_dot_size=max_dot_size, spacing_factor=spacing_factor,
        var_id=var_id, feature_col=feature_col, celltype_col=celltype_col,
        contrast_col=contrast_col, contrast_levels=contrast_levels, contrast_labels=contrast_labels,
        logfc_col=logfc_col, pval_col=pval_col, logfc_limits=logfc_limits,
        logfc_colors=logfc_colors, color_scale_name=color_scale_name,
        axis_text_size=axis_text_size, aspect_ratio=aspect_ratio,
        base_width=base_width, base_height=base_height,
        title=title, xlabel=xlabel, ylabel=ylabel,
    )
