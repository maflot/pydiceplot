"""Public dice/domino plot API — thin wrappers that dispatch to the active backend."""

from __future__ import annotations

import importlib
from typing import Optional


class _BackendPlot:
    def __init__(self):
        from pydiceplot._backend import _backend
        module_name = f"pydiceplot.plots.backends._{_backend}_backend"
        self._backend_module = importlib.import_module(module_name)
        self.fig = None

    def _prepare(self, fn_name: str, **kwargs):
        fn = getattr(self._backend_module, fn_name)
        self.fig = fn(**kwargs)

    def show(self):
        self._backend_module.show_plot(self.fig)

    def save(self, plot_path, output_str, formats):
        self._backend_module.save_plot(self.fig, plot_path, output_str, formats)


def dice_plot(
    data,
    cat_a: str,
    cat_b: str,
    cat_c: str,
    *,
    # Mode selection
    cat_c_colors: Optional[dict] = None,
    fill_col: Optional[str] = None,
    size_col: Optional[str] = None,
    # Ordering
    cat_a_order=None,
    cat_b_order=None,
    switch_axis: bool = False,
    # Dice shape
    ndots: Optional[int] = None,
    pip_scale: float = 0.85,
    cell_width: float = 0.85,
    cell_height: float = 0.85,
    grid_lines: bool = False,
    # Color scales
    fill_range=None,
    size_range=None,
    color_map: str = "viridis",
    # Labels
    title: Optional[str] = None,
    cat_a_labs: Optional[str] = None,
    cat_b_labs: Optional[str] = None,
    cat_c_labs: Optional[str] = None,
    fill_legend_label: Optional[str] = None,
    size_legend_label: Optional[str] = None,
    position_legend_label: Optional[str] = None,
    # Dimensions
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    max_dice_sides: int = 6,
    # Legacy (unused, accepted for back-compat)
    group=None,
    group_colors=None,
    group_alpha: float = 0.6,
):
    """Create a dice plot using the active backend.

    Three modes, chosen by the inputs you pass:

    - **Categorical** (default): supply `cat_c_colors={label: hex, ...}`. Each
      pip slot shows a filled circle in its category colour when present.
    - **Per-dot continuous**: pass `fill_col` and/or `size_col` (column names
      in `data`). Each pip encodes continuous fill and/or size, with a matching
      colorbar/size legend.
    - **Tile mode**: not auto-detected in the long-format API; use `fill_col`
      on a single-row-per-tile DataFrame.

    The legend stack (right-hand column) always includes a position legend
    showing which pip slot corresponds to which cat_c label, matching
    ggdiceplot's `draw_key` behavior.
    """
    plot = _BackendPlot()
    plot._prepare(
        "plot_dice",
        data=data, cat_a=cat_a, cat_b=cat_b, cat_c=cat_c,
        cat_c_colors=cat_c_colors, fill_col=fill_col, size_col=size_col,
        cat_a_order=cat_a_order, cat_b_order=cat_b_order, switch_axis=switch_axis,
        ndots=ndots, pip_scale=pip_scale,
        cell_width=cell_width, cell_height=cell_height, grid_lines=grid_lines,
        fill_range=fill_range, size_range=size_range, color_map=color_map,
        title=title, cat_a_labs=cat_a_labs, cat_b_labs=cat_b_labs, cat_c_labs=cat_c_labs,
        fill_legend_label=fill_legend_label, size_legend_label=size_legend_label,
        position_legend_label=position_legend_label,
        fig_width=fig_width, fig_height=fig_height, max_dice_sides=max_dice_sides,
        group=group, group_colors=group_colors, group_alpha=group_alpha,
    )
    return plot


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
    """Create a domino plot (per-row fold-change × p-value, paired contrasts)."""
    if logfc_colors is None:
        logfc_colors = {"low": "blue", "mid": "white", "high": "red"}
    plot = _BackendPlot()
    plot._prepare(
        "plot_domino",
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
    return plot
