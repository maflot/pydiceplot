"""Plotly backend for DicePlot / DominoPlot.

Uses a single-axes figure with shapes (tiles, pips, legend boxes) and
annotations (labels). The legend stack is rendered in paper coords on the
right of the plot, matching the matplotlib backend's layout.
"""

from __future__ import annotations

import os
from typing import List, Optional

import plotly.graph_objects as go
import matplotlib
import matplotlib.cm as mpl_cm
import matplotlib.colors as mcolors
import numpy as np

from ._dice_utils import DicePlotData, preprocess_dice_plot
from ._layout import (
    compute_dice_layout,
    dot_grid_positions,
    scaled_pip_radius,
    DiceLayout,
)
from ._domino_utils import preprocess_domino_plot, switch_axes_domino


# ───────────────────────────────────────────────────────────────────────────
# Public entry point
# ───────────────────────────────────────────────────────────────────────────

def plot_dice(
    data,
    cat_a: str,
    cat_b: str,
    cat_c: str,
    cat_c_colors: Optional[dict] = None,
    fill_col: Optional[str] = None,
    size_col: Optional[str] = None,
    cat_a_order=None,
    cat_b_order=None,
    switch_axis: bool = False,
    ndots: Optional[int] = None,
    pip_scale: float = 0.85,
    cell_width: float = 0.85,
    cell_height: float = 0.85,
    grid_lines: bool = False,
    fill_range=None,
    size_range=None,
    color_map: str = "viridis",
    title: Optional[str] = None,
    cat_a_labs: Optional[str] = None,
    cat_b_labs: Optional[str] = None,
    cat_c_labs: Optional[str] = None,
    fill_legend_label: Optional[str] = None,
    size_legend_label: Optional[str] = None,
    position_legend_label: Optional[str] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    max_dice_sides: int = 9,
    group=None, group_colors=None, group_alpha=0.6,
):
    dp = preprocess_dice_plot(
        data, cat_a, cat_b, cat_c,
        cat_c_colors=cat_c_colors, fill_col=fill_col, size_col=size_col,
        cat_a_order=cat_a_order, cat_b_order=cat_b_order,
        max_dice_sides=max_dice_sides,
    )
    if ndots is not None:
        dp.ndots = ndots

    if switch_axis:
        dp.x_categories, dp.y_categories = dp.y_categories, dp.x_categories
        for p in dp.points:
            p.x_cat, p.y_cat = p.y_cat, p.x_cat

    if fill_col and fill_legend_label is None:
        fill_legend_label = fill_col
    if size_col and size_legend_label is None:
        size_legend_label = size_col
    if position_legend_label is None:
        position_legend_label = cat_c_labs or cat_c

    n_x, n_y = dp.n_x, dp.n_y
    if fig_width is None:
        fig_width = max(n_x * 55 + 300, 700)
    if fig_height is None:
        fig_height = max(n_y * 45 + 150, 450)

    fig = go.Figure()

    # Main plot domain — reserve right portion for the legend stack
    fig.update_layout(
        width=fig_width, height=fig_height,
        plot_bgcolor="white", paper_bgcolor="white",
        title=title, showlegend=False,
        margin=dict(l=80, r=40, t=60, b=80),
        xaxis=dict(
            domain=[0.0, 0.72],
            range=[0.5, n_x + 0.5],
            tickmode="array",
            tickvals=list(range(1, n_x + 1)),
            ticktext=dp.x_categories,
            tickangle=-45,
            title_text=cat_a_labs or cat_a,
            showgrid=False, zeroline=False, mirror=False,
            showline=True, linecolor="#666666",
        ),
        yaxis=dict(
            range=[n_y + 0.5, 0.5],
            tickmode="array",
            tickvals=list(range(1, n_y + 1)),
            ticktext=dp.y_categories,
            title_text=cat_b_labs or cat_b,
            showgrid=False, zeroline=False, mirror=False,
            showline=True, linecolor="#666666",
            scaleanchor="x", scaleratio=1.0,
        ),
    )

    _draw_dice_grid(
        fig, dp,
        pip_scale=pip_scale, cell_width=cell_width, cell_height=cell_height,
        grid_lines=grid_lines,
        fill_range=fill_range, size_range=size_range, color_map=color_map,
    )

    _draw_legend_stack(
        fig, dp,
        position_legend_label=position_legend_label,
        fill_legend_label=fill_legend_label,
        size_legend_label=size_legend_label,
        fill_range=fill_range, size_range=size_range, color_map=color_map,
        fig_width=fig_width, fig_height=fig_height,
    )

    return fig


# ───────────────────────────────────────────────────────────────────────────
# Grid rendering
# ───────────────────────────────────────────────────────────────────────────

def _norm(v, vmin, vmax):
    if vmax <= vmin:
        return 0.5
    return max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))


def _draw_dice_grid(
    fig: go.Figure, dp: DicePlotData, *,
    pip_scale: float, cell_width: float, cell_height: float,
    grid_lines: bool, fill_range, size_range, color_map: str,
):
    n_x, n_y = dp.n_x, dp.n_y
    layout = compute_dice_layout(
        n_x=n_x, n_y=n_y,
        plot_width=float(n_x), plot_height=float(n_y),
        plot_x0=0.5, plot_y0=0.5,
        cell_width=cell_width, cell_height=cell_height,
        pip_scale=pip_scale, ndots=max(dp.ndots, 1),
    )

    cmap = matplotlib.colormaps.get_cmap(color_map)
    fmin, fmax = fill_range or dp.fill_extent or (0.0, 1.0)
    smin, smax = size_range or dp.size_extent or (0.0, 1.0)

    x_idx = {c: i for i, c in enumerate(dp.x_categories)}
    y_idx = {c: i for i, c in enumerate(dp.y_categories)}

    shapes: list[dict] = []

    # Tiles and pips
    for pt in dp.points:
        xi = x_idx.get(pt.x_cat)
        yi = y_idx.get(pt.y_cat)
        if xi is None or yi is None:
            continue
        cx, cy = layout.tile_center(xi, yi)
        half = layout.tile_sq / 2.0

        shapes.append(dict(
            type="rect",
            x0=cx - half, x1=cx + half,
            y0=cy - half, y1=cy + half,
            line=dict(color="#888888", width=0.8),
            fillcolor="white",
            layer="below",
        ))
        if grid_lines:
            for i in (1, 2):
                frac = i / 3.0
                shapes.append(dict(
                    type="line",
                    x0=cx - half + frac * layout.tile_sq, x1=cx - half + frac * layout.tile_sq,
                    y0=cy - half, y1=cy + half,
                    line=dict(color="#cccccc", width=0.4),
                    layer="below",
                ))
                shapes.append(dict(
                    type="line",
                    x0=cx - half, x1=cx + half,
                    y0=cy - half + frac * layout.tile_sq, y1=cy - half + frac * layout.tile_sq,
                    line=dict(color="#cccccc", width=0.4),
                    layer="below",
                ))

        # Pips
        for k, (px, py) in enumerate(layout.pip_centers(cx, cy, y_down=True)):
            if dp.mode == "categorical":
                color = pt.dot_colors[k] if k < len(pt.dot_colors) else None
                if color is None:
                    continue
                r = layout.base_pip_r * pip_scale
                shapes.append(dict(
                    type="circle",
                    x0=px - r, x1=px + r, y0=py - r, y1=py + r,
                    fillcolor=color, line=dict(color=color, width=0),
                    layer="above",
                ))
            elif dp.mode == "per_dot":
                fv = pt.dot_fills[k] if k < len(pt.dot_fills) else None
                sv = pt.dot_sizes[k] if k < len(pt.dot_sizes) else None
                if fv is None and sv is None:
                    continue
                r = scaled_pip_radius(layout, sv, smin, smax) if sv is not None else layout.base_pip_r * 0.6
                color = mcolors.to_hex(cmap(_norm(fv, fmin, fmax))) if fv is not None else "#444444"
                shapes.append(dict(
                    type="circle",
                    x0=px - r, x1=px + r, y0=py - r, y1=py + r,
                    fillcolor=color, line=dict(color=color, width=0),
                    layer="above",
                ))

    fig.update_layout(shapes=shapes)


# ───────────────────────────────────────────────────────────────────────────
# Legend stack (paper coords)
# ───────────────────────────────────────────────────────────────────────────

def _draw_legend_stack(
    fig: go.Figure, dp: DicePlotData, *,
    position_legend_label: Optional[str],
    fill_legend_label: Optional[str],
    size_legend_label: Optional[str],
    fill_range, size_range, color_map: str,
    fig_width: float, fig_height: float,
):
    """Paper-coord stack in [0.76 .. 0.99] × [top .. bottom]."""
    x0, x1 = 0.76, 0.99
    cursor = 0.98  # top (paper y=1 is top)

    # Paper coords span [0,1] in both axes but the figure pixel dimensions
    # usually differ, so a circle's y-radius must be stretched by w/h to
    # appear round on screen.
    aspect = fig_width / max(fig_height, 1.0)

    existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    existing_annos = list(fig.layout.annotations) if fig.layout.annotations else []

    if position_legend_label and dp.ndots > 0:
        cursor = _legend_position(existing_shapes, existing_annos, dp,
                                   position_legend_label, x0, x1, cursor, aspect)
        cursor -= 0.02

    if dp.mode == "categorical" and dp.cat_c_colors:
        cursor = _legend_dot_colors(existing_shapes, existing_annos, dp,
                                     x0, x1, cursor, aspect)
        cursor -= 0.02

    if dp.mode == "per_dot" and dp.size_extent is not None and size_legend_label:
        cursor = _legend_size(existing_shapes, existing_annos, dp, size_legend_label,
                               size_range or dp.size_extent, x0, x1, cursor, aspect)
        cursor -= 0.02

    fig.update_layout(shapes=existing_shapes, annotations=existing_annos)

    if dp.mode == "per_dot" and dp.fill_extent is not None and fill_legend_label:
        _add_colorbar(fig, fill_legend_label, fill_range or dp.fill_extent, color_map, cursor)


def _legend_position(shapes, annos, dp: DicePlotData, title: str,
                      x0: float, x1: float, y_top: float, aspect: float) -> float:
    box_pad = 0.01
    title_h = 0.03
    row_h = 0.055
    die_h = 3 * row_h
    total_h = title_h + die_h + 2 * box_pad
    box_bot = y_top - total_h

    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=x0, x1=x1, y0=box_bot, y1=y_top,
        fillcolor="#fafafa", line=dict(color="#cccccc", width=0.8),
    ))
    annos.append(dict(
        x=(x0 + x1) / 2, y=y_top - title_h / 2 - box_pad,
        xref="paper", yref="paper",
        text=f"<b>{title}</b>", showarrow=False, font=dict(size=10),
        xanchor="center", yanchor="middle",
    ))

    die_x0 = x0 + 0.02
    die_x1 = x1 - 0.02
    die_y0 = box_bot + box_pad
    die_y1 = y_top - title_h - box_pad
    die_w = die_x1 - die_x0
    die_hh = die_y1 - die_y0
    cell_w = die_w / 3.0
    cell_h = die_hh / 3.0

    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=die_x0, x1=die_x1, y0=die_y0, y1=die_y1,
        fillcolor="white", line=dict(color="#666666", width=1.0),
    ))
    for i in (1, 2):
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=die_x0 + i * cell_w, x1=die_x0 + i * cell_w,
            y0=die_y0, y1=die_y1,
            line=dict(color="#999999", width=0.5, dash="dot"),
        ))
        shapes.append(dict(
            type="line", xref="paper", yref="paper",
            x0=die_x0, x1=die_x1,
            y0=die_y0 + i * cell_h, y1=die_y0 + i * cell_h,
            line=dict(color="#999999", width=0.5, dash="dot"),
        ))

    # Font scaling
    longest = max((len(l) for l in dp.category_labels), default=0)
    fs = 9
    if dp.ndots >= 5 or longest >= 10:
        fs = 7
    if dp.ndots >= 6 and longest >= 12:
        fs = 6

    # Round pip: r_y = r_x * aspect compensates for figure w/h ratio.
    pip_r_x = 0.008
    pip_r_y = pip_r_x * aspect
    for k, (row, col) in enumerate(dot_grid_positions(dp.ndots)):
        cell_x_left = die_x0 + col * cell_w
        cell_y_top = die_y1 - row * cell_h
        pip_cx = cell_x_left + cell_w / 2
        pip_cy = cell_y_top - cell_h * 0.28
        label_cy = cell_y_top - cell_h * 0.72

        shapes.append(dict(
            type="circle", xref="paper", yref="paper",
            x0=pip_cx - pip_r_x, x1=pip_cx + pip_r_x,
            y0=pip_cy - pip_r_y, y1=pip_cy + pip_r_y,
            fillcolor="#222222", line=dict(color="#222222", width=0),
        ))
        label = dp.category_labels[k] if k < len(dp.category_labels) else ""
        if fs <= 6 and len(label) > 14:
            label = label[:12] + "…"
        annos.append(dict(
            x=pip_cx, y=label_cy, xref="paper", yref="paper",
            text=label, showarrow=False, font=dict(size=fs),
            xanchor="center", yanchor="middle",
        ))

    return box_bot


def _legend_dot_colors(shapes, annos, dp: DicePlotData,
                        x0: float, x1: float, y_top: float, aspect: float) -> float:
    n = len(dp.cat_c_colors)
    box_pad = 0.01
    title_h = 0.025
    row_h = 0.028
    total_h = title_h + n * row_h + 2 * box_pad
    box_bot = y_top - total_h

    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=x0, x1=x1, y0=box_bot, y1=y_top,
        fillcolor="#fafafa", line=dict(color="#cccccc", width=0.8),
    ))
    annos.append(dict(
        x=(x0 + x1) / 2, y=y_top - title_h / 2 - box_pad,
        xref="paper", yref="paper",
        text="<b>Category</b>", showarrow=False, font=dict(size=10),
        xanchor="center", yanchor="middle",
    ))
    r_x = 0.009
    r_y = r_x * aspect
    for i, (label, color) in enumerate(dp.cat_c_colors.items()):
        ry = y_top - title_h - box_pad - (i + 0.5) * row_h
        shapes.append(dict(
            type="circle", xref="paper", yref="paper",
            x0=x0 + 0.02 - r_x, x1=x0 + 0.02 + r_x,
            y0=ry - r_y, y1=ry + r_y,
            fillcolor=color, line=dict(color="#333333", width=0.5),
        ))
        annos.append(dict(
            x=x0 + 0.035, y=ry, xref="paper", yref="paper",
            text=label, showarrow=False, font=dict(size=8),
            xanchor="left", yanchor="middle",
        ))
    return box_bot


def _legend_size(shapes, annos, dp: DicePlotData, title: str, srange,
                  x0: float, x1: float, y_top: float, aspect: float) -> float:
    smin, smax = srange
    pcts = [0.25, 0.5, 1.0]
    n = len(pcts)
    box_pad = 0.01
    title_h = 0.025
    row_h = 0.040
    total_h = title_h + n * row_h + 2 * box_pad
    box_bot = y_top - total_h

    shapes.append(dict(
        type="rect", xref="paper", yref="paper",
        x0=x0, x1=x1, y0=box_bot, y1=y_top,
        fillcolor="#fafafa", line=dict(color="#cccccc", width=0.8),
    ))
    annos.append(dict(
        x=(x0 + x1) / 2, y=y_top - title_h / 2 - box_pad,
        xref="paper", yref="paper",
        text=f"<b>{title}</b>", showarrow=False, font=dict(size=10),
        xanchor="center", yanchor="middle",
    ))
    for i, pct in enumerate(pcts):
        r_x = 0.006 + pct * 0.012
        r_y = r_x * aspect
        ry = y_top - title_h - box_pad - (i + 0.5) * row_h
        shapes.append(dict(
            type="circle", xref="paper", yref="paper",
            x0=x0 + 0.04 - r_x, x1=x0 + 0.04 + r_x,
            y0=ry - r_y, y1=ry + r_y,
            fillcolor="#444444", line=dict(color="#444444", width=0),
        ))
        value = smin + pct * (smax - smin)
        annos.append(dict(
            x=x0 + 0.075, y=ry, xref="paper", yref="paper",
            text=f"{value:.2f}", showarrow=False, font=dict(size=8),
            xanchor="left", yanchor="middle",
        ))
    return box_bot


def _add_colorbar(fig, title: str, frange, color_map: str, cursor: float):
    """Plotly colorbar via an invisible scatter trace with marker.colorbar."""
    fmin, fmax = frange
    # Build a matplotlib→plotly colorscale
    cmap = matplotlib.colormaps.get_cmap(color_map)
    cs = [[i / 10.0, mcolors.to_hex(cmap(i / 10.0))] for i in range(11)]

    # Clamp cursor so the colorbar has at least some height
    cb_len = max(0.15, min(0.30, cursor - 0.08))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False,
        marker=dict(
            colorscale=cs, cmin=fmin, cmax=fmax, color=[fmin],
            colorbar=dict(
                title=dict(text=title, side="right", font=dict(size=10)),
                x=0.90, xanchor="left",
                y=cursor - cb_len / 2,
                yanchor="middle",
                len=cb_len, thickness=12,
                tickfont=dict(size=8),
            ),
        ),
    ))


# ───────────────────────────────────────────────────────────────────────────
# Domino (behaviour preserved)
# ───────────────────────────────────────────────────────────────────────────

def plot_domino(data, gene_list, switch_axis=False, min_dot_size=1, max_dot_size=5,
                spacing_factor=3, var_id="var", feature_col="gene", celltype_col="CellType",
                contrast_col="Contrast", contrast_levels=("Clinical", "Pathological"),
                contrast_labels=("Clinical", "Pathological"), logfc_col="avg_log2FC",
                pval_col="p_val_adj", logfc_limits=(-1.5, 1.5),
                logfc_colors=None, color_scale_name="Log2 Fold Change",
                axis_text_size=8, aspect_ratio=None, base_width=5, base_height=4,
                title=None, xlabel=None, ylabel=None):
    plot_data, aspect_ratio, unique_celltypes, unique_genes, logfc_colors = preprocess_domino_plot(
        data, gene_list, spacing_factor, list(contrast_levels), feature_col, celltype_col,
        contrast_col, var_id, logfc_col, pval_col, logfc_limits, min_dot_size, max_dot_size,
        logfc_colors,
    )
    if switch_axis:
        plot_data = switch_axes_domino(plot_data, "plotly")
        unique_celltypes, gene_list = gene_list, unique_celltypes

    fig = go.Figure()
    unique_pairs = plot_data[[feature_col, celltype_col]].drop_duplicates()
    shapes = []
    for _, row in unique_pairs.iterrows():
        gi = gene_list.index(row[feature_col]) + 1
        ci = unique_celltypes.index(row[celltype_col]) + 1
        y_min, y_max = ci - 0.4, ci + 0.4
        for idx, contrast in enumerate(contrast_levels):
            base_x = (gi - 1) * spacing_factor + (1 if contrast == contrast_levels[0] else 2)
            shapes.append(dict(
                type="rect", x0=base_x - 0.4, x1=base_x + 0.4, y0=y_min, y1=y_max,
                line=dict(color="grey", width=0.5), fillcolor="white", opacity=0.5,
                layer="below",
            ))
    fig.update_layout(shapes=shapes)

    fig.add_trace(go.Scatter(
        x=plot_data["x_pos"], y=plot_data["y_pos"], mode="markers",
        marker=dict(
            size=plot_data["size"], color=plot_data["adj_logfc"],
            colorscale=[[0, logfc_colors["low"]], [0.5, logfc_colors["mid"]], [1, logfc_colors["high"]]],
            cmin=logfc_limits[0], cmax=logfc_limits[1],
            colorbar=dict(title=color_scale_name), line=dict(width=1, color="black"),
        ),
        text=plot_data[var_id], showlegend=False,
    ))
    fig.update_xaxes(
        range=[0, len(gene_list) * spacing_factor + 2],
        tickvals=[(i * spacing_factor) + 1.5 for i in range(len(gene_list))],
        ticktext=gene_list, showgrid=False,
        title_text=xlabel or ("Genes" if not switch_axis else "Cell Types"),
    )
    fig.update_yaxes(
        range=[0, len(unique_celltypes) + 1],
        tickvals=list(range(1, len(unique_celltypes) + 1)),
        ticktext=unique_celltypes, autorange="reversed", showgrid=False,
        title_text=ylabel or ("Cell Types" if not switch_axis else "Genes"),
    )
    fig.update_layout(plot_bgcolor="white", title=title,
                      width=base_width * 100, height=base_height * 100)
    return fig


def show_plot(fig):
    fig.show()


def save_plot(fig, plot_path, output_str, formats):
    os.makedirs(plot_path, exist_ok=True)
    if isinstance(formats, str):
        formats = [formats]
    for fmt in formats:
        fmt = fmt if fmt.startswith(".") else f".{fmt}"
        file_path = os.path.join(plot_path, f"{output_str}{fmt}")
        if fmt.lower() == ".html":
            fig.write_html(file_path)
        else:
            fig.write_image(file_path)
