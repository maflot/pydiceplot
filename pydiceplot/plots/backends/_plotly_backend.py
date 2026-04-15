"""Plotly backend for dice plots.

Public entry point: `plot_dice(...)`. Returns a `plotly.graph_objects.Figure`.
When `fig=` is supplied we add shapes/traces to it and skip the legend stack
(the caller is composing their own layout).
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from ._dice_utils import DicePlotData, preprocess_dice_plot
from ._layout import (
    DiceLayout,
    compute_dice_layout,
    pip_grid_positions,
    scaled_pip_radius,
)


# ───────────────────────────────────────────────────────────────────────────
# Public entry point
# ───────────────────────────────────────────────────────────────────────────

def plot_dice(
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
    # colour scales
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
    # plot target
    fig: Optional[go.Figure] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_pips: int = 9,
) -> go.Figure:
    dp = preprocess_dice_plot(
        data, x, y, pips,
        pip_colors=pip_colors, fill=fill, fill_palette=fill_palette, size=size,
        x_order=x_order, y_order=y_order, pips_order=pips_order,
        max_pips=max_pips,
    )
    if npips is not None:
        dp.npips = npips

    owns_figure = fig is None
    n_x, n_y = dp.n_x, dp.n_y
    if width is None:
        width = max(n_x * 55 + 300, 700)
    if height is None:
        height = max(n_y * 45 + 150, 450)

    if owns_figure:
        fig = go.Figure()
        fig.update_layout(
            width=width, height=height,
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
                title_text=xlabel or x,
                showgrid=False, zeroline=False, mirror=False,
                showline=True, linecolor="#666666",
            ),
            yaxis=dict(
                range=[n_y + 0.5, 0.5],
                tickmode="array",
                tickvals=list(range(1, n_y + 1)),
                ticktext=dp.y_categories,
                title_text=ylabel or y,
                showgrid=False, zeroline=False, mirror=False,
                showline=True, linecolor="#666666",
                scaleanchor="x", scaleratio=1.0,
            ),
        )

    _draw_dice_grid(
        fig, dp,
        pip_scale=pip_scale, tile_width=tile_width, tile_height=tile_height,
        grid_lines=grid_lines, fill_range=fill_range, size_range=size_range,
        cmap=cmap,
    )

    if owns_figure:
        _draw_legend_stack(
            fig, dp,
            pips_label=pips_label or pips,
            fill_label=fill_label or fill,
            size_label=size_label or size,
            fill_range=fill_range, size_range=size_range, cmap=cmap,
            width=width, height=height,
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
    pip_scale: float, tile_width: float, tile_height: float,
    grid_lines: bool, fill_range, size_range, cmap: str,
):
    n_x, n_y = dp.n_x, dp.n_y
    layout = compute_dice_layout(
        n_x=n_x, n_y=n_y,
        plot_width=float(n_x), plot_height=float(n_y),
        plot_x0=0.5, plot_y0=0.5,
        cell_width=tile_width, cell_height=tile_height,
        pip_scale=pip_scale, npips=max(dp.npips, 1),
    )

    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    fmin, fmax = fill_range or dp.fill_extent or (0.0, 1.0)
    smin, smax = size_range or dp.size_extent or (0.0, 1.0)

    x_idx = {c: i for i, c in enumerate(dp.x_categories)}
    y_idx = {c: i for i, c in enumerate(dp.y_categories)}

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []

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

        for k, (px, py) in enumerate(layout.pip_centers(cx, cy, y_down=True)):
            if dp.mode == "categorical":
                color = pt.pip_colors[k] if k < len(pt.pip_colors) else None
                if color is None:
                    continue
                r = layout.base_pip_r * pip_scale
                shapes.append(dict(
                    type="circle",
                    x0=px - r, x1=px + r, y0=py - r, y1=py + r,
                    fillcolor=color, line=dict(color=color, width=0),
                    layer="above",
                ))
            else:  # per_dot
                fv = pt.pip_fills[k] if k < len(pt.pip_fills) else None
                sv = pt.pip_sizes[k] if k < len(pt.pip_sizes) else None
                if fv is None and sv is None:
                    continue
                if dp.size_extent is not None:
                    r = scaled_pip_radius(layout, sv, smin, smax)
                else:
                    r = layout.base_pip_r
                color = mcolors.to_hex(cmap_obj(_norm(fv, fmin, fmax))) if fv is not None else "#444444"
                shapes.append(dict(
                    type="circle",
                    x0=px - r, x1=px + r, y0=py - r, y1=py + r,
                    fillcolor=color, line=dict(color=color, width=0),
                    layer="above",
                ))

    fig.update_layout(shapes=shapes)


# ───────────────────────────────────────────────────────────────────────────
# Legend stack
# ───────────────────────────────────────────────────────────────────────────

def _draw_legend_stack(
    fig: go.Figure, dp: DicePlotData, *,
    pips_label: Optional[str],
    fill_label: Optional[str],
    size_label: Optional[str],
    fill_range, size_range, cmap: str,
    width: float, height: float,
):
    lx0, lx1 = 0.76, 0.99
    cursor = 0.98
    aspect = width / max(height, 1.0)

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    annos = list(fig.layout.annotations) if fig.layout.annotations else []

    if pips_label and dp.npips > 0:
        cursor = _legend_position(shapes, annos, dp, pips_label, lx0, lx1, cursor, aspect)
        cursor -= 0.02

    if dp.mode == "categorical" and dp.pip_colors:
        cursor = _legend_pip_colors(shapes, annos, dp, lx0, lx1, cursor, aspect)
        cursor -= 0.02

    if dp.mode == "per_dot" and dp.size_extent is not None and size_label:
        cursor = _legend_size(shapes, annos, dp, size_label,
                               size_range or dp.size_extent, lx0, lx1, cursor, aspect)
        cursor -= 0.02

    fig.update_layout(shapes=shapes, annotations=annos)

    if dp.mode == "per_dot" and dp.fill_extent is not None and fill_label:
        _add_colorbar(fig, fill_label, fill_range or dp.fill_extent, cmap, cursor)


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

    longest = max((len(l) for l in dp.pip_labels), default=0)
    fs = 9
    if dp.npips >= 5 or longest >= 10:
        fs = 7
    if dp.npips >= 6 and longest >= 12:
        fs = 6

    pip_r_x = 0.008
    pip_r_y = pip_r_x * aspect
    for k, (row, col) in enumerate(pip_grid_positions(dp.npips)):
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
        label = dp.pip_labels[k] if k < len(dp.pip_labels) else ""
        if fs <= 6 and len(label) > 14:
            label = label[:12] + "…"
        annos.append(dict(
            x=pip_cx, y=label_cy, xref="paper", yref="paper",
            text=label, showarrow=False, font=dict(size=fs),
            xanchor="center", yanchor="middle",
        ))

    return box_bot


def _legend_pip_colors(shapes, annos, dp: DicePlotData,
                        x0: float, x1: float, y_top: float, aspect: float) -> float:
    n = len(dp.pip_colors)
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
    for i, (label, color) in enumerate(dp.pip_colors.items()):
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
    box_pad = 0.01
    title_h = 0.025
    row_h = 0.040
    total_h = title_h + len(pcts) * row_h + 2 * box_pad
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


def _add_colorbar(fig: go.Figure, title: str, frange, cmap: str, cursor: float):
    fmin, fmax = frange
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    cs = [[i / 10.0, mcolors.to_hex(cmap_obj(i / 10.0))] for i in range(11)]

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
# Legacy shims
# ───────────────────────────────────────────────────────────────────────────

from ._domino_utils import preprocess_domino_plot, switch_axes_domino  # noqa: E402


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


def save_plot(fig, plot_path, output_str, formats):
    """Deprecated shim — use `fig.write_image(...)` or `fig.write_html(...)`."""
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


def show_plot(fig):
    """Deprecated shim — use `fig.show()`."""
    fig.show()
