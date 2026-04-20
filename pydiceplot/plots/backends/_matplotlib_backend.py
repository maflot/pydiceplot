"""Matplotlib backend for dice plots.

Public entry point: `plot_dice(...)`. Returns `(Figure, Axes)` when we own
the figure, or just `Axes` when the caller provided `ax=`. In the `ax=` path
we skip the right-side legend stack — the caller is composing their own
layout, so a second axes would fight with their figure.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm

from ._dice_utils import DicePlotData, preprocess_dice_plot
from ._layout import (
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
    pip_scale: float = 0.85,
    tile_size: float = 0.85,
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
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    max_pips: int = 9,
) -> Union[Tuple[plt.Figure, plt.Axes], plt.Axes]:
    """Render a dice plot onto a matplotlib Axes.

    Returns `(Figure, Axes)` when we create the figure, or just `Axes` when
    the caller passes `ax=`.
    """
    dp = preprocess_dice_plot(
        data,
        x,
        y,
        pips,
        pip_colors=pip_colors,
        fill=fill,
        fill_palette=fill_palette,
        size=size,
        x_order=x_order,
        y_order=y_order,
        pips_order=pips_order,
        max_pips=max_pips,
    )

    owns_figure = ax is None
    if owns_figure:
        n_x, n_y = dp.n_x, dp.n_y
        if figsize is None:
            base = 0.6
            figsize = (max(n_x * base + 4.0, 6.0), max(n_y * base + 2.0, 4.0))
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.1], wspace=0.05)
        ax = fig.add_subplot(gs[0, 0])
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.set_axis_off()
    else:
        fig = ax.figure
        legend_ax = None

    _draw_dice_grid(
        ax,
        dp,
        pip_scale=pip_scale,
        tile_size=tile_size,
        grid_lines=grid_lines,
        fill_range=fill_range,
        size_range=size_range,
        cmap=cmap,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif owns_figure:
        ax.set_xlabel(x)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif owns_figure:
        ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)

    if legend_ax is not None:
        _draw_legend_stack(
            legend_ax,
            dp,
            pips_label=pips_label or pips,
            fill_label=fill_label or fill,
            size_label=size_label or size,
            fill_range=fill_range,
            size_range=size_range,
            cmap=cmap,
            fig=fig,
        )

    return (fig, ax) if owns_figure else ax


# ───────────────────────────────────────────────────────────────────────────
# Grid rendering
# ───────────────────────────────────────────────────────────────────────────


def _draw_dice_grid(
    ax: plt.Axes,
    dp: DicePlotData,
    *,
    pip_scale: float,
    tile_size: float,
    grid_lines: bool,
    fill_range,
    size_range,
    cmap: str,
):
    n_x, n_y = dp.n_x, dp.n_y

    ax.set_xlim(0.5, n_x + 0.5)
    ax.set_ylim(0.5, n_y + 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    layout = compute_dice_layout(
        n_x=n_x,
        n_y=n_y,
        plot_width=float(n_x),
        plot_height=float(n_y),
        plot_x0=0.5,
        plot_y0=0.5,
        tile_frac=tile_size,
        pip_scale=pip_scale,
        npips=max(dp.npips, 1),
    )

    ax.set_xticks([i + 1 for i in range(n_x)])
    ax.set_xticklabels(dp.x_categories, rotation=45, ha="right")
    ax.set_yticks([i + 1 for i in range(n_y)])
    ax.set_yticklabels(dp.y_categories)
    ax.tick_params(axis="both", length=3, pad=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    fmin, fmax = fill_range or dp.fill_extent or (0.0, 1.0)
    smin, smax = size_range or dp.size_extent or (0.0, 1.0)

    x_idx = {c: i for i, c in enumerate(dp.x_categories)}
    y_idx = {c: i for i, c in enumerate(dp.y_categories)}

    for pt in dp.points:
        xi = x_idx.get(pt.x_cat)
        yi = y_idx.get(pt.y_cat)
        if xi is None or yi is None:
            continue
        cx, cy = layout.tile_center(xi, yi)
        half = layout.tile_sq / 2.0

        ax.add_patch(
            patches.Rectangle(
                (cx - half, cy - half),
                layout.tile_sq,
                layout.tile_sq,
                linewidth=0.6,
                edgecolor="#888888",
                facecolor="white",
            )
        )

        if grid_lines:
            for i in (1, 2):
                frac = i / 3.0
                ax.plot(
                    [cx - half + frac * layout.tile_sq] * 2,
                    [cy - half, cy + half],
                    color="#cccccc",
                    linewidth=0.4,
                    zorder=1,
                )
                ax.plot(
                    [cx - half, cx + half],
                    [cy - half + frac * layout.tile_sq] * 2,
                    color="#cccccc",
                    linewidth=0.4,
                    zorder=1,
                )

        for k, (px, py) in enumerate(layout.pip_centers(cx, cy, y_down=True)):
            if dp.mode == "categorical":
                color = pt.pip_colors[k] if k < len(pt.pip_colors) else None
                if color is None:
                    continue
                r = layout.base_pip_r
                ax.add_patch(
                    patches.Circle(
                        (px, py), r, facecolor=color, edgecolor="none", zorder=3
                    )
                )
            else:  # per_dot
                fv = pt.pip_fills[k] if k < len(pt.pip_fills) else None
                sv = pt.pip_sizes[k] if k < len(pt.pip_sizes) else None
                if fv is None and sv is None:
                    continue
                if dp.size_extent is not None:
                    r = scaled_pip_radius(layout, sv, smin, smax)
                else:
                    r = layout.base_pip_r
                color = (
                    mcolors.to_hex(cmap_obj(_norm(fv, fmin, fmax)))
                    if fv is not None
                    else "#444444"
                )
                ax.add_patch(
                    patches.Circle(
                        (px, py), r, facecolor=color, edgecolor="none", zorder=3
                    )
                )


def _norm(v, vmin, vmax):
    if vmax <= vmin:
        return 0.5
    return max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))


# ───────────────────────────────────────────────────────────────────────────
# Legend stack
# ───────────────────────────────────────────────────────────────────────────


def _draw_legend_stack(
    lax: plt.Axes,
    dp: DicePlotData,
    *,
    pips_label: Optional[str],
    fill_label: Optional[str],
    size_label: Optional[str],
    fill_range,
    size_range,
    cmap: str,
    fig: plt.Figure,
):
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)

    cursor = 0.98

    if pips_label and dp.npips > 0:
        cursor = _legend_position(lax, dp, pips_label, cursor)
        cursor -= 0.03

    if dp.mode == "categorical" and dp.pip_colors:
        cursor = _legend_pip_colors(lax, dp, cursor)
        cursor -= 0.03

    if dp.mode == "per_dot" and dp.size_extent is not None and size_label:
        cursor = _legend_size(lax, dp, size_label, size_range or dp.size_extent, cursor)
        cursor -= 0.03

    if dp.mode == "per_dot" and dp.fill_extent is not None and fill_label:
        _legend_colorbar(
            lax,
            dp,
            fill_label,
            fill_range or dp.fill_extent,
            cmap,
            fig=fig,
            cursor=cursor,
        )


def _legend_position(lax, dp: DicePlotData, title: str, y_top: float) -> float:
    box_pad = 0.015
    x0, x1 = 0.02, 0.98
    title_h = 0.028
    row_h = 0.055
    die_h = 3 * row_h
    total_h = title_h + die_h + 2 * box_pad

    box_top = y_top
    box_bot = y_top - total_h
    lax.add_patch(
        patches.Rectangle(
            (x0, box_bot),
            x1 - x0,
            total_h,
            facecolor="#fafafa",
            edgecolor="#cccccc",
            linewidth=0.6,
            transform=lax.transAxes,
            clip_on=False,
        )
    )
    lax.text(
        (x0 + x1) / 2,
        y_top - title_h / 2 - box_pad,
        title,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=9,
        transform=lax.transAxes,
    )

    die_x0 = x0 + 0.05
    die_x1 = x1 - 0.05
    die_y0 = box_bot + box_pad
    die_y1 = box_top - title_h - box_pad
    die_w = die_x1 - die_x0
    die_hh = die_y1 - die_y0
    cell_w = die_w / 3.0
    cell_h = die_hh / 3.0

    lax.add_patch(
        patches.Rectangle(
            (die_x0, die_y0),
            die_w,
            die_hh,
            facecolor="white",
            edgecolor="#666666",
            linewidth=0.8,
            transform=lax.transAxes,
            clip_on=False,
        )
    )
    for i in (1, 2):
        lax.plot(
            [die_x0 + i * cell_w, die_x0 + i * cell_w],
            [die_y0, die_y1],
            color="#999999",
            linewidth=0.4,
            linestyle=(0, (2, 2)),
            transform=lax.transAxes,
            clip_on=False,
        )
        lax.plot(
            [die_x0, die_x1],
            [die_y0 + i * cell_h, die_y0 + i * cell_h],
            color="#999999",
            linewidth=0.4,
            linestyle=(0, (2, 2)),
            transform=lax.transAxes,
            clip_on=False,
        )

    longest = max((len(label) for label in dp.pip_labels), default=0)
    label_fs = 8
    if dp.npips >= 5 or longest >= 10:
        label_fs = 6
    if dp.npips >= 6 and longest >= 12:
        label_fs = 5

    pip_xs: list[float] = []
    pip_ys: list[float] = []
    for k, (row, col) in enumerate(pip_grid_positions(dp.npips)):
        cell_x0 = die_x0 + col * cell_w
        cell_y1 = die_y1 - row * cell_h
        pip_cx = cell_x0 + cell_w / 2
        pip_cy = cell_y1 - cell_h * 0.28
        label_cy = cell_y1 - cell_h * 0.72
        pip_xs.append(pip_cx)
        pip_ys.append(pip_cy)

        label = dp.pip_labels[k] if k < len(dp.pip_labels) else ""
        if label_fs <= 5 and len(label) > 14:
            label = label[:12] + "…"
        lax.text(
            pip_cx,
            label_cy,
            label,
            ha="center",
            va="center",
            fontsize=label_fs,
            transform=lax.transAxes,
            clip_on=False,
        )

    lax.scatter(
        pip_xs,
        pip_ys,
        s=28,
        c="#222222",
        marker="o",
        transform=lax.transAxes,
        clip_on=False,
        zorder=4,
    )

    return box_bot


def _legend_pip_colors(lax, dp: DicePlotData, y_top: float) -> float:
    n = len(dp.pip_colors)
    row_h = 0.035
    box_pad = 0.02
    title_h = 0.03
    total_h = title_h + n * row_h + 2 * box_pad
    x0, x1 = 0.05, 0.95
    box_bot = y_top - total_h
    lax.add_patch(
        patches.Rectangle(
            (x0, box_bot),
            x1 - x0,
            total_h,
            facecolor="#fafafa",
            edgecolor="#cccccc",
            linewidth=0.6,
            transform=lax.transAxes,
            clip_on=False,
        )
    )
    lax.text(
        (x0 + x1) / 2,
        y_top - title_h / 2 - box_pad / 2,
        "Category",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        transform=lax.transAxes,
    )
    xs, ys, cs = [], [], []
    for i, (label, color) in enumerate(dp.pip_colors.items()):
        ry = y_top - title_h - box_pad - (i + 0.5) * row_h
        xs.append(x0 + 0.08)
        ys.append(ry)
        cs.append(color)
        lax.text(
            x0 + 0.14,
            ry,
            label,
            ha="left",
            va="center",
            fontsize=8,
            transform=lax.transAxes,
        )
    lax.scatter(
        xs,
        ys,
        s=45,
        c=cs,
        edgecolors="#333333",
        linewidths=0.5,
        transform=lax.transAxes,
        clip_on=False,
        zorder=4,
    )
    return box_bot


def _legend_size(lax, dp: DicePlotData, title: str, srange, y_top: float) -> float:
    smin, smax = srange
    pcts = [0.25, 0.5, 1.0]
    row_h = 0.045
    box_pad = 0.02
    title_h = 0.03
    total_h = title_h + len(pcts) * row_h + 2 * box_pad
    x0, x1 = 0.05, 0.95
    box_bot = y_top - total_h
    lax.add_patch(
        patches.Rectangle(
            (x0, box_bot),
            x1 - x0,
            total_h,
            facecolor="#fafafa",
            edgecolor="#cccccc",
            linewidth=0.6,
            transform=lax.transAxes,
            clip_on=False,
        )
    )
    lax.text(
        (x0 + x1) / 2,
        y_top - title_h / 2 - box_pad / 2,
        title,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        transform=lax.transAxes,
    )
    xs, ys, sizes = [], [], []
    for i, pct in enumerate(pcts):
        ry = y_top - title_h - box_pad - (i + 0.5) * row_h
        xs.append(x0 + 0.12)
        ys.append(ry)
        sizes.append(20 + pct * 90)
        value = smin + pct * (smax - smin)
        lax.text(
            x0 + 0.22,
            ry,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            transform=lax.transAxes,
        )
    lax.scatter(
        xs, ys, s=sizes, c="#444444", transform=lax.transAxes, clip_on=False, zorder=4
    )
    return box_bot


def _legend_colorbar(
    lax: plt.Axes,
    dp: DicePlotData,
    title: str,
    frange,
    cmap: str,
    fig: plt.Figure,
    cursor: float,
):
    fmin, fmax = frange
    sm = mpl_cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=fmin, vmax=fmax),
        cmap=matplotlib.colormaps.get_cmap(cmap),
    )
    sm.set_array([])
    lax_bbox = lax.get_position()
    cb_width = 0.018
    cb_height = min(0.35, max(0.15, cursor * lax_bbox.height - 0.02))
    cb_x = lax_bbox.x0 + (lax_bbox.width - cb_width) / 2
    cb_y = lax_bbox.y0 + 0.05
    cax = fig.add_axes([cb_x, cb_y, cb_width, cb_height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(title, fontsize=9)
    cb.ax.tick_params(labelsize=8)


# ───────────────────────────────────────────────────────────────────────────
# Domino plot
# ───────────────────────────────────────────────────────────────────────────

from ._domino_utils import (  # noqa: E402
    DominoPlotData,
    preprocess_domino_plot,
    scaled_domino_marker_area,
)


def plot_domino(
    data,
    feature: str,
    celltype: str,
    contrast: str,
    *,
    features=None,
    label=None,
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
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Union[Tuple[plt.Figure, plt.Axes], plt.Axes]:
    dp = preprocess_domino_plot(
        data,
        feature,
        celltype,
        contrast,
        features=features,
        label=label,
        fill=fill,
        size=size,
        feature_order=feature_order,
        celltype_order=celltype_order,
        contrast_order=contrast_order,
        contrast_labels=contrast_labels,
        switch_axis=switch_axis,
    )

    owns_figure = ax is None
    if owns_figure:
        if figsize is None:
            figsize = (
                max(dp.n_features * 1.15 + 4.5, 7.5),
                max(dp.n_celltypes * 0.55 + 2.5, 4.8),
            )
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0], wspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.set_axis_off()
    else:
        fig = ax.figure
        legend_ax = None

    _draw_domino_grid(
        ax,
        dp,
        fill_range=fill_range,
        size_range=size_range,
        cmap=cmap,
    )

    ax.set_xlabel(xlabel or dp.x_axis_name)
    ax.set_ylabel(ylabel or dp.y_axis_name)
    if title is not None:
        ax.set_title(title)

    if legend_ax is not None:
        _draw_domino_legend_stack(
            legend_ax,
            dp,
            size_label=size_label or size,
            fill_label=fill_label or fill,
            size_range=size_range,
            fill_range=fill_range,
            cmap=cmap,
            fig=fig,
        )

    return (fig, ax) if owns_figure else ax


def _draw_domino_grid(
    ax: plt.Axes,
    dp: DominoPlotData,
    *,
    fill_range,
    size_range,
    cmap: str,
):
    ax.set_xlim(*dp.x_range)
    ax.set_ylim(*dp.y_range)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(dp.x_tickvals)
    ax.set_xticklabels(dp.x_ticktext, rotation=45, ha="right")
    ax.set_yticks(dp.y_tickvals)
    ax.set_yticklabels(dp.y_ticktext)
    ax.tick_params(axis="both", length=3, pad=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for box in dp.boxes:
        ax.add_patch(
            patches.Rectangle(
                (box.x0, box.y0),
                box.x1 - box.x0,
                box.y1 - box.y0,
                linewidth=0.6,
                edgecolor="#888888",
                facecolor="white",
            )
        )

    points = [
        point
        for point in dp.points
        if point.fill_value is not None or point.size_value is not None
    ]
    if not points:
        return

    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    fmin, fmax = fill_range or dp.fill_extent or (0.0, 1.0)
    smin, smax = size_range or dp.size_extent or (0.0, 1.0)

    ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        s=[scaled_domino_marker_area(point.size_value, smin, smax) for point in points],
        c=[
            mcolors.to_hex(cmap_obj(_norm(point.fill_value, fmin, fmax)))
            if point.fill_value is not None
            else "#666666"
            for point in points
        ],
        edgecolors="#111111",
        linewidths=0.7,
        zorder=3,
    )


def _draw_domino_legend_stack(
    lax: plt.Axes,
    dp: DominoPlotData,
    *,
    size_label: Optional[str],
    fill_label: Optional[str],
    size_range,
    fill_range,
    cmap: str,
    fig: plt.Figure,
):
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    cursor = 0.98

    cursor = _legend_domino_contrasts(lax, dp.contrast_labels, cursor)
    cursor -= 0.03

    if dp.size_extent is not None and size_label:
        cursor = _legend_domino_size(
            lax, size_label, size_range or dp.size_extent, cursor
        )
        cursor -= 0.03

    if dp.fill_extent is not None and fill_label:
        _legend_domino_colorbar(
            lax,
            fill_label,
            fill_range or dp.fill_extent,
            cmap=cmap,
            fig=fig,
            cursor=cursor,
        )


def _legend_domino_contrasts(lax: plt.Axes, labels: List[str], y_top: float) -> float:
    legend_dp = SimpleNamespace(npips=len(labels), pip_labels=list(labels))
    return _legend_position(lax, legend_dp, "Contrast", y_top)


def _legend_domino_size(lax: plt.Axes, title: str, srange, y_top: float) -> float:
    smin, smax = srange
    levels = [smin] if smax <= smin else [smin, (smin + smax) / 2.0, smax]
    row_h = 0.045
    box_pad = 0.02
    title_h = 0.03
    total_h = title_h + len(levels) * row_h + 2 * box_pad
    x0, x1 = 0.05, 0.95
    box_bot = y_top - total_h

    lax.add_patch(
        patches.Rectangle(
            (x0, box_bot),
            x1 - x0,
            total_h,
            facecolor="#fafafa",
            edgecolor="#cccccc",
            linewidth=0.6,
            transform=lax.transAxes,
            clip_on=False,
        )
    )
    lax.text(
        (x0 + x1) / 2,
        y_top - title_h / 2 - box_pad / 2,
        title,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=10,
        transform=lax.transAxes,
    )

    xs, ys, sizes = [], [], []
    for i, level in enumerate(levels):
        ry = y_top - title_h - box_pad - (i + 0.5) * row_h
        xs.append(x0 + 0.12)
        ys.append(ry)
        sizes.append(scaled_domino_marker_area(level, smin, smax))
        lax.text(
            x0 + 0.25,
            ry,
            f"{level:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            transform=lax.transAxes,
        )
    lax.scatter(
        xs,
        ys,
        s=sizes,
        c="#444444",
        transform=lax.transAxes,
        clip_on=False,
        zorder=4,
    )
    return box_bot


def _legend_domino_colorbar(
    lax: plt.Axes,
    title: str,
    frange,
    *,
    cmap: str,
    fig: plt.Figure,
    cursor: float,
):
    fmin, fmax = frange
    sm = mpl_cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=fmin, vmax=fmax),
        cmap=matplotlib.colormaps.get_cmap(cmap),
    )
    sm.set_array([])
    lax_bbox = lax.get_position()
    cb_width = 0.018
    cb_height = min(0.32, max(0.14, cursor * lax_bbox.height - 0.03))
    cb_x = lax_bbox.x0 + (lax_bbox.width - cb_width) / 2
    cb_y = lax_bbox.y0 + 0.05
    cax = fig.add_axes([cb_x, cb_y, cb_width, cb_height])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(title, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def save_plot(fig, plot_path, output_str, formats):
    """Deprecated shim — use `fig.savefig(...)` directly."""
    os.makedirs(plot_path, exist_ok=True)
    if isinstance(formats, str):
        formats = [formats]
    for fmt in formats:
        fmt = fmt if fmt.startswith(".") else f".{fmt}"
        file_path = os.path.join(plot_path, f"{output_str}{fmt}")
        fig.savefig(file_path, format=fmt.strip("."), bbox_inches="tight", dpi=150)


def show_plot(fig):
    """Deprecated shim — use `plt.show()`."""
    plt.show()
