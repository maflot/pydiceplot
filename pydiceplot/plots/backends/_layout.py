"""
Dice grid layout — port of kuva's `src/plot/diceplot.rs` + `add_diceplot` geometry.

All plot backends share this module so that matplotlib and plotly produce identical
pip positions, tile sizes, and grid alignment.

The 3×3 pip sub-grid uses row-major indexing (1..9, left→right, top→bottom):

    1 2 3
    4 5 6
    7 8 9

The dice-face lookup table matches ggdiceplot's `make_offsets()` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


DICE_POSITIONS: dict[int, List[int]] = {
    1: [5],
    2: [1, 9],
    3: [1, 5, 9],
    4: [1, 3, 7, 9],              # four corners
    5: [1, 3, 5, 7, 9],            # corners + centre
    6: [1, 3, 4, 6, 7, 9],         # traditional die: two vertical columns
    7: [1, 3, 4, 5, 6, 7, 9],       # 6 + centre
    8: [1, 2, 3, 4, 6, 7, 8, 9],    # 3×3 minus centre
    9: [1, 2, 3, 4, 5, 6, 7, 8, 9], # fully populated 3×3
}

# Position→(row, col) mapping uses natural reading order:
#
#     col=0 col=1 col=2
#       1     2     3     row=0 (top)
#       4     5     6     row=1 (middle)
#       7     8     9     row=2 (bottom)
#
# So `row = (p-1) // 3` and `col = (p-1) % 3`. This gives traditional
# die faces for every n in 1..9.


def pip_grid_positions(npips: int) -> List[Tuple[int, int]]:
    """Return `(grid_row, grid_col)` for each pip, with row 0 = top, col 0 = left."""
    positions = DICE_POSITIONS.get(npips, [])
    return [((p - 1) // 3, (p - 1) % 3) for p in positions]


def pip_offsets(
    npips: int,
    cell_width: float = 0.8,
    cell_height: float = 0.8,
    pad: float = 0.1,
) -> List[Tuple[float, float]]:
    """Tile-centre-relative `(dx, dy)` for each pip, in tile-width data units.

    `dy` is positive toward the bottom (y-down SVG convention). Matplotlib
    backends that use `invert_yaxis()` get the correct visual layout directly;
    ggplot2/plotly y-up backends should negate `dy`.
    """
    avail_w = cell_width - 2.0 * pad
    avail_h = cell_height - 2.0 * pad
    out: List[Tuple[float, float]] = []
    for (row, col) in pip_grid_positions(npips):
        dx = col / 2.0 * avail_w + pad - cell_width / 2.0
        dy = row / 2.0 * avail_h + pad - cell_height / 2.0
        out.append((dx, dy))
    return out


@dataclass
class DiceLayout:
    """Computed geometry for one dice plot panel.

    All coordinates are in the target backend's "plot area" units. Caller is
    responsible for mapping those to physical pixels (matplotlib: data units;
    plotly: axis data units).
    """

    n_x: int
    n_y: int
    cell_sq: float          # square cell side (unit of the category grid)
    tile_sq: float          # square tile inside a cell (tile = cell * min(cw, ch))
    sub: float              # pip sub-cell side (tile_sq / 3)
    grid_x0: float          # x of left grid edge
    grid_y0: float          # y of top grid edge
    grid_w: float           # total grid width  (n_x * cell_sq)
    grid_h: float           # total grid height (n_y * cell_sq)
    base_pip_r: float       # max pip radius (sub/2 * pip_scale)
    pip_scale: float
    npips: int
    cell_width: float
    cell_height: float

    def tile_center(self, xi: int, yi: int) -> Tuple[float, float]:
        cx = self.grid_x0 + (xi + 0.5) * self.cell_sq
        cy = self.grid_y0 + (yi + 0.5) * self.cell_sq
        return cx, cy

    def pip_centers(self, cx: float, cy: float, y_down: bool = True) -> List[Tuple[float, float]]:
        """Absolute pip centers around tile center `(cx, cy)`.

        `y_down=True` (default) places row 0 above `cy` (matches SVG/matplotlib
        with an inverted y-axis). `y_down=False` mirrors for y-up coord systems.
        """
        out: List[Tuple[float, float]] = []
        for (row, col) in pip_grid_positions(self.npips):
            dx = (col - 1) * self.sub
            dy = (row - 1) * self.sub
            if not y_down:
                dy = -dy
            out.append((cx + dx, cy + dy))
        return out


def compute_dice_layout(
    n_x: int,
    n_y: int,
    plot_width: float,
    plot_height: float,
    plot_x0: float = 0.0,
    plot_y0: float = 0.0,
    cell_width: float = 0.8,
    cell_height: float = 0.8,
    pip_scale: float = 0.85,
    npips: int = 1,
) -> DiceLayout:
    """Compute a centred square-cell dice layout, mirroring kuva::add_diceplot.

    Picks `cell_sq = min(plot_w/n_x, plot_h/n_y)` so both axes use the same
    cell size, then centres the grid within the plot area.
    """
    if n_x <= 0 or n_y <= 0:
        raise ValueError(f"n_x and n_y must be positive, got ({n_x}, {n_y})")
    if npips < 1 or npips > 9:
        raise ValueError(f"npips must be in 1..9, got {npips}")

    cw = plot_width / n_x
    ch = plot_height / n_y
    cell_sq = min(cw, ch)
    tile_sq = cell_sq * min(cell_width, cell_height)

    grid_w = n_x * cell_sq
    grid_h = n_y * cell_sq
    grid_x0 = plot_x0 + (plot_width - grid_w) / 2.0
    grid_y0 = plot_y0 + (plot_height - grid_h) / 2.0

    sub = tile_sq / 3.0
    base_pip_r = sub * 0.5 * pip_scale

    return DiceLayout(
        n_x=n_x,
        n_y=n_y,
        cell_sq=cell_sq,
        tile_sq=tile_sq,
        sub=sub,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        grid_w=grid_w,
        grid_h=grid_h,
        base_pip_r=base_pip_r,
        pip_scale=pip_scale,
        npips=npips,
        cell_width=cell_width,
        cell_height=cell_height,
    )


def scaled_pip_radius(
    layout: DiceLayout,
    value: float | None,
    vmin: float,
    vmax: float,
    min_fill: float = 0.25,
) -> float:
    """Map a continuous size value to a pip radius, matching kuva's
    `0.25 + 0.75 * norm` rule (when pip_scale = 1.0 max)."""
    if value is None:
        return layout.base_pip_r * min_fill
    span = max(vmax - vmin, 1e-12)
    norm = max(0.0, min(1.0, (value - vmin) / span))
    return layout.base_pip_r * (min_fill + (1.0 - min_fill) * norm)
