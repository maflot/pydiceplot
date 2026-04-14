"""Geometry sanity for the dice grid layout.

The (x, y) offsets for each face must match `ggdiceplot::make_offsets()` when
the y-axis is flipped (we store y-down, ggdiceplot uses y-up). These values
were captured from the live R package at cell_width=cell_height=1.0, pad=0.1.
"""

import pytest

from pydiceplot.plots.backends._layout import (
    DICE_POSITIONS,
    compute_dice_layout,
    dot_grid_positions,
    dot_offsets,
    scaled_pip_radius,
)


# (x, y_up) from ggdiceplot
GGDICEPLOT_OFFSETS = {
    1: [(0.00, 0.00)],
    2: [(-0.40, 0.40), (0.40, -0.40)],
    3: [(-0.40, 0.40), (0.00, 0.00), (0.40, -0.40)],
    4: [(-0.40, 0.40), (0.40, 0.40), (-0.40, -0.40), (0.40, -0.40)],
    5: [(-0.40, 0.40), (0.40, 0.40), (0.00, 0.00), (-0.40, -0.40), (0.40, -0.40)],
    6: [(-0.40, 0.40), (0.00, 0.40), (0.40, 0.40),
        (-0.40, -0.40), (0.00, -0.40), (0.40, -0.40)],
}


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_dot_offsets_match_ggdiceplot(n):
    offs = dot_offsets(n, cell_width=1.0, cell_height=1.0, pad=0.1)
    expected = GGDICEPLOT_OFFSETS[n]
    assert len(offs) == len(expected)
    for (dx, dy_down), (ex, ey_up) in zip(offs, expected):
        # We use y-down; ggdiceplot uses y-up. Negate dy for comparison.
        assert dx == pytest.approx(ex, abs=1e-9)
        assert -dy_down == pytest.approx(ey_up, abs=1e-9)


def test_dice_positions_table():
    assert DICE_POSITIONS[1] == [5]
    assert DICE_POSITIONS[2] == [1, 9]
    assert DICE_POSITIONS[3] == [1, 5, 9]
    assert DICE_POSITIONS[4] == [1, 7, 3, 9]
    assert DICE_POSITIONS[5] == [1, 7, 5, 3, 9]
    assert DICE_POSITIONS[6] == [1, 4, 7, 3, 6, 9]


def test_grid_positions_are_row_col():
    """`dot_grid_positions(4)` → TL, TR, BL, BR as (row, col)."""
    grid = dot_grid_positions(4)
    assert grid == [(0, 0), (0, 2), (2, 0), (2, 2)]


def test_compute_dice_layout_centres_square_grid():
    lay = compute_dice_layout(
        n_x=4, n_y=2,
        plot_width=100.0, plot_height=100.0,
        cell_width=0.8, cell_height=0.8, ndots=4,
    )
    # Square cell = min(100/4, 100/2) = 25
    assert lay.cell_sq == pytest.approx(25.0)
    # Tile = cell * 0.8 = 20
    assert lay.tile_sq == pytest.approx(20.0)
    # Grid width = 4 * 25 = 100, fills x fully
    assert lay.grid_w == pytest.approx(100.0)
    # Grid height = 2 * 25 = 50, centred in 100
    assert lay.grid_h == pytest.approx(50.0)
    assert lay.grid_y0 == pytest.approx(25.0)


def test_compute_dice_layout_rejects_bad_inputs():
    with pytest.raises(ValueError):
        compute_dice_layout(n_x=0, n_y=2, plot_width=100, plot_height=100, ndots=4)
    with pytest.raises(ValueError):
        compute_dice_layout(n_x=2, n_y=2, plot_width=100, plot_height=100, ndots=7)


def test_scaled_pip_radius_clamps_and_maps():
    lay = compute_dice_layout(
        n_x=2, n_y=2, plot_width=60, plot_height=60,
        cell_width=0.8, cell_height=0.8, ndots=4,
    )
    base = lay.base_pip_r
    # None → min_fill (0.25)
    assert scaled_pip_radius(lay, None, 0.0, 10.0) == pytest.approx(base * 0.25)
    # Max value → 1.0
    assert scaled_pip_radius(lay, 10.0, 0.0, 10.0) == pytest.approx(base)
    # Below range is clamped
    assert scaled_pip_radius(lay, -5.0, 0.0, 10.0) == pytest.approx(base * 0.25)
