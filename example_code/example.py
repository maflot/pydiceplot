"""pydiceplot examples — regenerates the showcase images in ../images/.

Run with:

    pixi run example
"""

import os

import numpy as np

import pydiceplot
from pydiceplot import dice_plot
from pydiceplot.plots.backends._dice_utils import (
    get_diceplot_example_data,
    get_example_cat_c_colors,
)

IMAGES_DIR = "images"


def example_n_categorical(n: int, filename: str):
    """Categorical mode: each pip is a cat_c category coloured by a fixed palette.

    Mirrors ggdiceplot's `geom_dice(aes(dots=cat_c), fill=...)` with a fixed
    colour per category.
    """
    cat_c_colors = get_example_cat_c_colors()
    data = get_diceplot_example_data(n)
    present = list(data["PathologyVariable"].unique())
    fig = dice_plot(
        data=data,
        cat_a="CellType",
        cat_b="Pathway",
        cat_c="PathologyVariable",
        cat_c_colors={v: cat_c_colors[v] for v in present},
        title=f"Dice Plot with {n} Pathology Variables",
        fig_width=9,
        fig_height=10,
    )
    fig.save(IMAGES_DIR, filename, formats=".png")


def example_per_dot_continuous(filename: str):
    """Per-dot continuous mode: each pip encodes `fill_col` and `size_col`.

    Mirrors ggdiceplot's `geom_dice(aes(dots=cat_c, fill=lfc, size=-log10(q)))`.
    """
    rng = np.random.default_rng(1)
    data = get_diceplot_example_data(4)
    data["lfc"] = rng.normal(0, 1.2, len(data))
    data["nlq"] = rng.uniform(0.5, 4.0, len(data))
    fig = dice_plot(
        data=data,
        cat_a="CellType",
        cat_b="Pathway",
        cat_c="PathologyVariable",
        fill_col="lfc",
        size_col="nlq",
        fill_legend_label="Log2FC",
        size_legend_label="-log10(q)",
        color_map="RdBu_r",
        title="Per-dot continuous (Log2FC x -log10 q)",
        fig_width=10,
        fig_height=10,
    )
    fig.save(IMAGES_DIR, filename, formats=".png")


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    pydiceplot.set_backend("matplotlib")

    example_n_categorical(4, "dice_4_categorical")
    example_n_categorical(6, "dice_6_categorical")
    example_n_categorical(9, "dice_9_categorical")
    example_per_dot_continuous("dice_per_dot_continuous")

    print(f"Wrote showcase images to {IMAGES_DIR}/")
