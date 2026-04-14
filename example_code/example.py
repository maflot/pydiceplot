"""Regenerate all showcase images under `./images/`.

The first block is the quick categorical + per-dot tour used in the readme's
"Quick start" section. The `ggport_*` block runs 1-to-1 ports of the plots
in `ggdiceplot/demo_output/` plus a creative n=9 example that exercises the
fully-populated 3×3 die face.

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

from . import (
    oral_microbiome,
    oral_microbiome_fill_only,
    oral_microbiome_large,
    mirna_direction,
    zebra_domino,
    pathways_nine,
)

IMAGES_DIR = "images"


def example_n_categorical(n: int, filename: str):
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

    # Quick-start showcase
    example_n_categorical(4, "dice_4_categorical")
    example_n_categorical(6, "dice_6_categorical")
    example_n_categorical(9, "dice_9_categorical")
    example_per_dot_continuous("dice_per_dot_continuous")

    # ggdiceplot demo ports
    oral_microbiome.run(IMAGES_DIR)
    oral_microbiome_fill_only.run(IMAGES_DIR)
    oral_microbiome_large.run(IMAGES_DIR)
    mirna_direction.run(IMAGES_DIR)
    zebra_domino.run(IMAGES_DIR)

    # Creative n=9 example
    pathways_nine.run(IMAGES_DIR)

    print(f"Wrote showcase images to {IMAGES_DIR}/")
