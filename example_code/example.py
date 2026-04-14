"""Basic pydiceplot examples — categorical + per-dot continuous modes.

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

PLOT_PATH = "./plots"


def categorical_examples():
    """Four dice plots at n = 2..6 showing per-pip category color."""
    cat_c_colors = get_example_cat_c_colors()
    for n in (2, 3, 4, 5, 6):
        data = get_diceplot_example_data(n)
        pathology_vars = list(data["PathologyVariable"].unique())
        current_colors = {v: cat_c_colors[v] for v in pathology_vars}
        fig = dice_plot(
            data=data,
            cat_a="CellType",
            cat_b="Pathway",
            cat_c="PathologyVariable",
            cat_c_colors=current_colors,
            title=f"Dice Plot with {n} Pathology Variables",
            fig_width=9, fig_height=10,
        )
        fig.save(PLOT_PATH, f"dice_{n}_categorical", formats=".png")


def per_dot_continuous_example(backend: str):
    """Per-pip continuous fill + size, mirroring ggdiceplot's
    `geom_dice(aes(dots=cat_c, fill=lfc, size=-log10(q)))` usage.

    `fig_width`/`fig_height` are in inches for matplotlib and pixels for plotly,
    so we pick sensible defaults for each.
    """
    rng = np.random.default_rng(1)
    data = get_diceplot_example_data(4)
    data["lfc"] = rng.normal(0, 1.2, len(data))
    data["nlq"] = rng.uniform(0.5, 4.0, len(data))

    size_kwargs = (
        dict(fig_width=10, fig_height=10) if backend == "matplotlib"
        else dict(fig_width=900, fig_height=650)
    )
    fig = dice_plot(
        data=data,
        cat_a="CellType",
        cat_b="Pathway",
        cat_c="PathologyVariable",
        fill_col="lfc",
        size_col="nlq",
        title="Per-dot continuous (Log2FC × -log10 q)",
        fill_legend_label="Log2FC",
        size_legend_label="-log10(q)",
        color_map="RdBu_r",
        **size_kwargs,
    )
    fig.save(PLOT_PATH, f"dice_per_dot_continuous_{backend}", formats=".png")


if __name__ == "__main__":
    os.makedirs(PLOT_PATH, exist_ok=True)

    pydiceplot.set_backend("matplotlib")
    categorical_examples()
    per_dot_continuous_example("matplotlib")

    pydiceplot.set_backend("plotly")
    per_dot_continuous_example("plotly")

    print(f"All dice plots saved to {PLOT_PATH}/")
