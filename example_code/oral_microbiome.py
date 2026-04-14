"""Port of ggdiceplot's demo_output/example2.png.

Data: `sample_dice_data2` — 8 oral microbiome taxa × 5 specimen sites
× 4 disease states. Per-pip continuous fill (Log2FC) and size (-log10 q),
rendered on a 4-pip die face.

Reference: ggdiceplot/demo_output/create_demo_plots.R
"""

import os

import pandas as pd

import pydiceplot
from pydiceplot import dice_plot

from ._palette import register as register_palette


DATA = os.path.join(os.path.dirname(__file__), "data", "sample_dice_data2.csv")


def run(out_dir: str = "images") -> None:
    register_palette()
    pydiceplot.set_backend("matplotlib")

    data = pd.read_csv(DATA)
    data["neg_log10_q"] = -data["q"].apply(lambda x: 0 if x == 0 else _log10(x))

    fig = dice_plot(
        data=data,
        cat_a="specimen",
        cat_b="taxon",
        cat_c="disease",
        fill_col="lfc",
        size_col="neg_log10_q",
        title="Oral microbiome — Log2FC × -log10 q",
        fill_legend_label="Log2FC",
        size_legend_label="-log10(q)",
        position_legend_label="disease",
        color_map="ggdiceplot_pg",
        pip_scale=0.9,
        fig_width=10, fig_height=10,
    )
    fig.save(out_dir, "ggport_oral_microbiome", formats=".png")


def _log10(v: float) -> float:
    import math
    return math.log10(v)


if __name__ == "__main__":
    run()
