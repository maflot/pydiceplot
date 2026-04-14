"""Port of ggdiceplot's demo_output/example_large.png.

Same encoding as the small oral microbiome plot, but with 60 taxa × 2
specimens: a tall aspect ratio stress test for the legend layout and
axis label rendering.
"""

import math
import os

import pandas as pd

import pydiceplot
from pydiceplot import dice_plot

from ._palette import register as register_palette


DATA = os.path.join(os.path.dirname(__file__), "data", "sample_dice_large.csv")


def run(out_dir: str = "images") -> None:
    register_palette()
    pydiceplot.set_backend("matplotlib")

    data = pd.read_csv(DATA)
    data["neg_log10_q"] = [-math.log10(v) if v > 0 else 0 for v in data["q"].fillna(1)]

    fig = dice_plot(
        data=data,
        cat_a="specimen",
        cat_b="taxon",
        cat_c="disease",
        fill_col="lfc",
        size_col="neg_log10_q",
        title="Oral microbiome — 60 taxa",
        fill_legend_label="Log2FC",
        size_legend_label="-log10(q)",
        position_legend_label="disease",
        color_map="ggdiceplot_pg",
        pip_scale=0.9,
        fig_width=10, fig_height=24,
    )
    fig.save(out_dir, "ggport_oral_microbiome_large", formats=".png")


if __name__ == "__main__":
    run()
