"""Port of ggdiceplot's demo_output/example4_fill_only.png.

Same data as the oral microbiome plot but only fill is mapped. `pip_scale=1.0`
asks pips to fill their sub-cells maximally since size is constant.
"""

import os

import numpy as np
import pandas as pd

import pydiceplot
from pydiceplot import dice_plot

from ._palette import register as register_palette


DATA = os.path.join(os.path.dirname(__file__), "data", "sample_dice_data1.csv")


def run(out_dir: str = "images") -> None:
    register_palette()
    pydiceplot.set_backend("matplotlib")

    data = pd.read_csv(DATA)

    fig = dice_plot(
        data=data,
        cat_a="specimen",
        cat_b="taxon",
        cat_c="disease",
        fill_col="lfc",
        title="Fill-only dice plot (pip_scale = 1.0)",
        fill_legend_label="Log2FC",
        position_legend_label="disease",
        color_map="ggdiceplot_pg",
        pip_scale=1.0,
        cell_width=0.9, cell_height=0.9,
        fig_width=10, fig_height=10,
    )
    fig.save(out_dir, "ggport_oral_microbiome_fill_only", formats=".png")


if __name__ == "__main__":
    run()
