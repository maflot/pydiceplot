"""Port of ggdiceplot's demo_output/example4_fill_only.png.

Same data as the oral microbiome plot but only fill is mapped. `pip_scale=1.0`
asks pips to fill their sub-cells maximally since size is constant.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

import pydiceplot
from pydiceplot import dice_plot

from ._palette import register as register_palette


DATA = os.path.join(os.path.dirname(__file__), "data", "sample_dice_data1.csv")


def run(out_dir: str = "images") -> None:
    register_palette()
    pydiceplot.set_backend("matplotlib")

    data = pd.read_csv(DATA)

    fig, _ = dice_plot(
        data,
        x="specimen",
        y="taxon",
        pips="disease",
        fill="lfc",
        title="Fill-only dice plot (pip_scale = 1.0)",
        fill_label="Log2FC",
        pips_label="disease",
        cmap="ggdiceplot_pg",
        pip_scale=1.0,
        tile_size=0.9,
        figsize=(10, 10),
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(
        os.path.join(out_dir, "ggport_oral_microbiome_fill_only.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close(fig)


if __name__ == "__main__":
    run()
