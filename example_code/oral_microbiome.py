"""Port of ggdiceplot's demo_output/example2.png.

Data: `sample_dice_data2` — 8 oral microbiome taxa × 5 specimen sites
× 4 disease states. Per-pip continuous fill (Log2FC) and size (-log10 q),
rendered on a 4-pip die face.
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

import pydiceplot
from pydiceplot import dice_plot

from ._palette import register as register_palette


DATA = os.path.join(os.path.dirname(__file__), "data", "sample_dice_data2.csv")


def run(out_dir: str = "images") -> None:
    register_palette()
    pydiceplot.set_backend("matplotlib")

    data = pd.read_csv(DATA)
    data["neg_log10_q"] = [-math.log10(v) if v > 0 else 0 for v in data["q"].fillna(1)]

    fig, _ = dice_plot(
        data,
        x="specimen", y="taxon", pips="disease",
        fill="lfc", size="neg_log10_q",
        title="Oral microbiome — Log2FC × -log10 q",
        fill_label="Log2FC",
        size_label="-log10(q)",
        pips_label="disease",
        cmap="ggdiceplot_pg",
        pip_scale=0.9,
        figsize=(10, 10),
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "ggport_oral_microbiome.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
