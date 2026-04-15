"""Port of ggdiceplot's demo_output/ZEBRA_domino_example.png.

9 genes × up-to-27 cell types × 5 disease contrasts (MS-CT, AD-CT, ASD-CT,
FTD-CT, HD-CT). Produces a 5-pip die face per cell where each pip encodes
Log2FC as colour and -log10(FDR) as size — filtered to rows with `PValue < 0.05`
first so only significant contrasts show.
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd

import pydiceplot
from pydiceplot import dice_plot

from ._palette import register as register_palette


DATA = os.path.join(os.path.dirname(__file__), "data", "ZEBRA_sex_degs_set.csv")

GENES = ["SPP1", "APOE", "SERPINA1", "PINK1", "ANGPT1", "ANGPT2", "APP", "CLU", "ABCA7"]
CONTRASTS = ["MS-CT", "AD-CT", "ASD-CT", "FTD-CT", "HD-CT"]


def run(out_dir: str = "images") -> None:
    register_palette()
    pydiceplot.set_backend("matplotlib")

    df = pd.read_csv(DATA)
    df = df[df["gene"].isin(GENES) & df["contrast"].isin(CONTRASTS) & (df["PValue"] < 0.05)].copy()

    agg = (
        df.groupby(["gene", "cell_type", "contrast"], as_index=False)
        .agg(logFC=("logFC", "mean"), FDR=("FDR", "min"))
    )
    agg["neg_log10_fdr"] = [-math.log10(v) if v > 0 else 0 for v in agg["FDR"]]

    cell_types = sorted(agg["cell_type"].unique())

    fig, _ = dice_plot(
        agg,
        x="gene", y="cell_type", pips="contrast",
        fill="logFC", size="neg_log10_fdr",
        x_order=GENES,
        y_order=cell_types,
        pips_order=CONTRASTS,
        title="ZEBRA Sex DEGs Domino Plot",
        fill_label="Log2FC",
        size_label="-log10(FDR)",
        pips_label="contrast",
        cmap="ggdiceplot_pg",
        tile_width=0.9, tile_height=0.9,
        figsize=(12, 14),
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "ggport_zebra_domino.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
