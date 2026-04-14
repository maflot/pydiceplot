"""Port of ggdiceplot's demo_output/ZEBRA_domino_example.png.

9 genes × up-to-27 cell types × 5 disease contrasts (MS-CT, AD-CT, ASD-CT,
FTD-CT, HD-CT). Produces a 5-pip die face per cell where each pip encodes
Log2FC as colour and -log10(FDR) as size — filtering to rows with p < 0.05
first so only significant contrasts show.

Reference: create_demo_plots.R in ggdiceplot.
"""

import math
import os

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

    # Collapse duplicate (gene, cell_type, contrast) rows by averaging logFC
    # and taking the strongest FDR — mirrors the R pipeline's group_by/summarise.
    agg = (
        df.groupby(["gene", "cell_type", "contrast"], as_index=False)
        .agg(logFC=("logFC", "mean"), FDR=("FDR", "min"))
    )
    agg["neg_log10_fdr"] = [-math.log10(v) if v > 0 else 0 for v in agg["FDR"]]

    cell_types = sorted(agg["cell_type"].unique())

    fig = dice_plot(
        data=agg,
        cat_a="gene",
        cat_b="cell_type",
        cat_c="contrast",
        fill_col="logFC",
        size_col="neg_log10_fdr",
        cat_a_order=GENES,
        cat_b_order=cell_types,
        cat_c_order=CONTRASTS,
        title="ZEBRA Sex DEGs Domino Plot",
        fill_legend_label="Log2FC",
        size_legend_label="-log10(FDR)",
        position_legend_label="contrast",
        color_map="ggdiceplot_pg",
        cell_width=0.9, cell_height=0.9,
        fig_width=12, fig_height=14,
    )
    fig.save(out_dir, "ggport_zebra_domino", formats=".png")


if __name__ == "__main__":
    run()
