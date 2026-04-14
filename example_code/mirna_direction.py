"""Port of ggdiceplot's `sample_dice_miRNA` miRNA dysregulation example.

5 miRNAs × 5 compounds × 4 organs; each pip's colour encodes a discrete
regulation direction (Down / Unchanged / Up) rather than a continuous value.

This demonstrates the `fill_palette` argument: `cat_c="Organ"` selects the pip
slot, while `fill_col="direction"` + `fill_palette={...}` colours each pip
according to a separate categorical column.
"""

import os

import pandas as pd

import pydiceplot
from pydiceplot import dice_plot


DATA = os.path.join(os.path.dirname(__file__), "data", "sample_dice_miRNA.csv")


DIRECTION_COLORS = {
    "Down":      "#2166AC",
    "Unchanged": "#CCCCCC",
    "Up":        "#B2182B",
}


def run(out_dir: str = "images") -> None:
    pydiceplot.set_backend("matplotlib")

    data = pd.read_csv(DATA)
    # Preserve the R factor level order from the .rda file
    compound_order = ["Control", "Compound_1", "Compound_2", "Compound_3", "Compound_4"]
    mirna_order = [f"miR-{i}" for i in range(1, 6)]
    organ_order = ["Lung", "Liver", "Brain", "Kidney"]

    fig = dice_plot(
        data=data,
        cat_a="miRNA",
        cat_b="Compound",
        cat_c="Organ",
        fill_col="direction",
        fill_palette=DIRECTION_COLORS,
        cat_a_order=mirna_order,
        cat_b_order=compound_order,
        cat_c_order=organ_order,
        title="miRNA dysregulation direction per organ",
        position_legend_label="Organ",
        fig_width=10, fig_height=8,
    )
    fig.save(out_dir, "ggport_mirna_direction", formats=".png")


if __name__ == "__main__":
    run()
