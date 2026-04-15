"""Port of ggdiceplot's `sample_dice_miRNA` miRNA dysregulation example.

5 miRNAs × 5 compounds × 4 organs; each pip's colour encodes a discrete
regulation direction (Down / Unchanged / Up) rather than a continuous value.

This demonstrates `fill_palette`: `pips="Organ"` selects the pip slot, while
`fill="direction"` + `fill_palette={...}` colours each pip according to a
separate categorical column.
"""

import os

import matplotlib.pyplot as plt
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
    compound_order = ["Control", "Compound_1", "Compound_2", "Compound_3", "Compound_4"]
    mirna_order = [f"miR-{i}" for i in range(1, 6)]
    organ_order = ["Lung", "Liver", "Brain", "Kidney"]

    fig, _ = dice_plot(
        data,
        x="miRNA", y="Compound", pips="Organ",
        fill="direction",
        fill_palette=DIRECTION_COLORS,
        x_order=mirna_order,
        y_order=compound_order,
        pips_order=organ_order,
        title="miRNA dysregulation direction per organ",
        pips_label="Organ",
        figsize=(10, 8),
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "ggport_mirna_direction.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
