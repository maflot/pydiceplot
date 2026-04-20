"""Standalone example for the refactored domino plot API."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

import pydiceplot
from pydiceplot import domino_plot
from pydiceplot.plots.backends._domino_utils import get_domino_example_data


def run(out_dir: str = "images") -> None:
    pydiceplot.set_backend("matplotlib")
    data = get_domino_example_data()
    fig, _ = domino_plot(
        data,
        "gene", "Cell_Type", "Group",
        features=["GeneA", "GeneB", "GeneC"],
        label="var",
        fill="logFC",
        size="neg_log10_adj_p",
        contrast_order=["Type1", "Type2"],
        contrast_labels=["Type 1", "Type 2"],
        fill_label="Log2FC",
        size_label="-log10(adj p)",
        title="Domino plot example",
        figsize=(9, 5.5),
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "domino_example.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
