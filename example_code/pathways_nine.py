"""Creative n=9 example: nine canonical signaling pathways per die face.

Each tile encodes one (cell type, treatment) combination across the nine
most-studied developmental/oncogenic signaling pathways:

    Wnt     Notch      Hedgehog
    TGF-β   Hippo      PI3K-AKT
    MAPK    JAK-STAT   NF-κB

Pip colour encodes log2-fold-change relative to vehicle (diverging
purple→green, matching the ggdiceplot palette), and pip size encodes
pathway-activity significance (-log10 q). This is a fully-populated 3×3
die face — every pip slot is drawn.

Data is synthetic but built from plausible pathway-level cross-talk.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pydiceplot
from pydiceplot import dice_plot

from ._palette import register as register_palette


PATHWAYS = [
    "Wnt",
    "Notch",
    "Hedgehog",
    "TGF-β",
    "Hippo",
    "PI3K-AKT",
    "MAPK",
    "JAK-STAT",
    "NF-κB",
]

CELL_TYPES = [
    "Fibroblast",
    "Macrophage",
    "T cell",
    "Epithelial",
    "Endothelial",
    "IntestinalSC",
    "NeuralSC",
    "Hepatocyte",
]

TREATMENTS = ["Vehicle", "TGF-β1", "LPS", "WNT3A", "Hypoxia", "IFN-γ"]


def _synthesize() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for ct in CELL_TYPES:
        for tx in TREATMENTS:
            for pw in PATHWAYS:
                rows.append(
                    {
                        "cell_type": ct,
                        "treatment": tx,
                        "pathway": pw,
                        "lfc": rng.normal(0.0, 0.45),
                        "q": rng.uniform(0.01, 0.9),
                    }
                )
    df = pd.DataFrame(rows)

    def boost(mask, pathway, lfc_mean, q_mean):
        pw_mask = mask & (df["pathway"] == pathway)
        df.loc[pw_mask, "lfc"] = rng.normal(lfc_mean, 0.3, pw_mask.sum())
        df.loc[pw_mask, "q"] = rng.uniform(q_mean * 0.5, q_mean * 1.5, pw_mask.sum())

    fibro_tgf = (df["cell_type"] == "Fibroblast") & (df["treatment"] == "TGF-β1")
    boost(fibro_tgf, "TGF-β", 3.5, 1e-5)
    boost(fibro_tgf, "MAPK", 1.2, 1e-3)

    macro_lps = (df["cell_type"] == "Macrophage") & (df["treatment"] == "LPS")
    boost(macro_lps, "NF-κB", 3.8, 1e-6)
    boost(macro_lps, "JAK-STAT", 2.4, 1e-4)
    boost(macro_lps, "MAPK", 2.0, 1e-4)

    isc_wnt = (df["cell_type"] == "IntestinalSC") & (df["treatment"] == "WNT3A")
    boost(isc_wnt, "Wnt", 4.0, 1e-6)
    boost(isc_wnt, "Notch", 1.1, 1e-3)

    hep_hyp = (df["cell_type"] == "Hepatocyte") & (df["treatment"] == "Hypoxia")
    boost(hep_hyp, "PI3K-AKT", 2.8, 1e-5)
    boost(hep_hyp, "Hippo", -1.6, 1e-3)

    tc_ifn = (df["cell_type"] == "T cell") & (df["treatment"] == "IFN-γ")
    boost(tc_ifn, "JAK-STAT", 4.2, 1e-7)

    endo_hyp = (df["cell_type"] == "Endothelial") & (df["treatment"] == "Hypoxia")
    boost(endo_hyp, "PI3K-AKT", 2.0, 1e-4)
    boost(endo_hyp, "MAPK", 1.5, 1e-3)

    nsc_wnt = (df["cell_type"] == "NeuralSC") & (df["treatment"] == "WNT3A")
    boost(nsc_wnt, "Wnt", 3.2, 1e-5)
    boost(nsc_wnt, "Hedgehog", 1.8, 1e-3)

    df["neg_log10_q"] = -np.log10(df["q"].clip(lower=1e-12))
    return df


def run(out_dir: str = "images") -> None:
    register_palette()
    pydiceplot.set_backend("matplotlib")

    df = _synthesize()

    fig, _ = dice_plot(
        df,
        x="treatment",
        y="cell_type",
        pips="pathway",
        fill="lfc",
        size="neg_log10_q",
        x_order=TREATMENTS,
        y_order=CELL_TYPES,
        pips_order=PATHWAYS,
        title="Pathway activity — 9 canonical signaling modules",
        fill_label="Log2FC",
        size_label="-log10(q)",
        pips_label="pathway",
        cmap="ggdiceplot_pg",
        pip_scale=0.95,
        tile_size=0.92,
        figsize=(13, 10),
    )
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(
        os.path.join(out_dir, "ggport_pathways_nine.png"), bbox_inches="tight", dpi=150
    )
    plt.close(fig)


if __name__ == "__main__":
    run()
