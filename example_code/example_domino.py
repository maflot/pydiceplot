# example_domino.py
from pydiceplot.plots.backends._domino_utils import (get_domino_example_data)
from pydiceplot import domino_plot
import pydiceplot

# Set the backend to either "matplotlib" or "plotly"
pydiceplot.set_backend("matplotlib")  # You can change to "plotly" if desired
pydiceplot.set_backend("plotly")  # You can change to "plotly" if desired

if __name__ == "__main__":

    plot_path = "./plots"

    data_combined = get_domino_example_data()

    # Define logFC color scale
    logfc_colors = {
        "low": "blue",
        "mid": "white",
        "high": "red"
    }

    # Define gene list
    gene_list = ["GeneA", "GeneB", "GeneC"]

    # Use the domino_plot function
    fig = domino_plot(
        data=data_combined,
        gene_list=gene_list,
        feature_col="gene",
        celltype_col="Cell_Type",
        contrast_col="Group",
        contrast_levels=["Type1", "Type2"],
        contrast_labels=["Type 1", "Type 2"],
        var_id="var",
        logfc_col="logFC",
        pval_col="adj_p_value",
        switch_axis=False,
        min_dot_size=1,
        max_dot_size=5,
        logfc_limits=(-2, 2),
        logfc_colors=logfc_colors,
        title="Domino example"
    )

    # Optionally display the figure
    fig.show()
