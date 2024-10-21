# example_domino.py

import numpy as np
import pandas as pd
from pydiceplot import domino_plot
import pydiceplot

# Set the backend to either "matplotlib" or "plotly"
pydiceplot.set_backend("matploptlib")  # You can change to "plotly" if desired

if __name__ == "__main__":

    plot_path = "./plots"

    # Define genes
    gene_list = ["GeneA", "GeneB", "GeneC"]

    # Define cell types
    cell_types = ["Neuron", "Astrocyte", "Microglia"]

    # Define contrasts
    contrasts = ["Type1", "Type2"]

    # Define vars for each contrast
    vars_type1 = ["MCI-NCI", "AD-MCI", "AD-NCI"]
    vars_type2 = ["Amyloid", "Plaq N", "Tangles", "NFT"]

    # Function to create and save domino plots
    def create_and_save_domino_plot(output_str, title):
        # Create a data frame with all combinations
        data = pd.DataFrame(
            [(gene, cell_type, contrast) for gene in gene_list for cell_type in cell_types for contrast in contrasts],
            columns=["gene", "Cell_Type", "Group"]
        )

        # Assign the appropriate vars to each contrast
        np.random.seed(123)  # Ensure reproducibility

        data_type1 = data[data["Group"] == "Type1"].copy()
        data_type1["var"] = np.random.choice(vars_type1, size=len(data_type1), replace=True)

        data_type2 = data[data["Group"] == "Type2"].copy()
        data_type2["var"] = np.random.choice(vars_type2, size=len(data_type2), replace=True)

        # Combine the data
        data_combined = pd.concat([data_type1, data_type2], ignore_index=True)

        # Assign random values for logFC and adjusted p-values
        data_combined["logFC"] = np.random.uniform(-2, 2, size=len(data_combined))
        data_combined["adj_p_value"] = np.random.uniform(0.0001, 0.05, size=len(data_combined))

        # Define logFC color scale
        logfc_colors = {
            "low": "blue",
            "mid": "white",
            "high": "red"
        }

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
            title=title
        )

        # Optionally display the figure
        fig.show()

        # Save the figure
        fig.save(plot_path, output_str, formats=[".png"])

    # Create and save the domino plot
    create_and_save_domino_plot(
        output_str="domino_plot_example",
        title="Domino Plot Example"
    )

    print(f"Domino plot has been saved to the '{plot_path}' directory.")
