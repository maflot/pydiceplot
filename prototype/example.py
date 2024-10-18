import numpy as np
import pandas as pd
from pydiceplot import dice_plot
import pydiceplot
pydiceplot.set_backend("matplotlib")


if __name__ == "__main__":

    plot_path = "./plots"

    # Define cell types (cat_a)
    cell_types = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", "Endothelial"]

    # Define pathways (cat_b) and groups
    pathways_initial = [
        "Apoptosis", "Inflammation", "Metabolism", "Signal Transduction", "Synaptic Transmission",
        "Cell Cycle", "DNA Repair", "Protein Synthesis", "Lipid Metabolism", "Neurotransmitter Release"
    ]

    # Extend pathways to 15 for higher examples
    pathways_extended = pathways_initial + [
        "Oxidative Stress", "Energy Production", "Calcium Signaling", "Synaptic Plasticity", "Immune Response"
    ]


    # Function to create and save dice plots
    def create_and_save_dice_plot(num_vars, pathology_vars, cat_c_colors, output_str, title):
        # Assign groups to pathways
        # Ensure that each pathway has only one group
        pathway_groups = pd.DataFrame({
            "Pathway": pathways_extended[:15],  # Ensure 15 pathways
            "Group": [
                "Linked", "UnLinked", "Other", "Linked", "UnLinked",
                "UnLinked", "Other", "Other", "Other", "Linked",
                "Other", "Other", "Linked", "UnLinked", "Other"
            ]
        })

        # Define group colors
        group_colors = {
            "Linked": "#333333",
            "UnLinked": "#888888",
            "Other": "#DDDDDD"
        }

        # Create dummy data
        np.random.seed(123)
        data = pd.DataFrame([(ct, pw) for ct in cell_types for pw in pathways_extended[:15]],
                            columns=["CellType", "Pathway"])

        # Assign random pathology variables to each combination
        data_list = []
        for idx, row in data.iterrows():
            variables = np.random.choice(pathology_vars, size=np.random.randint(1, num_vars + 1), replace=False)
            for var in variables:
                data_list.append({
                    "CellType": row["CellType"],
                    "Pathway": row["Pathway"],
                    "PathologyVariable": var
                })

        # Create DataFrame from data_list
        data_expanded = pd.DataFrame(data_list)

        # Merge the group assignments into the data
        data_expanded = data_expanded.merge(pathway_groups, left_on="Pathway", right_on="Pathway", how="left")

        # Use the dice_plot function
        fig = dice_plot(
            data=data_expanded,
            cat_a="CellType",
            cat_b="Pathway",
            cat_c="PathologyVariable",
            group="Group",
            switch_axis=False,
            group_alpha=0.6,
            title=title,
            cat_c_colors=cat_c_colors,
            group_colors=group_colors,
            max_dice_sides=6  # Adjust if needed
        )

        # Optionally display the figure
        fig.show()
        fig.save(plot_path, output_str, formats=".png")


    # Example 1: 3 Pathology Variables
    pathology_vars_3 = ["Stroke", "Cancer", "Flu"]
    cat_c_colors_3 = {
        "Stroke": "#d5cccd",
        "Cancer": "#cb9992",
        "Flu": "#ad310f"
    }
    create_and_save_dice_plot(
        num_vars=3,
        pathology_vars=pathology_vars_3,
        cat_c_colors=cat_c_colors_3,
        output_str="dice_plot_3_example",
        title="Dice Plot with 3 Pathology Variables"
    )

    # Example 2: 4 Pathology Variables
    pathology_vars_4 = ["Stroke", "Cancer", "Flu", "ADHD"]
    cat_c_colors_4 = {
        "Stroke": "#d5cccd",
        "Cancer": "#cb9992",
        "Flu": "#ad310f",
        "ADHD": "#7e2a20"
    }
    create_and_save_dice_plot(
        num_vars=4,
        pathology_vars=pathology_vars_4,
        cat_c_colors=cat_c_colors_4,
        output_str="dice_plot_4_example",
        title="Dice Plot with 4 Pathology Variables"
    )

    # Example 3: 5 Pathology Variables
    pathology_vars_5 = ["Stroke", "Cancer", "Flu", "ADHD", "Lymphom"]
    cat_c_colors_5 = {
        "Stroke": "#d5cccd",
        "Cancer": "#cb9992",
        "Flu": "#ad310f",
        "ADHD": "#7e2a20",
        "Lymphom": "#FFD700"  # Gold color for Lymphom
    }
    create_and_save_dice_plot(
        num_vars=5,
        pathology_vars=pathology_vars_5,
        cat_c_colors=cat_c_colors_5,
        output_str="dice_plot_5_example",
        title="Dice Plot with 5 Pathology Variables"
    )

    # Example 4: 6 Pathology Variables
    pathology_vars_6 = ["Alzheimer's disease", "Cancer", "Flu", "ADHD", "Age", "Weight"]
    cat_c_colors_6 = {
        "Alzheimer's disease": "#d5cccd",
        "Cancer": "#cb9992",
        "Flu": "#ad310f",
        "ADHD": "#7e2a20",
        "Age": "#FFD700",  # Gold color for Age
        "Weight": "#FF6622"  # Orange color for Weight
    }
    create_and_save_dice_plot(
        num_vars=6,
        pathology_vars=pathology_vars_6,
        cat_c_colors=cat_c_colors_6,
        output_str="dice_plot_6_example",
        title="Dice Plot with 6 Pathology Variables"
    )

    print(f"All dice plots have been saved to the '{plot_path}' directory in both HTML and PNG formats.")
