# example_dice_plots.py

import pandas as pd
import numpy as np
import os
from prototype._diceplot import dice_plot  # Import the dice_plot function from your dice_plot.py

# Set plot path
plot_path = "../images"

# Ensure the plot_path directory exists
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

### First Example: 3 Pathology Variables ###

# Define the variables and their colors for 3 variables
pathology_variables = ["Amyloid", "NFT", "Tangles"]
cat_c_colors = {
    "Amyloid": "#d5cccd",
    "NFT": "#cb9992",
    "Tangles": "#ad310f"
}

# Define cell types (cat_a)
cell_types = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", "Endothelial"]

# Define pathways (cat_b) and groups
pathways = [
    "Apoptosis", "Inflammation", "Metabolism", "Signal Transduction", "Synaptic Transmission",
    "Cell Cycle", "DNA Repair", "Protein Synthesis", "Lipid Metabolism", "Neurotransmitter Release"
]

# Assign groups to pathways
pathway_groups = pd.DataFrame({
    "Pathway": pathways,
    "Group": [
        "BBB-linked", "Cell-proliferation", "Other", "BBB-linked", "Cell-proliferation",
        "Cell-proliferation", "Other", "Other", "Other", "BBB-linked"
    ]
})

# Define group colors
group_colors = {
    "BBB-linked": "#333333",
    "Cell-proliferation": "#888888",
    "Other": "#DDDDDD"
}

# Create dummy data
np.random.seed(123)
data = pd.DataFrame([(ct, pw) for ct in cell_types for pw in pathways], columns=["CellType", "Pathway"])

# Assign random pathology variables to each combination
data_list = []
for idx, row in data.iterrows():
    n_vars = np.random.randint(1, 4)  # Random number between 1 and 3
    variables = np.random.choice(pathology_variables, size=n_vars, replace=False)
    for var in variables:
        data_list.append({
            "CellType": row["CellType"],
            "Pathway": row["Pathway"],
            "PathologyVariable": var
        })

# Create DataFrame from data_list
data_expanded = pd.DataFrame(data_list)

# Merge the group assignments into the data
data_expanded = data_expanded.merge(pathway_groups, on="Pathway", how="left")

# Use the dice_plot function
dice_plot(
    data=data_expanded,
    cat_a="CellType",
    cat_b="Pathway",
    cat_c="PathologyVariable",
    group="Group",
    plot_path=plot_path,
    output_str="dice_plot_3_example",
    switch_axis=False,
    group_alpha=0.6,
    title="Dice Plot with 3 Pathology Variables",
    cat_c_colors=cat_c_colors,
    group_colors=group_colors,
    format=".png"
)

### Second Example: 4 Pathology Variables ###

# Define the variables and their colors for 4 variables
pathology_variables = ["Amyloid", "NFT", "Tangles", "Plaq N"]
cat_c_colors = {
    "Amyloid": "#d5cccd",
    "NFT": "#cb9992",
    "Tangles": "#ad310f",
    "Plaq N": "#7e2a20"
}

# Update pathways to add more pathways for a total of 15
pathways = [
    "Apoptosis", "Inflammation", "Metabolism", "Signal Transduction", "Synaptic Transmission",
    "Cell Cycle", "DNA Repair", "Protein Synthesis", "Lipid Metabolism", "Neurotransmitter Release",
    "Oxidative Stress", "Energy Production", "Calcium Signaling", "Synaptic Plasticity", "Immune Response"
]

# Assign groups to pathways
pathway_groups = pd.DataFrame({
    "Pathway": pathways,
    "Group": [
        "BBB-linked", "Cell-proliferation", "Other", "BBB-linked", "Cell-proliferation",
        "Cell-proliferation", "Other", "Other", "Other", "BBB-linked",
        "Other", "Other", "BBB-linked", "Cell-proliferation", "Other"
    ]
})

# Create dummy data
np.random.seed(123)
data = pd.DataFrame([(ct, pw) for ct in cell_types for pw in pathways], columns=["CellType", "Pathway"])

# Assign random pathology variables to each combination
data_list = []
for idx, row in data.iterrows():
    n_vars = np.random.randint(1, 5)  # Random number between 1 and 4
    variables = np.random.choice(pathology_variables, size=n_vars, replace=False)
    for var in variables:
        data_list.append({
            "CellType": row["CellType"],
            "Pathway": row["Pathway"],
            "PathologyVariable": var
        })

# Create DataFrame from data_list
data_expanded = pd.DataFrame(data_list)

# Merge the group assignments into the data
data_expanded = data_expanded.merge(pathway_groups, on="Pathway", how="left")

# Use the dice_plot function
dice_plot(
    data=data_expanded,
    cat_a="CellType",
    cat_b="Pathway",
    cat_c="PathologyVariable",
    group="Group",
    plot_path=plot_path,
    output_str="dice_plot_4_example",
    switch_axis=False,
    group_alpha=0.6,
    title="Dice Plot with 4 Pathology Variables",
    cat_c_colors=cat_c_colors,
    group_colors=group_colors,
    format=".png"
)

### Third Example: 5 Pathology Variables ###

# Define the variables and their colors for 5 variables
pathology_variables = ["Amyloid", "NFT", "Tangles", "Plaq N", "Var5"]
cat_c_colors = {
    "Amyloid": "#d5cccd",
    "NFT": "#cb9992",
    "Tangles": "#ad310f",
    "Plaq N": "#7e2a20",
    "Var5": "#FFD700"  # Gold color for Var5
}

# Create dummy data
np.random.seed(123)
data = pd.DataFrame([(ct, pw) for ct in cell_types for pw in pathways], columns=["CellType", "Pathway"])

# Assign random pathology variables to each combination
data_list = []
for idx, row in data.iterrows():
    n_vars = np.random.randint(1, 6)  # Random number between 1 and 5
    variables = np.random.choice(pathology_variables, size=n_vars, replace=False)
    for var in variables:
        data_list.append({
            "CellType": row["CellType"],
            "Pathway": row["Pathway"],
            "PathologyVariable": var
        })

# Create DataFrame from data_list
data_expanded = pd.DataFrame(data_list)

# Merge the group assignments into the data
data_expanded = data_expanded.merge(pathway_groups, on="Pathway", how="left")

# Use the dice_plot function
dice_plot(
    data=data_expanded,
    cat_a="CellType",
    cat_b="Pathway",
    cat_c="PathologyVariable",
    group="Group",
    plot_path=plot_path,
    output_str="dice_plot_5_example",
    switch_axis=False,
    group_alpha=0.6,
    title="Dice Plot with 5 Pathology Variables",
    cat_c_colors=cat_c_colors,
    group_colors=group_colors,
    format=".png"
)

### Fourth Example: 6 Pathology Variables ###

# Define the variables and their colors for 6 variables
pathology_variables = ["Amyloid", "NFT", "Tangles", "Plaq N", "Age", "Weight"]
cat_c_colors = {
    "Amyloid": "#d5cccd",
    "NFT": "#cb9992",
    "Tangles": "#ad310f",
    "Plaq N": "#7e2a20",
    "Age": "#FFD700",  # Gold color for Age
    "Weight": "#FF6622"  # Orange color for Weight
}

# Create dummy data
np.random.seed(123)
data = pd.DataFrame([(ct, pw) for ct in cell_types for pw in pathways], columns=["CellType", "Pathway"])

# Assign random pathology variables to each combination
data_list = []
for idx, row in data.iterrows():
    n_vars = np.random.randint(1, 7)  # Random number between 1 and 6
    variables = np.random.choice(pathology_variables, size=n_vars, replace=False)
    for var in variables:
        data_list.append({
            "CellType": row["CellType"],
            "Pathway": row["Pathway"],
            "PathologyVariable": var
        })

# Create DataFrame from data_list
data_expanded = pd.DataFrame(data_list)

# Merge the group assignments into the data
data_expanded = data_expanded.merge(pathway_groups, on="Pathway", how="left")

# Use the dice_plot function
dice_plot(
    data=data_expanded,
    cat_a="CellType",
    cat_b="Pathway",
    cat_c="PathologyVariable",
    group="Group",
    plot_path=plot_path,
    output_str="dice_plot_6_example",
    switch_axis=False,
    group_alpha=0.6,
    title="Dice Plot with 6 Pathology Variables",
    cat_c_colors=cat_c_colors,
    group_colors=group_colors,
    format=".png"
)