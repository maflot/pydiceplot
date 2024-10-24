import pandas as pd
import numpy as np
import warnings
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

def prepare_data(data, cat_a, cat_b, cat_c, group, cat_c_colors, group_colors):
    """
    Prepares the data by setting categorical variables and ordering factors.

    Parameters:
    - data: DataFrame containing the necessary variables.
    - cat_a, cat_b, cat_c, group: Column names for categories and grouping.
    - cat_c_colors, group_colors: Dictionaries for category colors.

    Returns:
    - data: Updated DataFrame with categorical types.
    - cat_a_order: List of ordered categories for cat_a.
    - cat_b_order: List of ordered categories for cat_b.
    """
    # Ensure consistent ordering of factors
    data[cat_a] = pd.Categorical(
        data[cat_a],
        categories=sorted(data[cat_a].unique()),
        ordered=True
    )
    data[cat_b] = pd.Categorical(
        data[cat_b],
        categories=sorted(data[cat_b].unique()),
        ordered=True
    )
    data[cat_c] = pd.Categorical(
        data[cat_c],
        categories=list(cat_c_colors.keys()),
        ordered=True
    )
    if group is not None:
        data[group] = pd.Categorical(
        data[group],
        categories=list(group_colors.keys()),
        ordered=True
        )

    cat_a_order = data[cat_a].cat.categories.tolist()
    cat_b_order = data[cat_b].cat.categories.tolist()

    return data, cat_a_order, cat_b_order

def create_binary_matrix(data, cat_a, cat_b, cat_c):
    """
    Creates a binary matrix for clustering.

    Parameters:
    - data: DataFrame containing the necessary variables.
    - cat_a, cat_b, cat_c: Column names for categories.

    Returns:
    - binary_matrix_df: Binary matrix DataFrame.
    """
    data['present'] = 1
    grouped = data.groupby([cat_a, cat_b, cat_c])['present'].sum().reset_index()
    grouped['combined'] = grouped[cat_b].astype(str) + "_" + grouped[cat_c].astype(str)
    binary_matrix_df = grouped.pivot(
        index=cat_a,
        columns='combined',
        values='present'
    ).fillna(0)
    return binary_matrix_df

def perform_clustering(data, cat_a, cat_b, cat_c, group):
    """
    Performs hierarchical clustering on the data to order cat_b within each group.

    Parameters:
    - data: DataFrame containing the necessary variables.
    - cat_a, cat_b, cat_c, group: Column names for categories.
    - cluster_on: Specify whether to cluster on 'cat_a' or 'cat_b'.

    Returns:
    - cat_order: List of ordered categories for cat_b based on clustering within groups.
    """
    cat_order = []
    groups = data[group].cat.categories.tolist()
    for grp in groups:
        data_grp = data[data[group] == grp]
        # Create a binary matrix for clustering within the group
        # Pivot the data to have 'cat_b' as rows
        data_grp['present'] = 1
        grouped = data_grp.groupby([cat_b, cat_a, cat_c])['present'].sum().reset_index()
        grouped['combined'] = grouped[cat_a].astype(str) + "_" + grouped[cat_c].astype(str)
        binary_matrix_df = grouped.pivot(
            index=cat_b,
            columns='combined',
            values='present'
        ).fillna(0)
        binary_matrix = binary_matrix_df.values

        # Perform hierarchical clustering within the group
        cat_b_list = binary_matrix_df.index.tolist()
        if binary_matrix.shape[0] > 1:
            distance_matrix = pdist(binary_matrix, metric='jaccard')
            Z = linkage(distance_matrix, method='ward')
            dendro = dendrogram(Z, labels=cat_b_list, no_plot=True)
            cat_grp_order = dendro['ivl']  # No need to reverse for rows
        else:
            cat_grp_order = cat_b_list

        cat_order.extend(cat_grp_order)

    # Remove duplicates while preserving order
    from collections import OrderedDict
    cat_order = list(OrderedDict.fromkeys(cat_order))

    return cat_order

def calculate_var_positions(cat_c_colors, max_dice_sides):
    """
    Calculates positions for dice sides based on the number of variables.

    Parameters:
    - cat_c_colors: Dictionary of colors for cat_c variables.
    - max_dice_sides: Maximum number of dice sides (1-6).

    Returns:
    - var_positions: DataFrame with variable positions.
    """
    num_vars = len(cat_c_colors)
    if num_vars > max_dice_sides:
        raise ValueError(f"Number of variables ({num_vars}) exceeds max_dice_sides ({max_dice_sides}).")

    # Define positions for up to 6 sides (dice faces)
    positions_dict = {
        1: [(0, 0)],
        2: [(-0.2, 0), (0.2, 0)],
        3: [(-0.2, -0.2), (0, 0.2), (0.2, -0.2)],
        4: [(-0.2, -0.2), (-0.2, 0.2), (0.2, -0.2), (0.2, 0.2)],
        5: [(-0.2, -0.2), (-0.2, 0.2), (0, 0), (0.2, -0.2), (0.2, 0.2)],
        6: [(-0.2, -0.3), (-0.2, 0), (-0.2, 0.3), (0.2, -0.3), (0.2, 0), (0.2, 0.3)]
    }

    positions = positions_dict[num_vars]
    var_positions = pd.DataFrame({
        'var': list(cat_c_colors.keys()),
        'x_offset': [pos[0] for pos in positions],
        'y_offset': [pos[1] for pos in positions]
    })
    return var_positions

def generate_plot_dimensions(n_x, n_y, n_dice):
    """
    Generates plot dimensions to make boxes square, adjusting size based on the number of dice sides.

    Parameters:
    - n_x: Number of categories along the x-axis.
    - n_y: Number of categories along the y-axis.
    - n_dice: Number of sides (eyes) the dice is showing.

    Returns:
    - plot_width: Width of the plot in pixels.
    - plot_height: Height of the plot in pixels.
    - margins: Dictionary with plot margins.
    """
    # Base box size
    base_box_size = 50  # pixels per box

    # Adjust box size based on n_dice
    # Increase box size slightly for higher n_dice to accommodate more details
    box_size = base_box_size + (n_dice - 1) * 5

    # Adjust margins based on n_dice
    # Larger n_dice might require more space for labels and dice representation
    margin_l = 150 + (n_dice - 1) * 10
    margin_r = 300 + (n_dice - 1) * 10
    margin_t = 100 + (n_dice - 1) * 10
    margin_b = 200 + (n_dice - 1) * 10

    # Calculate plot dimensions
    plot_width = box_size * n_x + margin_l + margin_r
    plot_height = box_size * n_y + margin_t + margin_b
    margins = dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b)
    return plot_width, plot_height, margins


def preprocess_dice_plot(data, cat_a, cat_b, cat_c, group=None, cat_c_colors=None, group_colors=None, max_dice_sides=6):
    """
    Preprocesses data for dice plot generation with handling for None group.

    Parameters:
    - data: DataFrame containing the necessary variables.
    - cat_a, cat_b, cat_c: Column names for categories.
    - group: Optional, column name for grouping (can be None).
    - cat_c_colors, group_colors: Dictionaries for category colors.
    - max_dice_sides: Maximum number of dice sides.

    Returns:
    - plot_data: DataFrame prepared for plotting.
    - box_data: DataFrame with box dimensions.
    - cat_a_order: Ordered categories for cat_a.
    - cat_b_order: Ordered categories for cat_b.
    - var_positions: DataFrame with variable positions.
    - plot_dimensions: Tuple with plot width, height, and margins.
    """
    # Prepare data and ensure consistent ordering
    data, cat_a_order, cat_b_order = prepare_data(
        data, cat_a, cat_b, cat_c, group, cat_c_colors, group_colors
    )

    if group is not None:
        # Check for unique group per cat_b
        group_check = data.groupby(cat_b)[group].nunique().reset_index()
        group_check = group_check[group_check[group] > 1]
        if not group_check.empty:
            warnings.warn("Warning: The following cat_b categories have multiple groups assigned:\n{}".format(
                ', '.join(group_check[cat_b].tolist())
            ))

    # Calculate variable positions for dice sides
    var_positions = calculate_var_positions(cat_c_colors, max_dice_sides)

    if group is not None:
        # Perform hierarchical clustering to order cat_a if group is present
        cat_b_order = perform_clustering(data, cat_a, cat_b, cat_c, group)
        data[cat_b] = pd.Categorical(data[cat_b], categories=cat_b_order, ordered=True)

    # Update plot_data
    plot_data = data.merge(var_positions, left_on=cat_c, right_on='var', how='left')
    plot_data = plot_data.dropna(subset=['x_offset', 'y_offset'])
    plot_data['x_num'] = plot_data[cat_a].cat.codes + 1
    plot_data['y_num'] = plot_data[cat_b].cat.codes + 1
    plot_data['x_pos'] = plot_data['x_num'] + plot_data['x_offset']
    plot_data['y_pos'] = plot_data['y_num'] + plot_data['y_offset']

    if group is not None:
        plot_data = plot_data.sort_values(by=[cat_a, group, cat_b])
    else:
        plot_data = plot_data.sort_values(by=[cat_a, cat_b])

    # Prepare box_data
    if group is not None:
        box_data = data[[cat_a, cat_b, group]].drop_duplicates()
    else:
        box_data = data[[cat_a, cat_b]].drop_duplicates()
        box_data[group] = None  # Add a dummy column for consistency

    box_data['x_num'] = box_data[cat_a].cat.codes + 1
    box_data['y_num'] = box_data[cat_b].cat.codes + 1
    box_data['x_min'] = box_data['x_num'] - 0.4
    box_data['x_max'] = box_data['x_num'] + 0.4
    box_data['y_min'] = box_data['y_num'] - 0.4
    box_data['y_max'] = box_data['y_num'] + 0.4

    if group is not None:
        box_data = box_data.sort_values(by=[cat_a, group, cat_b])
    else:
        box_data = box_data.sort_values(by=[cat_a, cat_b])

    # Generate plot dimensions
    plot_dimensions = generate_plot_dimensions(len(cat_a_order), len(cat_b_order), len(cat_c_colors))

    return plot_data, box_data, cat_a_order, cat_b_order, var_positions, plot_dimensions


def get_diceplot_example_data(n):
    """
    Returns a DataFrame suitable for creating a dice plot with n pathology variables.

    Parameters:
    n (int): Number of pathology variables (1 <= n <= 6)

    Returns:
    DataFrame: DataFrame containing the data for the dice plot, including 'Group' variable.
    """
    # Ensure n is between 1 and 6
    if n < 1 or n > 6:
        raise ValueError("n must be between 1 and 6")

    # Define cell types
    cell_types = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", "Endothelial"]

    # Define pathways (ensure 15 pathways)
    pathways_extended = [
        "Apoptosis", "Inflammation", "Metabolism", "Signal Transduction", "Synaptic Transmission",
        "Cell Cycle", "DNA Repair", "Protein Synthesis", "Lipid Metabolism", "Neurotransmitter Release",
        "Oxidative Stress", "Energy Production", "Calcium Signaling", "Synaptic Plasticity", "Immune Response"
    ]

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

    # Define pathology variables for n=6
    pathology_vars_6 = ["Alzheimer's disease", "Cancer", "Flu", "ADHD", "Age", "Weight"]

    # Use the first n variables
    pathology_vars = pathology_vars_6[:n]

    # Create dummy data
    np.random.seed(123)
    data = pd.DataFrame([(ct, pw) for ct in cell_types for pw in pathways_extended[:15]],
                        columns=["CellType", "Pathway"])

    # Merge the group assignments into the data
    data = data.merge(pathway_groups, on="Pathway", how="left")

    # Assign random pathology variables to each combination
    data_list = []
    for idx, row in data.iterrows():
        # For each cell type and pathway, randomly assign between 1 and n pathology variables
        variables = np.random.choice(pathology_vars, size=np.random.randint(1, n + 1), replace=False)
        for var in variables:
            data_list.append({
                "CellType": row["CellType"],
                "Pathway": row["Pathway"],
                "Group": row["Group"],
                "PathologyVariable": var
            })

    # Create DataFrame from data_list
    data_expanded = pd.DataFrame(data_list)

    return data_expanded


def get_example_group_colors():
    """
        Returns a Colorpalette fitting the example dataframe
    """
    return {
        "Linked": "#333333",
        "UnLinked": "#888888",
        "Other": "#DDDDDD"
    }

def get_example_cat_c_colors():
    """
        Returns a Colorpalette fitting the example dataframe
    """
    return {
        "Alzheimer's disease": "#d5cccd",
        "Cancer": "#cb9992",
        "Flu": "#ad310f",
        "ADHD": "#7e2a20",
        "Age": "#FFD700",
        "Weight": "#FF6622"
    }