# _utils.py

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

def perform_clustering(data, cat_a, cat_b, cat_c):
    """
    Performs hierarchical clustering on the data to order cat_a.

    Parameters:
    - data: DataFrame containing the necessary variables.
    - cat_a, cat_b, cat_c: Column names for categories.

    Returns:
    - cat_a_order: List of ordered categories for cat_a based on clustering.
    """
    # Create a binary matrix for clustering
    binary_matrix_df = create_binary_matrix(data, cat_a, cat_b, cat_c)
    binary_matrix = binary_matrix_df.values

    # Perform hierarchical clustering
    cell_types = binary_matrix_df.index.tolist()
    if binary_matrix.shape[0] > 1:
        distance_matrix = pdist(binary_matrix, metric='jaccard')
        Z = linkage(distance_matrix, method='ward')
        dendro = dendrogram(Z, labels=cell_types, no_plot=True)
        cat_a_order = dendro['ivl'][::-1]  # Reverse to match desired order
    else:
        cat_a_order = cell_types

    return cat_a_order

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

def generate_plot_dimensions(n_x, n_y):
    """
    Generates plot dimensions to make boxes square.

    Parameters:
    - n_x: Number of categories along the x-axis.
    - n_y: Number of categories along the y-axis.

    Returns:
    - plot_width: Width of the plot in pixels.
    - plot_height: Height of the plot in pixels.
    - margins: Dictionary with plot margins.
    """
    box_size = 50  # pixels per box
    margin_l = 150
    margin_r = 300
    margin_t = 100
    margin_b = 200
    plot_width = box_size * n_x + margin_l + margin_r
    plot_height = box_size * n_y + margin_t + margin_b
    margins = dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b)
    return plot_width, plot_height, margins

def preprocess_dice_plot(data, cat_a, cat_b, cat_c, group, cat_c_colors, group_colors, max_dice_sides):
    """
    Preprocesses data for dice plot generation.

    Parameters:
    - data: DataFrame containing the necessary variables.
    - cat_a, cat_b, cat_c, group: Column names for categories and grouping.
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

    # Check for unique group per cat_b
    group_check = data.groupby(cat_b)[group].nunique().reset_index()
    group_check = group_check[group_check[group] > 1]
    if not group_check.empty:
        warnings.warn("Warning: The following cat_b categories have multiple groups assigned:\n{}".format(
            ', '.join(group_check[cat_b].tolist())
        ))

    # Calculate variable positions for dice sides
    var_positions = calculate_var_positions(cat_c_colors, max_dice_sides)

    # Perform hierarchical clustering to order cat_a
    cat_a_order = perform_clustering(data, cat_a, cat_b, cat_c)
    data[cat_a] = pd.Categorical(data[cat_a], categories=cat_a_order, ordered=True)

    # Update plot_data
    plot_data = data.merge(var_positions, left_on=cat_c, right_on='var', how='left')
    plot_data = plot_data.dropna(subset=['x_offset', 'y_offset'])
    plot_data['x_num'] = plot_data[cat_a].cat.codes + 1
    plot_data['y_num'] = plot_data[cat_b].cat.codes + 1
    plot_data['x_pos'] = plot_data['x_num'] + plot_data['x_offset']
    plot_data['y_pos'] = plot_data['y_num'] + plot_data['y_offset']
    plot_data = plot_data.sort_values(by=[cat_a, group, cat_b])

    # Prepare box_data
    box_data = data[[cat_a, cat_b, group]].drop_duplicates()
    box_data['x_num'] = box_data[cat_a].cat.codes + 1
    box_data['y_num'] = box_data[cat_b].cat.codes + 1
    box_data['x_min'] = box_data['x_num'] - 0.4
    box_data['x_max'] = box_data['x_num'] + 0.4
    box_data['y_min'] = box_data['y_num'] - 0.4
    box_data['y_max'] = box_data['y_num'] + 0.4
    box_data = box_data.sort_values(by=[cat_a, group, cat_b])

    # Generate plot dimensions
    plot_dimensions = generate_plot_dimensions(len(cat_a_order), len(cat_b_order))

    return plot_data, box_data, cat_a_order, cat_b_order, var_positions, plot_dimensions
