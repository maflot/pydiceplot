import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.graph_objects as go
import os

# Utility Functions
def prepare_data(data, cat_a, cat_b, cat_c, group, cat_c_colors, group_colors):
    """
    Prepares the data by setting categorical variables and ordering factors.
    """
    # Ensure consistent ordering of factors
    data[cat_a] = pd.Categorical(data[cat_a], categories=sorted(data[cat_a].unique()), ordered=True)
    data[cat_b] = pd.Categorical(data[cat_b], categories=sorted(data[cat_b].unique()), ordered=True)
    data[cat_c] = pd.Categorical(data[cat_c], categories=list(cat_c_colors.keys()), ordered=True)
    data[group] = pd.Categorical(data[group], categories=list(group_colors.keys()), ordered=True)

    cat_a_order = data[cat_a].cat.categories.tolist()
    cat_b_order = data[cat_b].cat.categories.tolist()

    return data, cat_a_order, cat_b_order


def calculate_var_positions(cat_c_colors, max_dice_sides):
    """
    Calculates positions for dice sides based on the number of variables.
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


def perform_clustering(data, cat_a, cat_b, cat_c):
    """
    Performs hierarchical clustering on the data to order cat_a.
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


def create_binary_matrix(data, cat_a, cat_b, cat_c):
    """
    Creates a binary matrix for clustering.
    """
    data['present'] = 1
    grouped = data.groupby([cat_a, cat_b, cat_c])['present'].sum().reset_index()
    grouped['combined'] = grouped[cat_b].astype(str) + "_" + grouped[cat_c].astype(str)
    binary_matrix_df = grouped.pivot(index=cat_a, columns='combined', values='present').fillna(0)
    return binary_matrix_df


def generate_plot_dimensions(n_x, n_y):
    """
    Generates plot dimensions to make boxes square.
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


def add_rectangles_to_plot(fig, box_data, group, group_colors, group_alpha):
    """
    Adds rectangles to the plot for each combination in box_data.
    """
    for idx, row in box_data.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['x_min'], x1=row['x_max'],
            y0=row['y_min'], y1=row['y_max'],
            line=dict(color="grey", width=0.5),
            fillcolor=group_colors[row[group]],
            opacity=group_alpha,
            layer="below"
        )


def add_scatter_traces(fig, plot_data, cat_c, cat_c_colors):
    """
    Adds scatter traces to the plot for each variable in cat_c.
    """
    for var in cat_c_colors.keys():
        var_data = plot_data[plot_data[cat_c] == var]
        fig.add_trace(go.Scatter(
            x=var_data['x_pos'],
            y=var_data['y_pos'],
            mode='markers',
            marker=dict(
                size=10,
                color=cat_c_colors[var],
                line=dict(width=1, color='black')
            ),
            name=var,
            legendgroup=var,
            showlegend=True
        ))

def save_plot(fig, plot_path, output_str, formats):
    """
    Saves the plot to the specified path with the given formats.

    Parameters:
    - fig: Plotly Figure object.
    - plot_path: Directory path to save the plots.
    - output_str: Base name for the output files.
    - formats: List of file formats (e.g., ['.png', '.html']).
    """
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    for fmt in formats:
        file_path = os.path.join(plot_path, f"{output_str}{fmt}")
        if fmt.lower() == ".html":
            fig.write_html(file_path)
        else:
            # For image formats, ensure that the necessary engine is installed
            # You might need to install kaleido: pip install -U kaleido
            fig.write_image(file_path)
