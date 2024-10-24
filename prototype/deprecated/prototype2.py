# diceplot.py

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.graph_objects as go
import os
import warnings


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


def set_axes(fig, cat_a_order, cat_b_order, switch_axis):
    """
    Sets the axes of the plot.
    """
    x_tickvals = list(range(1, len(cat_a_order) + 1))
    x_ticktext = cat_a_order
    y_tickvals = list(range(1, len(cat_b_order) + 1))
    y_ticktext = cat_b_order

    if switch_axis:
        fig.update_xaxes(
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            tickangle=45,
            tickmode='array',
            title_text='',
            showgrid=False,
            zeroline=False,
            range=[0.5, len(y_tickvals) + 0.5]
        )
        fig.update_yaxes(
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            tickmode='array',
            title_text='',
            showgrid=False,
            zeroline=False,
            range=[0.5, len(x_tickvals) + 0.5],
            scaleanchor="x",
            scaleratio=1
        )
    else:
        fig.update_xaxes(
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            tickangle=45,
            tickmode='array',
            title_text='',
            showgrid=False,
            zeroline=False,
            range=[0.5, len(x_tickvals) + 0.5]
        )
        fig.update_yaxes(
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            tickmode='array',
            title_text='',
            showgrid=False,
            zeroline=False,
            range=[0.5, len(y_tickvals) + 0.5],
            scaleanchor="x",
            scaleratio=1
        )
    return fig


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


# Main Dice Plot Function
def dice_plot(data,
              cat_a,
              cat_b,
              cat_c,
              group,
              plot_path="./",
              output_str="dice_plot",
              switch_axis=False,
              group_alpha=0.6,
              title=None,
              cat_c_colors=None,
              group_colors=None,
              formats=[".html", ".png"],  # Modified to accept multiple formats
              max_dice_sides=6):
    """
    Generates a dice plot visualization based on the provided data.

    Parameters:
    - data: DataFrame containing the necessary variables.
    - cat_a: Name of the category A variable (e.g., 'CellType').
    - cat_b: Name of the category B variable (e.g., 'Pathway').
    - cat_c: Name of the category C variable (e.g., 'PathologyVariable').
    - group: Name of the grouping variable (e.g., 'Group').
    - plot_path: Path to save the plot.
    - output_str: Output string for the filename.
    - switch_axis: Whether to switch the axes.
    - group_alpha: Transparency level for group rectangles.
    - title: Plot title.
    - cat_c_colors: Dictionary of colors for cat_c variables.
    - group_colors: Dictionary of colors for group variables.
    - formats: List of file formats for saving the plot (e.g., ['.html', '.png']).
    - max_dice_sides: Maximum number of dice sides (1-6).

    Returns:
    - fig: Plotly Figure object.
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

    # Create the figure
    fig = go.Figure()

    # Add rectangles for the boxes
    add_rectangles_to_plot(fig, box_data, group, group_colors, group_alpha)

    # Add scatter traces for the PathologyVariables
    add_scatter_traces(fig, plot_data, cat_c, cat_c_colors)

    # Set axes
    fig = set_axes(fig, cat_a_order, cat_b_order, switch_axis)

    # Calculate dynamic width and height to make boxes square
    plot_width, plot_height, margins = generate_plot_dimensions(len(cat_a_order), len(cat_b_order))

    # Set layout
    fig.update_layout(
        plot_bgcolor='white',
        title=title,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.05
        ),
        margin=margins,
        width=plot_width,
        height=plot_height
    )

    # Save the plot in specified formats
    save_plot(fig, plot_path, output_str, formats)

    return fig


# Adapted Main Function with Multiple Examples
if __name__ == "__main__":
    plot_path = "../plots"

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
            plot_path=plot_path,
            output_str=output_str,
            switch_axis=False,
            group_alpha=0.6,
            title=title,
            cat_c_colors=cat_c_colors,
            group_colors=group_colors,
            formats=[".html", ".png"],  # Save as both HTML and PNG
            max_dice_sides=6  # Adjust if needed
        )

        # Optionally display the figure
        # fig.show()


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