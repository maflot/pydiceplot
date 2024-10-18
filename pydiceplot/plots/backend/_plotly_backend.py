# plot_dice_plotly.py

import plotly.graph_objects as go
import os
import warnings
from _utils import (
    prepare_data,
    perform_clustering,
    calculate_var_positions,
    generate_plot_dimensions,
    add_rectangles_to_plot,
    add_scatter_traces,
    save_plot
)

def plot_dice_plotly(data,
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
                     formats=[".html", ".png"],  # Typically interactive and image formats
                     max_dice_sides=6):
    """
    Generates a dice plot visualization using Plotly based on the provided data.

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

    # Handle axis switching if required
    if switch_axis:
        cat_a_order, cat_b_order = cat_b_order, cat_a_order
        plot_data = plot_data.rename(columns={'x_num': 'y_num', 'y_num': 'x_num',
                                              'x_pos': 'y_pos', 'y_pos': 'x_pos'})
        box_data = box_data.rename(columns={'x_num': 'y_num', 'y_num': 'x_num',
                                           'x_min': 'y_min', 'x_max': 'y_max'})

    # Create the figure
    fig = go.Figure()

    # Add rectangles for the boxes using utility function
    add_rectangles_to_plot(fig, box_data, group, group_colors, group_alpha)

    # Add scatter traces for the PathologyVariables using utility function
    add_scatter_traces(fig, plot_data, cat_c, cat_c_colors)

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
