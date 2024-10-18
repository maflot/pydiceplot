# plot_dice_matplotlib.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import warnings
from pydiceplot.plots.backend._utils import (
    prepare_data,
    perform_clustering,
    calculate_var_positions,
    generate_plot_dimensions
)


def plot_dice_matplotlib(data,
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
                         formats=[".png"],  # Typically image formats for Matplotlib
                         max_dice_sides=6):
    """
    Generates a dice plot visualization using Matplotlib based on the provided data.

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
    - formats: List of file formats for saving the plot (e.g., ['.png']).
    - max_dice_sides: Maximum number of dice sides (1-6).

    Returns:
    - fig: Matplotlib Figure object.
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

    # Generate plot dimensions
    plot_width, plot_height, margins = generate_plot_dimensions(len(cat_a_order), len(cat_b_order))
    box_size = 1  # Assuming each box is 1 unit in size

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100))  # Convert pixels to inches
    ax.set_xlim(0, len(cat_a_order) + 1)
    ax.set_ylim(0, len(cat_b_order) + 1)

    # Add rectangles for the boxes
    for idx, row in box_data.iterrows():
        rect = patches.Rectangle(
            (row['x_min'], row['y_min']),
            row['x_max'] - row['x_min'],
            row['y_max'] - row['y_min'],
            linewidth=0.5,
            edgecolor='grey',
            facecolor=group_colors.get(row[group], '#FFFFFF'),
            alpha=group_alpha
        )
        ax.add_patch(rect)

    # Add scatter points for the PathologyVariables
    for var, color in cat_c_colors.items():
        var_data = plot_data[plot_data[cat_c] == var]
        ax.scatter(
            var_data['x_pos'],
            var_data['y_pos'],
            s=100,  # Marker size
            color=color,
            edgecolors='black',
            label=var
        )

    # Customize axes
    ax.set_xticks(range(1, len(cat_a_order) + 1))
    ax.set_xticklabels(cat_a_order)
    ax.set_yticks(range(1, len(cat_b_order) + 1))
    ax.set_yticklabels(cat_b_order)
    ax.invert_yaxis()  # Match Plotly's default orientation
    ax.set_title(title)
    ax.legend(title=cat_c, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust margins
    plt.subplots_adjust(left=0.2, right=0.75, top=0.8, bottom=0.2)

    # Save the plot in specified formats
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    for fmt in formats:
        file_path = os.path.join(plot_path, f"{output_str}{fmt}")
        plt.savefig(file_path, format=fmt.strip('.'), bbox_inches='tight')

    return fig
