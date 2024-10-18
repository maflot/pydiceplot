import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from ._utils import preprocess_dice_plot


def plot_dice(data,
             cat_a,
             cat_b,
             cat_c,
             group,
             switch_axis=False,
             group_alpha=0.6,
             title=None,
             cat_c_colors=None,
             group_colors=None,
             max_dice_sides=6):
    """
    Matplotlib-specific dice plot function.

    Parameters:
    - All parameters as defined in _utils.py's preprocess_dice_plot and additional plotting parameters.

    Returns:
    - fig: Matplotlib Figure object.
    """
    # Preprocess data
    plot_data, box_data, cat_a_order, cat_b_order, var_positions, plot_dimensions = preprocess_dice_plot(
        data, cat_a, cat_b, cat_c, group, cat_c_colors, group_colors, max_dice_sides
    )

    # Handle axis switching
    if switch_axis:
        cat_a_order, cat_b_order = cat_b_order, cat_a_order
        plot_data = plot_data.rename(columns={'x_num': 'y_num', 'y_num': 'x_num',
                                              'x_pos': 'y_pos', 'y_pos': 'x_pos'})
        box_data = box_data.rename(columns={'x_num': 'y_num', 'y_num': 'x_num',
                                           'x_min': 'y_min', 'x_max': 'y_max'})

    # Unpack plot dimensions
    plot_width, plot_height, margins = plot_dimensions

    # Create Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100))  # Convert pixels to inches
    ax.set_xlim(0, len(cat_a_order) + 1)
    ax.set_ylim(0, len(cat_b_order) + 1)

    # Add rectangles for the boxes
    for _, row in box_data.iterrows():
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

    return fig


def show_plot(fig):
    """
    Displays the Matplotlib figure.

    Parameters:
    - fig: Matplotlib Figure object.
    """
    plt.show()


def save_plot(fig, plot_path, output_str, formats):
    """
    Saves the Matplotlib figure in specified formats.

    Parameters:
    - fig: Matplotlib Figure object.
    - plot_path: Directory path to save the plots.
    - output_str: Base name for the output files.
    - formats: List of file formats (e.g., ['.png']).
    """
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    for fmt in formats:
        file_path = os.path.join(plot_path, f"{output_str}{fmt}")
        fig.savefig(file_path, format=fmt.strip('.'), bbox_inches='tight')
