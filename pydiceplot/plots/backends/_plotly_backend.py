import plotly.graph_objects as go
import os
from ._utils import (
    preprocess_dice_plot,
)


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
    Plotly-specific dice plot function.

    Parameters:
    - All parameters as defined in _utils.py's preprocess_dice_plot and additional plotting parameters.

    Returns:
    - fig: Plotly Figure object.
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

    # Create Plotly figure
    fig = go.Figure()

    # Add rectangles for the boxes
    for _, row in box_data.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['x_min'],
            y0=row['y_min'],
            x1=row['x_max'],
            y1=row['y_max'],
            line=dict(color="grey", width=0.5),
            fillcolor=group_colors.get(row[group], "#FFFFFF"),
            opacity=group_alpha,
            layer="below"
        )

    # Add scatter traces for the PathologyVariables
    for var, color in cat_c_colors.items():
        var_data = plot_data[plot_data[cat_c] == var]
        fig.add_trace(go.Scatter(
            x=var_data['x_pos'],
            y=var_data['y_pos'],
            mode='markers',
            marker=dict(
                size=10,
                color=color,
                line=dict(width=1, color='black')
            ),
            name=var,
            legendgroup=var,
            showlegend=True
        ))

    # Update layout
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

    return fig


def show_plot(fig):
    """
    Displays the Plotly figure.

    Parameters:
    - fig: Plotly Figure object.
    """
    fig.show()


def save_plot(fig, plot_path, output_str, formats):
    """
    Saves the Plotly figure in specified formats.

    Parameters:
    - fig: Plotly Figure object.
    - plot_path: Directory path to save the plots.
    - output_str: Base name for the output files.
    - formats: List of file formats (e.g., ['.html', '.png']).
    """
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    for fmt in formats:
        file_path = os.path.join(plot_path, f"{output_str}{fmt}")
        if fmt.lower() == ".html":
            fig.write_html(file_path)
        else:
            fig.write_image(file_path)
