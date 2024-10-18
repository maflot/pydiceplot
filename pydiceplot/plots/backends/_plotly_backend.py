import plotly.graph_objects as go
import os
from _utils import preprocess_dice_plot


def plot_dice(data,
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
              formats=[".html", ".png"],
              max_dice_sides=6):
    """
    Plotly-specific dice plot function.

    Parameters:
    - All parameters are identical to those in the general plot_dice function.

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

    # Create figure
    fig = go.Figure()

    # Add rectangles
    for _, row in box_data.iterrows():
        fig.add_shape(
            type="rect",
            x0=row['x_min'], x1=row['x_max'],
            y0=row['y_min'], y1=row['y_max'],
            line=dict(color="grey", width=0.5),
            fillcolor=group_colors.get(row[group], '#FFFFFF'),
            opacity=group_alpha,
            layer="below"
        )

    # Add scatter points
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

    # Customize layout
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

    # Save the plot
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

    return fig
