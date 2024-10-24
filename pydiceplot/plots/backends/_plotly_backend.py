import plotly.graph_objects as go
import os
from ._dice_utils import (
    preprocess_dice_plot,
)
from ._domino_utils import (
    preprocess_domino_plot,
    switch_axes_domino
)


def plot_dice(data,
              cat_a,
              cat_b,
              cat_c,
              group=None,  # Default to None
              switch_axis=False,
              group_alpha=0.6,
              title=None,
              cat_c_colors=None,
              group_colors=None,
              max_dice_sides=6):
    """
    Adapted Plotly-specific dice plot function to resemble Matplotlib's style, with handling for None group.

    Parameters:
    - All parameters as defined in _dice_utils.py's preprocess_dice_plot and additional plotting parameters.

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
        # If group is None, use white color for the boxes, otherwise use group_colors
        fill_color = "#FFFFFF" if group is None else group_colors.get(row[group], "#FFFFFF")
        fig.add_shape(
            type="rect",
            x0=row['x_min'],
            y0=row['y_min'],
            x1=row['x_max'],
            y1=row['y_max'],
            line=dict(color="grey", width=0.5),
            fillcolor=fill_color,
            opacity=group_alpha,
            layer="below"
        )

    # Add scatter points for the PathologyVariables
    for var, color in cat_c_colors.items():
        var_data = plot_data[plot_data[cat_c] == var]
        fig.add_trace(go.Scatter(
            x=var_data['x_pos'],
            y=var_data['y_pos'],
            mode='markers',
            marker=dict(
                size=10,
                color=color,
                line=dict(width=1, color='black')  # Similar to Matplotlib edgecolors
            ),
            name=var,
            legendgroup=var,
            showlegend=True
        ))

    # Set axis limits and labels, similar to Matplotlib
    fig.update_xaxes(
        range=[0, len(cat_a_order) + 1],
        tickvals=list(range(1, len(cat_a_order) + 1)),
        ticktext=cat_a_order,
        showgrid=False
    )
    fig.update_yaxes(
        range=[0, len(cat_b_order) + 1],
        tickvals=list(range(1, len(cat_b_order) + 1)),
        ticktext=cat_b_order,
        autorange="reversed",  # Invert y-axis to match Matplotlib
        showgrid=False
    )

    # Update layout to simplify the style and match Matplotlib's
    fig.update_layout(
        plot_bgcolor='white',
        title=title,
        legend=dict(
            title=cat_c,
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.05
        ),
        margin=dict(l=100, r=100, t=80, b=100),  # Match Matplotlib-like margins
        width=plot_width,
        height=plot_height
    )

    return fig

def plot_domino(data,
                gene_list,
                switch_axis=False,
                min_dot_size=1,
                max_dot_size=5,
                spacing_factor=3,
                var_id="var",
                feature_col="gene",
                celltype_col="Celltype",
                contrast_col="Contrast",
                contrast_levels=["Clinical", "Pathological"],
                contrast_labels=["Clinical", "Pathological"],
                logfc_col="avg_log2FC",
                pval_col="p_val_adj",
                logfc_limits=(-1.5, 1.5),
                logfc_colors={"low": "blue", "mid": "white", "high": "red"},
                color_scale_name="Log2 Fold Change",
                axis_text_size=8,
                aspect_ratio=None,
                base_width=5,
                base_height=4,
                title=None):
    """
    Plotly-specific domino plot function.

    Parameters:
    - All parameters as defined in the domino plot specifications.

    Returns:
    - fig: Plotly Figure object.
    """
    # Preprocess data
    plot_data, calculated_aspect_ratio, unique_celltypes, unique_genes = preprocess_domino_plot(
        data,
        gene_list,
        spacing_factor,
        contrast_levels,
        feature_col,
        celltype_col,
        contrast_col,
        var_id,
        logfc_col,
        pval_col,
        logfc_limits,
        min_dot_size,
        max_dot_size
    )

    # Use provided aspect_ratio if given, else use calculated
    if aspect_ratio is None:
        aspect_ratio = calculated_aspect_ratio

    # Create Plotly figure
    fig = go.Figure()

    # Add rectangles for each gene-celltype pair
    unique_pairs = plot_data[[feature_col, celltype_col]].drop_duplicates()
    for _, row in unique_pairs.iterrows():
        gene_idx = gene_list.index(row[feature_col]) + 1
        celltype_idx = unique_celltypes.index(row[celltype_col]) + 1
        y_min = celltype_idx - 0.4
        y_max = celltype_idx + 0.4

        # Add rectangles for each contrast
        for idx, contrast in enumerate(contrast_levels):
            if contrast == contrast_levels[0]:
                x_min = (gene_idx - 1) * spacing_factor + 1 - 0.4
                x_max = (gene_idx - 1) * spacing_factor + 1 + 0.4
            else:
                x_min = (gene_idx - 1) * spacing_factor + 2 - 0.4
                x_max = (gene_idx - 1) * spacing_factor + 2 + 0.4
            fig.add_shape(
                type="rect",
                x0=x_min,
                y0=y_min,
                x1=x_max,
                y1=y_max,
                line=dict(color="grey", width=0.5),
                fillcolor="white",
                opacity=0.5,
                layer="below"
            )

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=plot_data['x_pos'],
        y=plot_data['y_pos'],
        mode='markers',
        marker=dict(
            size=plot_data['size'],
            color=plot_data['adj_logfc'],
            colorscale=[[0, logfc_colors["low"]], [0.5, logfc_colors["mid"]], [1, logfc_colors["high"]]],
            cmin=logfc_limits[0],
            cmax=logfc_limits[1],
            colorbar=dict(title=color_scale_name),
            line=dict(width=1, color='black')
        ),
        text=plot_data[var_id],
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='Genes',
            tickmode='array',
            tickvals=[(i * spacing_factor) + 1.5 for i in range(len(gene_list))],
            ticktext=gene_list,
            showgrid=True
        ),
        yaxis=dict(
            title='Cell Types',
            tickmode='array',
            tickvals=[i + 1 for i in range(len(unique_celltypes))],
            ticktext=unique_celltypes,
            showgrid=True
        ),
        width=base_width * 100,
        height=base_height * 100,
        margin=dict(l=50, r=200, t=100, b=50),
        aspectratio=dict(x=1, y=aspect_ratio)
    )

    # Add contrast annotations
    for idx, label in enumerate(contrast_labels):
        x_pos = (idx * spacing_factor) + 1.5
        fig.add_annotation(
            x=x_pos,
            y=len(unique_celltypes) + 1,
            text=label,
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=axis_text_size)
        )

    # Apply axis switching if needed
    if switch_axis:
        fig = switch_axes_domino(fig, backend='plotly')

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
