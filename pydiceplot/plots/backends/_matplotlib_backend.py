import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
             group=None,
             switch_axis=False,
             group_alpha=0.6,
             title=None,
             cat_c_colors=None,
             group_colors=None,
             max_dice_sides=6):
    """
    Matplotlib-specific dice plot function.

    Parameters:
    - All parameters as defined in _dice_utils.py's preprocess_dice_plot and additional plotting parameters.

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
    Matplotlib-specific domino plot function.

    Parameters:
    - All parameters as defined in the domino plot specifications.

    Returns:
    - fig: Matplotlib Figure object.
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

    # Create Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(base_width, base_height))

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
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=0.5,
                edgecolor='grey',
                facecolor='white',
                alpha=0.5
            )
            ax.add_patch(rect)

    # Add scatter points
    sc = ax.scatter(
        plot_data['x_pos'],
        plot_data['y_pos'],
        s=plot_data['size'] * 20,  # Adjust marker size
        c=plot_data['adj_logfc'],
        cmap='RdBu_r',
        edgecolors='black'
    )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_scale_name)
    sc.set_clim(logfc_limits)

    # Customize axes
    ax.set_title(title)
    ax.set_xlabel('Genes' if not switch_axis else 'Cell Types')
    ax.set_ylabel('Cell Types' if not switch_axis else 'Genes')
    ax.set_xticks([(i * spacing_factor) + 1.5 for i in range(len(gene_list))])
    ax.set_xticklabels(gene_list, rotation=90)
    ax.set_yticks([i + 1 for i in range(len(unique_celltypes))])
    ax.set_yticklabels(unique_celltypes)
    ax.set_xlim(0, (len(gene_list) - 1) * spacing_factor + 3)
    ax.set_ylim(0, len(unique_celltypes) + 1)

    # Add annotations for contrasts
    for idx, label in enumerate(contrast_labels):
        x_pos = (idx * spacing_factor) + 1.5
        ax.text(
            x_pos,
            len(unique_celltypes) + 1,
            label,
            ha='center',
            va='bottom',
            fontsize=axis_text_size
        )

    # Apply axis switching if needed
    if switch_axis:
        fig = switch_axes_domino(fig, backend='matplotlib')

    # Adjust aspect ratio
    ax.set_aspect(aspect_ratio)

    # Adjust layout
    plt.tight_layout()

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
