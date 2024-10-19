import pandas as pd
import numpy as np

def filter_genes_domino(data, gene_list, feature_col):
    """
    Filters the dataset based on a list of genes.

    Parameters:
    - data (pd.DataFrame): The input data.
    - gene_list (list): List of genes to include.
    - feature_col (str): Column name representing genes.

    Returns:
    - pd.DataFrame: Filtered data.
    """
    filtered_data = data[data[feature_col].isin(gene_list)].copy()
    return filtered_data

def create_complete_dataset_domino(data, gene_list, celltype_col, contrast_col, var_id, feature_col):
    """
    Ensures that all combinations of genes, cell types, contrasts, and variables are present.

    Parameters:
    - data (pd.DataFrame): Filtered data.
    - gene_list (list): List of genes.
    - celltype_col (str): Column name for cell types.
    - contrast_col (str): Column name for contrasts.
    - var_id (str): Column name for variables.
    - feature_col (str): Column name for genes.

    Returns:
    - pd.DataFrame: Complete dataset with all combinations.
    """
    all_celltypes = data[celltype_col].unique()
    all_contrasts = data[contrast_col].unique()
    all_vars = data[var_id].unique()

    # Create a complete multi-index of all combinations
    complete_data = pd.MultiIndex.from_product(
        [gene_list, all_celltypes, all_contrasts, all_vars],
        names=[feature_col, celltype_col, contrast_col, var_id]
    ).to_frame(index=False)

    # Merge with original data to fill in existing values
    complete_data = complete_data.merge(data, on=[feature_col, celltype_col, contrast_col, var_id], how='left')

    return complete_data

def calculate_positions_domino(data, spacing_factor, contrast_levels, feature_col, celltype_col, contrast_col):
    """
    Calculates x and y positions for each data point.

    Parameters:
    - data (pd.DataFrame): Complete dataset.
    - spacing_factor (int): Spacing between gene pairs.
    - contrast_levels (list): List of contrast levels.
    - feature_col (str): Column name for genes.
    - celltype_col (str): Column name for cell types.
    - contrast_col (str): Column name for contrasts.

    Returns:
    - pd.DataFrame: Data with position information.
    """
    # Assign numerical indices to genes
    data['gene_index'] = data[feature_col].astype('category').cat.codes + 1

    # Assign numerical indices to cell types
    data['celltype_num'] = data[celltype_col].astype('category').cat.codes + 1

    # Calculate x positions based on contrast
    data['x_pos'] = np.where(
        data[contrast_col] == contrast_levels[0],
        (data['gene_index'] - 1) * spacing_factor + 1 - 0.2,  # Offset for first contrast
        (data['gene_index'] - 1) * spacing_factor + 2 + 0.2   # Offset for second contrast
    )

    # Calculate y positions with offsets
    data['y_pos'] = data['celltype_num'] + np.where(
        data[contrast_col] == contrast_levels[0],
        0.2,   # Offset for first contrast
        -0.2   # Offset for second contrast
    )

    return data

def calculate_aspect_ratio_domino(n_genes, n_celltypes, spacing_factor=3, base_width=5, base_height=4):
    """
    Calculates the aspect ratio for the plot to ensure visual consistency.

    Parameters:
    - n_genes (int): Number of genes.
    - n_celltypes (int): Number of cell types.
    - spacing_factor (int): Spacing between gene pairs.
    - base_width (float): Base width in inches.
    - base_height (float): Base height in inches.

    Returns:
    - aspect_ratio (float): Calculated aspect ratio.
    """
    aspect_ratio = (n_celltypes * 0.4) / (n_genes * spacing_factor * 0.5)
    return aspect_ratio

def apply_color_scale_domino(data, logfc_col, logfc_limits):
    """
    Applies a color scale based on log fold change values.

    Parameters:
    - data (pd.DataFrame): Data to be colored.
    - logfc_col (str): Column name for log fold change.
    - logfc_limits (tuple): Limits for the color scale.

    Returns:
    - pd.Series: Adjusted log fold change values for coloring.
    """
    data['adj_logfc'] = data[logfc_col].clip(lower=logfc_limits[0], upper=logfc_limits[1])
    return data['adj_logfc']

def apply_size_scale_domino(data, pval_col, min_dot_size, max_dot_size):
    """
    Applies a size scale based on adjusted p-values.

    Parameters:
    - data (pd.DataFrame): Data to be sized.
    - pval_col (str): Column name for adjusted p-values.
    - min_dot_size (float): Minimum dot size.
    - max_dot_size (float): Maximum dot size.

    Returns:
    - pd.Series: Size values.
    """
    data['log_pval'] = -np.log10(data[pval_col].replace(0, np.nan))  # Avoid log(0)
    data['log_pval'].fillna(data['log_pval'].max(), inplace=True)  # Replace NaNs with max value
    # Normalize log_pval to a range between min_dot_size and max_dot_size
    min_log_pval = data['log_pval'].min()
    max_log_pval = data['log_pval'].max()
    data['size'] = ((data['log_pval'] - min_log_pval) / (max_log_pval - min_log_pval)) * (max_dot_size - min_dot_size) + min_dot_size
    return data['size']

def switch_axes_domino(fig, backend):
    """
    Switches the x and y axes of the plot.

    Parameters:
    - fig: Plot object (Plotly Figure or Matplotlib Axes).
    - backend (str): 'plotly' or 'matplotlib'.

    Returns:
    - fig: Modified plot object with switched axes.
    """
    if backend == 'plotly':
        # Switch x and y data for all traces
        for trace in fig.data:
            trace.x, trace.y = trace.y, trace.x
        # Swap axis titles
        x_title = fig.layout.xaxis.title.text
        y_title = fig.layout.yaxis.title.text
        fig.update_layout(xaxis_title=y_title, yaxis_title=x_title)
    elif backend == 'matplotlib':
        # Swap x and y data
        ax = fig.axes[0]
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            line.set_xdata(y_data)
            line.set_ydata(x_data)
        # Swap axis labels
        x_label = ax.get_xlabel()
        y_label = ax.get_ylabel()
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
        # Invert y-axis if necessary
        ax.invert_yaxis()
    return fig


def preprocess_domino_plot(data,
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
                           max_dot_size):
    """
    Preprocesses data for domino plot generation.

    Parameters:
    - data (pd.DataFrame): Original data.
    - gene_list (list): List of genes to include.
    - spacing_factor (int): Spacing between gene pairs.
    - contrast_levels (list): List of contrast levels.
    - feature_col (str): Column name for genes.
    - celltype_col (str): Column name for cell types.
    - contrast_col (str): Column name for contrasts.
    - var_id (str): Column name for variables.
    - logfc_col (str): Column name for log fold change.
    - pval_col (str): Column name for adjusted p-values.
    - logfc_limits (tuple): Limits for log fold change values.
    - min_dot_size (float): Minimum dot size.
    - max_dot_size (float): Maximum dot size.

    Returns:
    - plot_data (pd.DataFrame): Data prepared for plotting.
    - aspect_ratio (float): Calculated aspect ratio for the plot.
    - unique_celltypes (list): List of unique cell types.
    - unique_genes (list): List of unique genes.
    """
    # Filter genes
    data_filtered = filter_genes_domino(data, gene_list, feature_col)

    # Create complete dataset
    complete_data = create_complete_dataset_domino(data_filtered, gene_list, celltype_col, contrast_col, var_id, feature_col)

    # Convert celltype_col and contrast_col to categorical with specified order
    complete_data[celltype_col] = pd.Categorical(complete_data[celltype_col], categories=sorted(complete_data[celltype_col].unique()), ordered=True)
    complete_data[contrast_col] = pd.Categorical(complete_data[contrast_col], categories=contrast_levels, ordered=True)

    # Calculate positions
    plot_data = calculate_positions_domino(complete_data, spacing_factor, contrast_levels, feature_col, celltype_col, contrast_col)

    # Aspect Ratio Calculation
    n_genes = len(gene_list)
    unique_celltypes = complete_data[celltype_col].cat.categories.tolist()
    n_celltypes = len(unique_celltypes)
    aspect_ratio = calculate_aspect_ratio_domino(n_genes, n_celltypes, spacing_factor)

    # Apply color and size scaling
    plot_data['adj_logfc'] = apply_color_scale_domino(plot_data, logfc_col, logfc_limits)
    plot_data['size'] = apply_size_scale_domino(plot_data, pval_col, min_dot_size, max_dot_size)

    # Store unique genes and cell types
    unique_genes = gene_list  # Since gene_list is already ordered
    unique_celltypes = plot_data[celltype_col].cat.categories.tolist()

    return plot_data, aspect_ratio, unique_celltypes, unique_genes
