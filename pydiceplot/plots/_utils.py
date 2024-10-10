# _utils.py
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

def create_var_positions(cat_c_colors, num_vars):
    num_vars = str(num_vars)
    if num_vars == "3":
        var_positions = pd.DataFrame({
            'var': pd.Categorical(list(cat_c_colors.keys()), categories=list(cat_c_colors.keys()), ordered=True),
            'x_offset': [0, -0.2, 0.2],
            'y_offset': [0, 0.2, -0.2]
        })
    elif num_vars == "4":
        var_positions = pd.DataFrame({
            'var': pd.Categorical(list(cat_c_colors.keys()), categories=list(cat_c_colors.keys()), ordered=True),
            'x_offset': [-0.2, 0.2, -0.2, 0.2],
            'y_offset': [0.2, 0.2, -0.2, -0.2]
        })
    elif num_vars == "5":
        var_positions = pd.DataFrame({
            'var': pd.Categorical(list(cat_c_colors.keys()), categories=list(cat_c_colors.keys()), ordered=True),
            'x_offset': [0, -0.2, 0.2, -0.2, 0.2],
            'y_offset': [0, 0.2, 0.2, -0.2, -0.2]
        })
    elif num_vars == "6":
        var_positions = pd.DataFrame({
            'var': pd.Categorical(list(cat_c_colors.keys()), categories=list(cat_c_colors.keys()), ordered=True),
            'x_offset': [-0.2, 0.2, -0.2, 0.2, -0.2, 0.2],
            'y_offset': [0.2, 0.2, 0, 0, -0.2, -0.2]
        })
    else:
        raise ValueError("Unsupported number of variables for variable positions.")
    return var_positions

def perform_clustering(data, cat_a, cat_b, cat_c):
    data['present'] = 1
    grouped = data.groupby([cat_a, cat_b, cat_c])['present'].sum().reset_index()
    grouped['combined'] = grouped[cat_b].astype(str) + "_" + grouped[cat_c].astype(str)
    binary_matrix_df = grouped.pivot(index=cat_a, columns='combined', values='present').fillna(0)
    binary_matrix = binary_matrix_df.values

    # Perform hierarchical clustering
    cell_types = binary_matrix_df.index.tolist()
    distance_matrix = pdist(binary_matrix, metric='jaccard')
    Z = linkage(distance_matrix, method='ward')
    dendro = dendrogram(Z, labels=cell_types, no_plot=True)
    cat_a_order = dendro['ivl'][::-1]  # Reverse to match the R code
    return cat_a_order

def order_cat_b(data, group, cat_b, group_colors):
    group_levels = list(group_colors.keys())[::-1]  # Reverse order
    data[group] = pd.Categorical(data[group], categories=group_levels, ordered=True)
    grouped_b = data.groupby([group, cat_b]).size().reset_index(name='count')
    grouped_b = grouped_b.sort_values(by=[group, 'count', cat_b], ascending=[True, False, True])
    cat_b_order = grouped_b[cat_b].unique().tolist()
    return cat_b_order

def prepare_plot_data(data, cat_a, cat_b, cat_c, group, var_positions, cat_a_order, cat_b_order):
    plot_data = data.merge(var_positions, left_on=cat_c, right_on='var', how='left')
    plot_data[cat_a] = pd.Categorical(plot_data[cat_a], categories=cat_a_order, ordered=True)
    plot_data[cat_b] = pd.Categorical(plot_data[cat_b], categories=cat_b_order, ordered=True)
    plot_data['x_num'] = plot_data[cat_a].cat.codes + 1
    plot_data['y_num'] = plot_data[cat_b].cat.codes + 1
    plot_data['x_pos'] = plot_data['x_num'] + plot_data['x_offset']
    plot_data['y_pos'] = plot_data['y_num'] + plot_data['y_offset']
    plot_data = plot_data.sort_values(by=[cat_a, group, cat_b])
    return plot_data

def prepare_box_data(data, cat_a, cat_b, group, cat_a_order, cat_b_order):
    data[cat_a] = pd.Categorical(data[cat_a], categories=cat_a_order, ordered=True)
    data[cat_b] = pd.Categorical(data[cat_b], categories=cat_b_order, ordered=True)
    box_data = data[[cat_a, cat_b, group]].drop_duplicates()
    box_data['x_num'] = box_data[cat_a].cat.codes + 1
    box_data['y_num'] = box_data[cat_b].cat.codes + 1
    box_data['x_min'] = box_data['x_num'] - 0.4
    box_data['x_max'] = box_data['x_num'] + 0.4
    box_data['y_min'] = box_data['y_num'] - 0.4
    box_data['y_max'] = box_data['y_num'] + 0.4
    box_data = box_data.sort_values(by=[cat_a, group, cat_b])
    return box_data