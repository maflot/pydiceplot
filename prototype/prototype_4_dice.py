import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings

# Define the pathology variables and their colors
pathology_variables = ["Amyloid", "NFT", "Tangles", "Plaq N"]
cat_c_colors = {
    "Amyloid": "#d5cccd",
    "NFT": "#cb9992",
    "Tangles": "#ad310f",
    "Plaq N": "#7e2a20"
}

# Define cell types (cat_a)
cell_types = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", "Endothelial"]

# Define pathways (cat_b) and add 10 more to make a total of 15
pathways = [
    "Apoptosis", "Inflammation", "Metabolism", "Signal Transduction", "Synaptic Transmission",
    "Cell Cycle", "DNA Repair", "Protein Synthesis", "Lipid Metabolism", "Neurotransmitter Release",
    "Oxidative Stress", "Energy Production", "Calcium Signaling", "Synaptic Plasticity", "Immune Response"
]

# Assign groups to pathways (ensuring each pathway has only one group)
pathway_groups = pd.DataFrame({
    "Pathway": pathways,
    "Group": [
        "BBB-linked", "Cell-proliferation", "Other", "BBB-linked", "Cell-proliferation",
        "Cell-proliferation", "Other", "Other", "Other", "BBB-linked",
        "Other", "Other", "BBB-linked", "Cell-proliferation", "Other"
    ]
})

# Update group colors to shades of greys
group_colors = {
    "BBB-linked": "#333333",
    "Cell-proliferation": "#888888",
    "Other": "#DDDDDD"
}

# Create dummy data
np.random.seed(123)
data = pd.DataFrame([(ct, pw) for ct in cell_types for pw in pathways], columns=["CellType", "Pathway"])

# Assign random pathology variables to each combination
data_list = []
for idx, row in data.iterrows():
    n_vars = np.random.randint(1, 5)  # Random number between 1 and 4
    variables = np.random.choice(pathology_variables, size=n_vars, replace=False)
    for var in variables:
        data_list.append({
            "CellType": row["CellType"],
            "Pathway": row["Pathway"],
            "PathologyVariable": var
        })

# Create DataFrame from data_list
data_expanded = pd.DataFrame(data_list)

# Merge the group assignments into the data
data_expanded = data_expanded.merge(pathway_groups, on="Pathway", how="left")

def dice_plot(data,
              cat_a,
              cat_b,
              cat_c,
              group,
              plot_path="./",
              output_str="",
              switch_axis=False,
              group_alpha=0.6,
              title=None,
              cat_c_colors=None,
              group_colors=None,
              format=".pdf",
              custom_theme=None):
    import warnings
    import plotly.graph_objects as go
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, dendrogram

    # Set default colors if not provided
    if group_colors is None:
        group_colors = {
            "BBB-linked": "#333333",
            "Cell-proliferation": "#888888",
            "Other": "#DDDDDD"
        }

    if cat_c_colors is None:
        cat_c_colors = {
            "Amyloid": "#d5cccd",
            "NFT": "#cb9992",
            "Tangles": "#ad310f",
            "Plaq N": "#7e2a20"
        }

    # Ensure consistent ordering of factors
    data[cat_a] = pd.Categorical(data[cat_a], categories=data[cat_a].unique(), ordered=True)
    data[cat_b] = pd.Categorical(data[cat_b], categories=data[cat_b].unique(), ordered=True)
    data[cat_c] = pd.Categorical(data[cat_c], categories=list(cat_c_colors.keys()), ordered=True)

    # Check for unique group per cat_b
    group_check = data.groupby(cat_b)[group].nunique().reset_index()
    group_check = group_check[group_check[group] > 1]
    if not group_check.empty:
        warnings.warn("Warning: The following cat_b categories have multiple groups assigned:\n{}".format(
            ', '.join(group_check[cat_b].tolist())
        ))

    # Ensure group is a factor with levels matching group_colors
    data[group] = pd.Categorical(data[group], categories=list(group_colors.keys()), ordered=True)

    # Define variable positions for the mini plots
    var_positions = pd.DataFrame({
        'var': pd.Categorical(list(cat_c_colors.keys()), categories=list(cat_c_colors.keys()), ordered=True),
        'x_offset': [-0.2, 0.2, -0.2, 0.2],
        'y_offset': [0.2, 0.2, -0.2, -0.2]
    })

    # Create a binary matrix for clustering
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
    cat_a_order = dendro['ivl'][::-1]  # Reverse to match R code

    # Order cat_b based on group and frequency
    group_levels = list(group_colors.keys())[::-1]  # Reverse order
    data[group] = pd.Categorical(data[group], categories=group_levels, ordered=True)
    grouped_b = data.groupby([group, cat_b]).size().reset_index(name='count')
    grouped_b = grouped_b.sort_values(by=[group, 'count', cat_b], ascending=[True, False, True])
    cat_b_order = grouped_b[cat_b].unique().tolist()

    # Update plot_data
    plot_data = data.merge(var_positions, left_on=cat_c, right_on='var', how='left')
    plot_data[cat_a] = pd.Categorical(plot_data[cat_a], categories=cat_a_order, ordered=True)
    plot_data[cat_b] = pd.Categorical(plot_data[cat_b], categories=cat_b_order, ordered=True)
    plot_data['x_num'] = plot_data[cat_a].cat.codes + 1
    plot_data['y_num'] = plot_data[cat_b].cat.codes + 1
    plot_data['x_pos'] = plot_data['x_num'] + plot_data['x_offset']
    plot_data['y_pos'] = plot_data['y_num'] + plot_data['y_offset']
    plot_data = plot_data.sort_values(by=[cat_a, group, cat_b])

    # Update box_data
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

    # Create the figure
    fig = go.Figure()

    # Add rectangles for the boxes
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

    # Add scatter traces for the PathologyVariables
    scatter_data = plot_data.copy()
    for var in cat_c_colors.keys():
        var_data = scatter_data[scatter_data[cat_c] == var]
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

    # Add group legend entries
    for g in group_colors.keys():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=10,
                color=group_colors[g],
                opacity=group_alpha
            ),
            legendgroup=g,
            showlegend=True,
            name=g
        ))

    # Set axes
    x_tickvals = list(range(1, len(cat_a_order)+1))
    x_ticktext = cat_a_order
    y_tickvals = list(range(1, len(cat_b_order)+1))
    y_ticktext = cat_b_order

    fig.update_xaxes(
        tickvals=x_tickvals,
        ticktext=x_ticktext,
        tickangle=45,
        tickmode='array',
        title_text='',
        showgrid=False,
        zeroline=False,
        range=[0.5, len(x_tickvals)+0.5]
    )

    fig.update_yaxes(
        tickvals=y_tickvals,
        ticktext=y_ticktext,
        tickmode='array',
        title_text='',
        showgrid=False,
        zeroline=False,
        range=[0.5, len(y_tickvals)+0.5],
        scaleanchor="x",
        scaleratio=1
    )

    # Calculate dynamic width and height to make boxes square
    box_size = 50  # pixels per box
    n_x = len(x_tickvals)
    n_y = len(y_tickvals)
    margin_l = 150
    margin_r = 300
    margin_t = 100
    margin_b = 200
    plot_width = box_size * n_x + margin_l + margin_r
    plot_height = box_size * n_y + margin_t + margin_b

    # Set layout
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
        ),
        yaxis=dict(
            showline=False,
            zeroline=False,
            showgrid=False,
            scaleanchor="x",
            scaleratio=1
        ),
        title=title,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.05
        ),
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        width=plot_width,
        height=plot_height
    )

    if switch_axis:
        fig.update_layout(yaxis_side='right')

    # Save the plot
    if format == ".html":
        fig.write_html(f"{plot_path}/{output_str}_dice_plot.html")
    else:
        fig.write_image(f"{plot_path}/{output_str}_dice_plot{format}")

    return fig

# Use the modified dice_plot function
fig = dice_plot(data=data_expanded,
                cat_a="CellType",
                cat_b="Pathway",
                cat_c="PathologyVariable",
                group="Group",
                plot_path="./",
                output_str="dummy_dice_plot",
                switch_axis=False,
                group_alpha=0.6,
                title="Dummy Dice Plot with Pathology Variables",
                cat_c_colors=cat_c_colors,
                group_colors=group_colors,
                format=".html")  # You can change to .pdf or .png if needed

# To display the figure in a Jupyter notebook
fig.show()