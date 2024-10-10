# dice_plot.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from utils import create_var_positions, perform_clustering, order_cat_b, prepare_plot_data, prepare_box_data
def dice_plot(data,
              cat_a,
              cat_b,
              cat_c,
              group,
              plot_path="./", output_str="",
              switch_axis=False,
              group_alpha=0.6,
              title=None,
              cat_c_colors=None,
              group_colors=None,
              format=".pdf",
              custom_theme=None):
    # Set default colors if not provided
    if group_colors is None:
        group_colors = {
            "BBB-linked": "#333333",
            "Cell-proliferation": "#888888",
            "Other": "#DDDDDD"
        }

    if cat_c_colors is None:
        default_cat_c_colors = {
            "Var1": "#d5cccd",
            "Var2": "#cb9992",
            "Var3": "#ad310f",
            "Var4": "#7e2a20",
            "Var5": "#FFD700",
            "Var6": "#FF6622"
        }
        cat_c_colors = default_cat_c_colors

    num_unique_cat_c = len(data[cat_c].unique())
    if num_unique_cat_c not in [3, 4, 5, 6]:
        raise ValueError("Unsupported number of categories for cat_c. Must be 3, 4, 5, or 6.")

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
    var_positions = create_var_positions(cat_c_colors, num_unique_cat_c)

    # Perform hierarchical clustering
    cat_a_order = perform_clustering(data, cat_a, cat_b, cat_c)

    # Order cat_b based on group and frequency
    cat_b_order = order_cat_b(data, group, cat_b, group_colors)

    # Prepare plot data
    plot_data = prepare_plot_data(data, cat_a, cat_b, cat_c, group, var_positions, cat_a_order, cat_b_order)

    # Prepare box data
    box_data = prepare_box_data(data, cat_a, cat_b, group, cat_a_order, cat_b_order)

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
    for var in var_positions['var']:
        var_data = plot_data[plot_data[cat_c] == var]
        fig.add_trace(go.Scatter(
            x=var_data['x_pos'],
            y=var_data['y_pos'],
            mode='markers',
            marker=dict(
                size=10,
                color=cat_c_colors[str(var)],
                line=dict(width=1, color='black')
            ),
            name=str(var),
            legendgroup=str(var),
            showlegend=False  # Will create custom legend
        ))

    # Add custom legend entries for cat_c
    legend_entries = []
    for idx, row in var_positions.iterrows():
        legend_entries.append(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=10,
                color=cat_c_colors[str(row['var'])],
                line=dict(width=1, color='black')
            ),
            legendgroup=str(row['var']),
            showlegend=True,
            name=str(row['var'])
        ))
    fig.add_traces(legend_entries)

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



if __name__ == "__main__":
    # main.py

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

    # Define pathways (cat_b) and add more to make a total of 15
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
    data_expanded = data_expanded.merge(pathway_groups, left_on="Pathway", right_on="Pathway", how="left")

    # Use the dice_plot function
    fig = dice_plot(
        data=data_expanded,
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
        format=".html"  # You can change to ".pdf" or ".png" if needed
    )

    # To display the figure in a Jupyter notebook
    fig.show()