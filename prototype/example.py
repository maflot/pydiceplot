import numpy as np
import pandas as pd
from pydiceplot import dice_plot
from pydiceplot.plots.backends._dice_utils import (get_diceplot_example_data,
                                                   get_example_group_colors,
                                                   get_example_cat_c_colors)
import pydiceplot

# Set the backend for pydiceplot
pydiceplot.set_backend("matplotlib")

if __name__ == "__main__":
    plot_path = "./plots"

    # define colors for the example plot


    # Generate and save dice plots for different numbers of pathology variables
    for n in [2,3, 4, 5, 6]:
        # Get the data using the utility function
        # load example data
        data_expanded = get_diceplot_example_data(n)
        group_colors = get_example_group_colors()
        cat_c_colors = get_example_cat_c_colors()
        # Define pathology variables and their colors
        # extract pathology variables and select proper color scale
        pathology_vars = data_expanded["PathologyVariable"].unique()
        current_cat_c_colors = {var: cat_c_colors[var] for var in pathology_vars}

        # Create the dice plot
        title = f"Dice Plot with {n} Pathology Variables"
        fig = dice_plot(
            data=data_expanded,
            cat_a="CellType",
            cat_b="Pathway",
            cat_c="PathologyVariable",
            group="Group",
            switch_axis=False,
            title=title,
            cat_c_colors=current_cat_c_colors,
            group_colors=group_colors,  # Include group colors
            max_dice_sides=6  # Adjust if needed
        )

        # Display and save the figure
        fig.show()
