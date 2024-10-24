# pydiceplot
[![PyPI version](https://badge.fury.io/py/pydiceplot.svg)](https://pypi.org/project/pydiceplot/)

The **pyDicePlot** package allows you to create visualizations (dice plots) for datasets with more than two categorical variables and additional continuous variables. This tool is particularly useful for exploring complex categorical data and their relationships with continuous variables.

## Requirements
This code requires python 3.
```bash
conda create --name pydiceplot python=3
conda activate pydiceplot
pip install -r requirements.txt
```

Installation via pip
To install the package via pip, run
```bash
pip install -e .
```
from the base directory.

## Use the dice plots in R
for using dice plots in R please refer to [DicePlot](https://github.com/maflot/DicePlot/tree/main)

## Sample Output

![Sample Dice with 3 categories Plot](images/dice_plot_3_example_dice_plot.png)
![Sample Dice with 6 categories Plot](images/dice_plot_5_example_dice_plot.png)

*Figure: A sample dice plot generated using the `DicePlot` package.*

## Documentation

For full documentation and additional examples, please refer to the [documentation](https://dice-and-domino-plot.readthedocs.io/en/latest/index.html#)

## Usage example

```python 

    import numpy as np
    import pandas as pd
    from pydiceplot import dice_plot
    from pydiceplot.plots.backends._dice_utils import (get_diceplot_example_data,
                                                       get_example_group_colors,
                                                       get_example_cat_c_colors)
    import pydiceplot

    #Set the backend for pydiceplot
    pydiceplot.set_backend("matplotlib")
    pydiceplot.set_backend("plotly")
    
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
                group="Group", # default is set to None, it will color the boxes plain white
                switch_axis=False,
                title=title,
                cat_c_colors=current_cat_c_colors,
                group_colors=group_colors,  # Include group colors
                max_dice_sides=6  # Adjust if needed
            )
    
            # Display and save the figure
            fig.show()

```

## Features

- **Visualize Complex Data:** Easily create plots for datasets with multiple categorical variables.
- **Customization:** Customize plots with titles, labels, and themes.
- **Integration with ggplot2:** Leverages the power of `ggplot2` for advanced plotting capabilities.

## Contributing

We welcome contributions from the community! If you'd like to contribute:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## Contact

If you have any questions, suggestions, or issues, please open an issue on GitHub.
