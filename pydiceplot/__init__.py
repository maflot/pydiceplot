from .plots._diceplot import dice_plot
from .plots._dominoplot import domino_plot
from .backends._plotly_backend import plot_with_plotly
from .backends._matplotlib_backend import plot_with_matplotlib


def plot(plot_type, data, backend="plotly", **kwargs):
    """Plots a dice or domino plot.

    Parameters
        plot_type : str
            The type of plot to create. Either "dice" or "domino".
        data : pd.DataFrame
            The data to plot.
        backend : str
            The plotting backend to use. Either "plotly" or "matplotlib".
    """

    if plot_type == "dice":
        plot_func = dice_plot
    elif plot_type == "domino":
        plot_func = domino_plot
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    if backend == "plotly":
        return plot_func(data, renderer=plot_with_plotly, **kwargs)
    elif backend == "matplotlib":
        return plot_func(data, renderer=plot_with_matplotlib, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
