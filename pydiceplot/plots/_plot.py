from pydiceplot._backend import _backend
import importlib


class Plot:
    def __init__(self, **kwargs):
        self.plot_type = kwargs.get("plot_type", "dice")
        self.data = kwargs.get("data")
        module_name = f"pydiceplot.backends.{_backend}_backend"
        self._backend_module = importlib.import_module(module_name)
        self.fig = None

    def show(self):
        getattr(self._backend_module, "show_plot")(self.fig)

    def save(self, plot_path, output_str, formats):
        (getattr(self._backend_module, "save_plot")
         (self.fig, plot_path, output_str, formats))


class DicePlot(Plot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plot_function = getattr(
            self._backend_module, "plot_dice")

class DominoPlot(Plot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def __plot(plot_type, data, backend=_backend, **kwargs):
    """Plots a dice or domino plot.

    Parameters
        plot_type : str
            The type of plot to create. Either "dice" or "domino".
        data : pd.DataFrame
            The data to plot.
        backends : str
            The plotting backends to use. Either "plotly" or "matplotlib".
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
        raise ValueError(f"Unknown backends: {backend}")


def dice_plot():
    pass


def domino_plot():
    pass
