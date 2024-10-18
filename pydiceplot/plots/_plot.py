from pydiceplot._backend import _backend


class Plot:
    def __init__(self, **kwargs):
        self.plot_type = kwargs.get("plot_type", "dice")
        self.data = kwargs.get("data")

    def plot(self):
        if self.plot_type == "dice":
            return self.dice_plot()
        elif self.plot_type == "domino":
            return self.domino_plot()
        else:
            raise ValueError(f"Unknown plot type: {self.plot_type}")

    def dice_plot(self):
        pass

    def domino_plot(self):
        pass

    def show(self):
        pass

    def save(self):
        pass


class DicePlot(Plot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DominoPlot(Plot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plot_type = "domino"



def __plot(plot_type, data, backend=_backend, **kwargs):
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


def dice_plot():
    pass


def domino_plot():
    pass
