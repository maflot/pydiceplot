"""The ggdiceplot diverging purple→white→green palette.

Registered globally as `ggdiceplot_pg` so the examples can just pass
`color_map="ggdiceplot_pg"` and get the same look as the R `scale_fill_gradient2(
low="#40004B", high="#00441B", mid="white")`.
"""

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


NAME = "ggdiceplot_pg"


def register() -> None:
    if NAME in colormaps:
        return
    cmap = LinearSegmentedColormap.from_list(
        NAME,
        ["#40004B", "#FFFFFF", "#00441B"],
    )
    colormaps.register(cmap, name=NAME)
