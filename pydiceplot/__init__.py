from pydiceplot.plots._plot import dice_plot as dice_plot
from ._backend import set_backend as set_backend
from .plots._plot import domino_plot as domino_plot

__all__ = ["dice_plot", "domino_plot", "set_backend", "__version__"]

try:
    from ._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version as _pkg_version
    except ImportError:  # pragma: no cover
        from importlib_metadata import version as _pkg_version  # type: ignore

    try:
        __version__ = _pkg_version("pydiceplot")
    except Exception:  # pragma: no cover
        __version__ = "0+unknown"
