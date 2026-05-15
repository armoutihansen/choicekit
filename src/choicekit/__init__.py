"""ChoiceKit: tools for choice modelling in Python."""

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("choicekit")
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"
