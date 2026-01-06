try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

from sgnts.compose import (
    TSCompose,
    TSComposedSinkElement,
    TSComposedSourceElement,
    TSComposedTransformElement,
)

__all__ = [
    "__version__",
    "TSCompose",
    "TSComposedSourceElement",
    "TSComposedTransformElement",
    "TSComposedSinkElement",
]
