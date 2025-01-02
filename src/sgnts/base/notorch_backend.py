from sgnts.base.array_backend import ArrayBackend


class TorchBackend(ArrayBackend):
    """A nonfunctional TorchBackend stub"""

    # FIXME: How to handle different device/dtypes in the same pipeline?
    DTYPE = None
    DEVICE = None


TorchArray = None
