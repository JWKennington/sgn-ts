from __future__ import annotations

from typing import Any

from sgnts.base.array_backend import Array as _Array
from sgnts.base.array_backend import ArrayBackend as _ArrayBackend
from sgnts.base.numpy_backend import NumpyArray as _NumpyArray
from sgnts.base.numpy_backend import NumpyBackend as _NumpyBackend

Array = _Array
NumpyArray = _NumpyArray
NumpyBackend = _NumpyBackend
ArrayBackend = _ArrayBackend

_TorchBackend: Any
_TorchArray: Any
try:
    from sgnts.base.torch_backend import TorchArray as _TorchArray
    from sgnts.base.torch_backend import TorchBackend as _TorchBackend
except ImportError:
    from sgnts.base.notorch_backend import TorchArray as _TorchArray
    from sgnts.base.notorch_backend import TorchBackend as _TorchBackend
TorchBackend = _TorchBackend
TorchArray = _TorchArray
