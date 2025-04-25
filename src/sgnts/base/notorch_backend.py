from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

from sgnts.base.array_backend import Array, ArrayBackend


class TorchBackend(ArrayBackend):
    """A fallback TorchBackend that raises informative errors when torch is not
    available"""

    DTYPE = None
    DEVICE = None

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @staticmethod
    def all(*args, **kwargs):
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @classmethod
    def arange(cls, stop: float, start: float = 0, step: float = 1) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @staticmethod
    def cat(data: Sequence[Array], axis: int) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @classmethod
    def full(cls, shape: Tuple[int, ...], fill_value: Any) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @staticmethod
    def matmul(a: Array, b: Array) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @staticmethod
    def pad(data: Array, pad_samples: tuple[int, int]) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @classmethod
    def set_device(cls, device: Union[str, Any]) -> None:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @classmethod
    def set_dtype(cls, dtype: Any) -> None:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @staticmethod
    def stack(data: Sequence[Array], axis: int = 0) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @staticmethod
    def sum(a: Array, axis: Optional[Union[int, tuple[int, ...]]] = None) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> Array:
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )


class TorchArray:
    """A placeholder to indicate TorchArray is not available"""

    def __new__(cls, *args, **kwargs):
        raise ImportError(
            "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
        )
