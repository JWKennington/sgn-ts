"""Classes for specifying array operation implementations. Specifically,
we define a set of generic array operations that can be implemented in various
backends (e.g., numpy, pytorch, tensorflow, etc.). This allows us to write
generic code that can be run on different backends without modification.

The operations are defined as static methods in the `ArrayBackend` class, and
must be implemented in subclasses. The current set of operations includes:

- `cat`: Concatenate arrays along a specified axis
- `pad`: Pad an array with zeros
- `full`: Create an array filled with a specified value
- `zeros`: Create an array of zeros
- `stack`: Stack arrays along a new axis
- `matmul`: Perform matrix multiplication of two arrays
"""

from __future__ import annotations

from functools import wraps
from typing import Any, ClassVar, Iterable, Optional, Tuple

import numpy

try:
    import torch

    TorchArray = torch.Tensor

    # Set some global PyTorch settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except ImportError:
    torch = None
    TorchArray = Any

# Alias for a generic array type
Array = Any
NumpyArray = numpy.ndarray


class ArrayBackend:
    """Base class for array operation implementations. Subclasses should
    implement the array operations as static methods.
    """

    DEVICE: ClassVar[Optional[str]] = None
    DTYPE: ClassVar[Optional[type]] = None

    @staticmethod
    def cat(data: Iterable[Array], axis: int) -> Array:
        """Concatenate arrays along a specified axis.

        Args:
            data:
                Iterable[Array]: Arrays to concatenate, all with the same shape
            axis:
                int: Axis along which to concatenate the arrays

        Returns:
            Array: Concatenated array
        """
        raise NotImplementedError

    @staticmethod
    def pad(data: Array, pad_samples: tuple[int, int]) -> Array:
        """Pad an array with zeros.

        Args:
            data:
                Array: Array to pad
            pad_samples:
                tuple: Number of zeros to pad at each end of the array

        Returns:
            Array: Padded array
        """
        raise NotImplementedError

    @staticmethod
    def full(shape: Tuple[int, ...], fill_value: Any) -> Array:
        """Create an array filled with a specified value.

        Args:
            shape:
                Tuple[int, ...]: Shape of the array
            fill_value:
                Any: Value to fill the array with

        Returns:
            Array: Array filled with the specified value
        """
        raise NotImplementedError

    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> Array:
        """Create an array of zeros.

        Args:
            shape:
                Tuple[int, ...]: Shape of the array

        Returns:
            Array: Array of zeros
        """
        raise NotImplementedError

    @classmethod
    def stack(cls, data: Iterable[Array], axis: int = 0) -> Array:
        """Stack arrays along a new axis.

        Args:
            data:
                Iterable[Array]: Arrays to stack, all with the same shape
            axis:
                int: Axis along which to stack the arrays

        Returns:
            Array: Stacked array
        """
        return ArrayBackend.cat(data, axis=axis)

    @staticmethod
    def matmul(a: Array, b: Array) -> Array:
        """Matrix multiplication of two arrays.
            out = a x b

        Args:
            a:
                Array, the first array
            b:
                Array, the second array

        Returns:
            Array, the result of the matrix multiplication
        """
        raise NotImplementedError


class NumpyBackend(ArrayBackend):
    """Implementation of array operations using numpy."""

    DEVICE = "cpu"
    DTYPE = numpy.float64

    @staticmethod
    def cat(data: Iterable[NumpyArray], axis: int) -> NumpyArray:
        """Concatenate arrays along a specified axis

        Args:
            data:
                Iterable[NumpyArray], Arrays to concatenate, all with the same shape
            axis:
                int, Axis along which to concatenate the arrays

        Returns:
            NumpyArray, concatenated array
        """
        return numpy.concatenate(data, axis=axis)

    @staticmethod
    def pad(data: NumpyArray, pad_samples: tuple[int, int]) -> NumpyArray:
        """Pad an array with zeros

        Args:
            data:
                NumpyArray, Array to pad
            pad_samples:
                tuple, Number of zeros to pad at each end of the array

        Returns:
            NumpyArray, Padded array
        """
        npad = [(0, 0)] * data.ndim
        npad[-1] = pad_samples
        return numpy.pad(data, npad, "constant")

    @staticmethod
    def full(shape: Tuple[int, ...], fill_value: Any) -> NumpyArray:
        """Create an array filled with a specified value

        Args:
            shape:
                Tuple[int, ...], Shape of the array
            fill_value:
                Any, Value to fill the array with

        Returns:
            NumpyArray, Array filled with the specified value
        """
        return numpy.full(shape, fill_value)

    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> NumpyArray:
        """Create an array of zeros

        Args:
            shape:
                Tuple[int, ...], Shape of the array

        Returns:
            NumpyArray, Array of zeros
        """
        return numpy.zeros(shape)

    @classmethod
    def stack(cls, data: Iterable[NumpyArray], axis: int = 0) -> NumpyArray:
        """Stack arrays along a new axis

        Args:
            data:
                Iterable[NumpyArray], Arrays to stack, all with the same shape
            axis:
                int, Axis along which to stack the arrays

        Returns:
            NumpyArray, Stacked array
        """
        return numpy.stack(data, axis=axis)

    @staticmethod
    def matmul(a: NumpyArray, b: NumpyArray) -> NumpyArray:
        """Matrix multiplication of two arrays.
            out = a x b

        Args:
            a:
                NumpyArray, the first array
            b:
                NumpyArray, the second array

        Returns:
            NumpyArray, the result of the matrix multiplication
        """
        return numpy.matmul(a, b)

    @staticmethod
    @wraps(numpy.all)
    def all(*args, **kwargs):
        return numpy.all(*args, **kwargs)


class TorchBackend(ArrayBackend):
    """Implementation of array operations using PyTorch tensors."""

    # FIXME: How to handle different device/dtypes in the same pipeline?
    DTYPE = None if torch is None else torch.float32
    DEVICE = None if torch is None else "cpu"

    @classmethod
    def set_device(cls, device: str) -> None:
        """Set the torch device.

        Args:
            device:
                str, the device on which to create torch tensors
        """
        cls.DEVICE = device

    @classmethod
    def set_dtype(cls, dtype: torch.dtype) -> None:
        """Set the torch data type.

        Args:
            dtype:
                torch.dtype, the data type of the torch tensors
        """
        cls.DTYPE = dtype

    @staticmethod
    def _check_torch():
        """Check if PyTorch is available"""
        if torch is None:
            raise ImportError("PyTorch is required to use TorchBackend")

    @staticmethod
    def cat(data: Iterable[TorchArray], axis: int) -> TorchArray:
        """Concatenate arrays along a specified axis

        Args:
            data:
                Iterable[TorchArray], Arrays to concatenate, all with the same shape
            axis:
                int, Axis along which to concatenate the arrays

        Returns:
            TorchArray, concatenated array
        """
        TorchBackend._check_torch()
        return torch.cat(data, dim=axis)

    @staticmethod
    def pad(data: TorchArray, pad_samples: tuple[int, int]) -> TorchArray:
        """Pad an array with zeros

        Args:
            data:
                TorchArray, Array to pad
            pad_samples:
                tuple[int, int], Number of zeros to pad at each end of the array

        Returns:
            TorchArray, Padded array
        """
        TorchBackend._check_torch()
        return torch.nn.functional.pad(data, pad_samples, "constant")

    @classmethod
    def full(cls, shape: Tuple[int, ...], fill_value: Any) -> TorchArray:
        """Create an array filled with a specified value

        Args:
            shape:
                Tuple[int, ...], Shape of the array
            fill_value:
                Any, Value to fill the array with

        Returns:
            TorchArray, Array filled with the specified value
        """
        TorchBackend._check_torch()
        return torch.full(shape, fill_value, device=cls.DEVICE, dtype=cls.DTYPE)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> TorchArray:
        """Create an array of zeros

        Args:
            shape:
                Tuple[int, ...], Shape of the array

        Returns:
            TorchArray, Array of zeros
        """
        TorchBackend._check_torch()
        return torch.zeros(shape, device=cls.DEVICE, dtype=cls.DTYPE)

    @staticmethod
    def stack(data: Iterable[TorchArray], axis: int = 0) -> TorchArray:
        """Stack arrays along a new axis

        Args:
            data:
                Iterable[TorchArray], Arrays to stack, all with the same shape
            axis:
                int, Axis along which to stack the arrays

        Returns:
            TorchArray, Stacked array
        """
        TorchBackend._check_torch()
        return torch.stack(data, axis)

    @staticmethod
    def matmul(a: TorchArray, b: TorchArray) -> TorchArray:
        """Matrix multiplication of two arrays.
            out = a x b

        Args:
            a:
                TorchArray, the first array
            b:
                TorchArray, the second array

        Returns:
            TorchArray, the result of the matrix multiplication
        """
        return torch.matmul(a, b)

    @staticmethod
    def all(input: TorchArray, out: Optional[TorchArray] = None):
        """Returns true if all elements are true"""
        TorchBackend._check_torch()
        return torch.all(input=input, out=out)
