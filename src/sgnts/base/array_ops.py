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
"""

from typing import Any, ClassVar, Tuple, Iterable, Optional

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
    def pad(data: Array, pad_samples: tuple) -> Array:
        """Pad an array with zeros.

        Args:
            data:
                Array: Array to pad
            pad_samples:
                tuple: Number of samples to pad at the end of the array

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

    # TODO Remove this backwards compatibility set of aliases when larger refactor
    #  complete
    pad_func = pad
    zeros_func = zeros
    full_func = full
    cat_func = cat
    stack_func = stack


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
    def pad(data: NumpyArray, pad_samples: tuple) -> NumpyArray:
        """Pad an array with zeros

        Args:
            data:
                NumpyArray, Array to pad
            pad_samples:
                tuple, Number of samples to pad at the end of the array

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

    # TODO Remove this backwards compatibility set of aliases when larger refactor
    #  complete
    pad_func = pad
    zeros_func = zeros
    full_func = full
    cat_func = cat
    stack_func = stack


# TODO remove this alias when refactor complete
ArrayOps = NumpyBackend


class _TorchBackend(ArrayBackend):
    """Implementation of array operations using PyTorch tensors."""

    DTYPE = None if torch is None else torch.float32

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
        _TorchBackend._check_torch()
        return torch.cat(data, dim=axis)

    @staticmethod
    def pad(data: TorchArray, pad_samples: tuple) -> TorchArray:
        """Pad an array with zeros

        Args:
            data:
                TorchArray, Array to pad
            pad_samples:
                tuple, Number of samples to pad at the end of the array

        Returns:
            TorchArray, Padded array
        """
        _TorchBackend._check_torch()
        return torch.nn.functional.pad(data, pad_samples, "constant")

    @staticmethod
    def full(shape: Tuple[int, ...], fill_value: Any) -> TorchArray:
        """Create an array filled with a specified value

        Args:
            shape:
                Tuple[int, ...], Shape of the array
            fill_value:
                Any, Value to fill the array with

        Returns:
            TorchArray, Array filled with the specified value
        """
        _TorchBackend._check_torch()
        return torch.full(shape, fill_value)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> TorchArray:
        """Create an array of zeros

        Args:
            shape:
                Tuple[int, ...], Shape of the array

        Returns:
            TorchArray, Array of zeros
        """
        _TorchBackend._check_torch()
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
        _TorchBackend._check_torch()
        return torch.stack(data, axis)

    # TODO Remove this backwards compatibility set of aliases when larger refactor
    #  complete
    pad_func = pad
    zeros_func = zeros
    full_func = full
    cat_func = cat
    stack_func = stack


class CPUTorchBackend(_TorchBackend):
    """Implementation of array operations using PyTorch tensors on CPU."""

    DEVICE = "cpu"


class GPUTorchBackend(_TorchBackend):
    """Implementation of array operations using PyTorch tensors on GPU."""

    DEVICE = "cuda"
