from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import torch

from sgnts.base.array_backend import ArrayBackend

# Set some global PyTorch settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

TorchArray = torch.Tensor


class TorchBackend(ArrayBackend):
    """Implementation of array operations using PyTorch tensors."""

    # FIXME: How to handle different device/dtypes in the same pipeline?
    DTYPE = torch.float32
    DEVICE = torch.device("cpu")

    @staticmethod
    def all(input: torch.Tensor, out: Optional[torch.Tensor] = None):
        """Returns true if all elements are true"""
        return torch.all(input=input, out=out)

    @classmethod
    def arange(cls, stop: float, start: float = 0, step: float = 1) -> torch.Tensor:
        """Returns a 1-D array with values from the interval `[start, stop)`, taken
        with common difference `step` begining from `start`.

        Args:
            stop:
                float, the stopping value for the set of points
            start:
                float, default 0, the starting value for the set of points
            step:
                float, default 1, the gap between each pair of adjacent points

        Returns:
            Tensor, an array of evenly spaced values
        """
        return torch.arange(
            start=start, end=stop, step=step, device=cls.DEVICE, dtype=cls.DTYPE
        )

    @staticmethod
    def cat(data: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        """Concatenate arrays along a specified axis

        Args:
            data:
                Iterable[Tensor], Arrays to concatenate, all with the same shape
            axis:
                int, Axis along which to concatenate the arrays

        Returns:
            Tensor, concatenated array
        """
        return torch.cat(tuple(data), dim=axis)

    @classmethod
    def full(cls, shape: Tuple[int, ...], fill_value: Any) -> torch.Tensor:
        """Create an array filled with a specified value

        Args:
            shape:
                Tuple[int, ...], Shape of the array
            fill_value:
                Any, Value to fill the array with

        Returns:
            Tensor, Array filled with the specified value
        """
        return torch.full(shape, fill_value, device=cls.DEVICE, dtype=cls.DTYPE)

    @staticmethod
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication of two arrays.
            out = a x b

        Args:
            a:
                Tensor, the first array
            b:
                Tensor, the second array

        Returns:
            Tensor, the result of the matrix multiplication
        """
        return torch.matmul(a, b)

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create an array of ones.

        Args:
            shape:
                Tuple[int, ...]: Shape of the array

        Returns:
            Array: Array of ones
        """
        return torch.ones(shape, device=cls.DEVICE, dtype=cls.DTYPE)

    @staticmethod
    def pad(data: torch.Tensor, pad_samples: tuple[int, int]) -> torch.Tensor:
        """Pad an array with zeros

        Args:
            data:
                Tensor, Array to pad
            pad_samples:
                tuple[int, int], Number of zeros to pad at each end of the array

        Returns:
            Tensor, Padded array
        """
        return torch.nn.functional.pad(data, pad_samples, "constant")

    @classmethod
    def set_device(cls, device: Union[torch.device, str]) -> None:
        """Set the torch device.

        Args:
            device:
                str|device the device on which to create torch tensors
        """
        cls.DEVICE = torch.device(device)

    @classmethod
    def set_dtype(cls, dtype: torch.dtype) -> None:
        """Set the torch data type.

        Args:
            dtype:
                torch.dtype, the data type of the torch tensors
        """
        cls.DTYPE = dtype

    @staticmethod
    def stack(data: Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
        """Stack arrays along a new axis

        Args:
            data:
                Iterable[Tensor], Arrays to stack, all with the same shape
            axis:
                int, Axis along which to stack the arrays

        Returns:
            Tensor, Stacked array
        """
        return torch.stack(tuple(data), axis)

    @staticmethod
    def sum(
        a: torch.Tensor, axis: Optional[Union[int, tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """Sum of array elements over a given axis.

        Args:
            a:
                Tensor, elements to sum
            axis:
                Optional[int, tuple[int, ...]], axis or axes along which a sum is
                performed

        Returns:
            Tensor, an array of summed elements
        """
        return torch.sum(a, dim=axis)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create an array of zeros

        Args:
            shape:
                Tuple[int, ...], Shape of the array

        Returns:
            Tensor, Array of zeros
        """
        return torch.zeros(shape, device=cls.DEVICE, dtype=cls.DTYPE)
