from __future__ import annotations

from functools import wraps
from typing import Any, Optional, Sequence, Tuple, Union

import numpy

from sgnts.base.array_backend import ArrayBackend

NumpyArray = numpy.ndarray


class NumpyBackend(ArrayBackend):
    """Implementation of array operations using numpy."""

    DEVICE = "cpu"
    DTYPE = numpy.float64

    @staticmethod
    @wraps(numpy.all)
    def all(*args, **kwargs):
        return numpy.all(*args, **kwargs)

    @staticmethod
    def arange(stop: float, start: float = 0, step: float = 1) -> NumpyArray:
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
            NumpyArray, an array of evenly spaced values
        """
        return numpy.arange(start=start, stop=stop, step=step)

    @staticmethod
    def cat(data: Sequence[NumpyArray], axis: Optional[int]) -> NumpyArray:
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
    def ones(shape: Tuple[int, ...]) -> NumpyArray:
        """Create an array of ones.

        Args:
            shape:
                Tuple[int, ...]: Shape of the array

        Returns:
            NumpyArray: Array of ones
        """
        return numpy.ones(shape)

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

    @classmethod
    def stack(cls, data: Sequence[NumpyArray], axis: int = 0) -> NumpyArray:
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
    def sum(
        a: NumpyArray, axis: Optional[Union[int, tuple[int, ...]]] = None
    ) -> NumpyArray:
        """Sum of array elements over a given axis.

        Args:
            a:
                NumpyArray, elements to sum
            axis:
                Optional[int, tuple[int, ...]], axis or axes along which a sum is
                performed

        Returns:
            NumpyArray, an array of summed elements
        """
        return numpy.sum(a, axis=axis)

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
