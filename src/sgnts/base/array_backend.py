"""Class for specifying array operation implementations. Specifically,
we define a set of generic array operations that can be implemented in various
backends (e.g., numpy, pytorch, tensorflow, etc.). This allows us to write
generic code that can be run on different backends without modification.

The operations are defined as static methods in the `ArrayBackend` class, and
must be implemented in subclasses. The current set of operations includes:

- `arange`: Create an array of evenly spaced values
- `cat`: Concatenate arrays along a specified axis
- `full`: Create an array filled with a specified value
- `matmul`: Perform matrix multiplication of two arrays
- `ones`: Create an array of ones
- `pad`: Pad an array with zeros
- `stack`: Stack arrays along a new axis
- `sum`: Sum of array elements over a given axis
- `zeros`: Create an array of zeros
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

# Alias for a generic array type
Array = Any


class ArrayBackend:
    """Base class for array operation implementations. Subclasses should
    implement the array operations as static methods.
    """

    @staticmethod
    def arange(stop: float, start: float = 0, step: float = 1) -> Array:
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
            Array, an array of evenly spaced values
        """
        raise NotImplementedError

    @staticmethod
    def cat(data: Sequence[Array], axis: int) -> Array:
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

    @staticmethod
    def ones(shape: Tuple[int, ...]) -> Array:
        """Create an array of ones.

        Args:
            shape:
                Tuple[int, ...]: Shape of the array

        Returns:
            Array: Array of ones
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

    @classmethod
    def stack(cls, data: Sequence[Array], axis: int = 0) -> Array:
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
    def sum(a: Array, axis: Optional[Union[int, tuple[int, ...]]] = None) -> Array:
        """Sum of array elements over a given axis.

        Args:
            a:
                Array, elements to sum
            axis:
                Optional[int, tuple[int, ...]], axis or axes along which a sum is
                performed

        Returns:
            Array, an array of summed elements
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
