from __future__ import annotations

from dataclasses import dataclass

from sgn import validator

from sgnts.base import (
    Array,
    ArrayBackend,
    NumpyBackend,
    TSCollectFrame,
    TSFrame,
    TSTransform,
)
from sgnts.decorators import transform


@dataclass(kw_only=True)
class Matmul(TSTransform):
    """Performs matrix multiplication with provided matrix.

    Args:
        matrix:
            Array, the matrix to multiply the data with, out = matrix x data
        backend:
            type[ArrayBackend], the array backend for array operations
    """

    matrix: Array
    backend: type[ArrayBackend] = NumpyBackend

    def configure(self) -> None:
        self.shape = self.matrix.shape

    @validator.one_to_one
    def validate(self) -> None:
        pass

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Perform matrix multiplication on non-gap data."""
        for buf in input_frame:
            if buf.is_gap:
                data = None
                shape = self.shape[:-1] + (buf.samples,)
            else:
                data = self.backend.matmul(self.matrix, buf.data)
                shape = data.shape

            buf = buf.copy(data=data, shape=shape)
            output_frame.append(buf)
