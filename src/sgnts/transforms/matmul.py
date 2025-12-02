from __future__ import annotations

from dataclasses import dataclass

from sgn import validator

from sgnts.base import (
    Array,
    ArrayBackend,
    NumpyBackend,
    TSTransform,
)


@dataclass
class Matmul(TSTransform):
    """Performs matrix multiplication with provided matrix.

    Args:
        matrix:
            Array | None, the matrix to multiply the data with, out = matrix x data
        backend:
            type[ArrayBackend], the array backend for array operations
    """

    matrix: Array | None = None
    backend: type[ArrayBackend] = NumpyBackend

    def configure(self) -> None:
        assert self.matrix is not None
        self.shape = self.matrix.shape

    @validator.one_to_one
    def validate(self) -> None:
        assert self.matrix is not None, "Matrix must be provided for MatMul operation"

    def internal(self) -> None:
        """Perform matrix multiplication on non-gap data."""
        super().internal()

        _, input_frame = self.next_input()
        _, output_frame = self.next_output()

        for buf in input_frame:
            if buf.is_gap:
                data = None
                shape = self.shape[:-1] + (buf.samples,)
            else:
                data = self.backend.matmul(self.matrix, buf.data)
                shape = data.shape

            buf = buf.replace(data=data, shape=shape)
            output_frame.append(buf)
