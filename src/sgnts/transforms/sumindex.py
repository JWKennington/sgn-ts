from __future__ import annotations

from dataclasses import dataclass

from sgn import validator

from sgnts.base import ArrayBackend, NumpyBackend, TSCollectFrame, TSFrame, TSTransform
from sgnts.decorators import transform


@dataclass(kw_only=True)
class SumIndex(TSTransform):
    """Sum array values over slices in the zero-th dimension.

    Args:
        sl:
            list[slice], the slices to sum over
        backend:
            type[ArrayBackend], the wrapper around array operations.
    """

    sl: list[slice]
    backend: type[ArrayBackend] = NumpyBackend

    @validator.one_to_one
    def validate(self) -> None:
        for sl in self.sl:
            assert isinstance(sl, slice)

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Sum array values over slices."""
        for buf in input_frame:
            if buf.is_gap:
                data = None
                shape = (len(self.sl),) + buf.shape[-2:]
            else:
                data_all = []
                for sl in self.sl:
                    if sl.stop - sl.start == 1:
                        data_all.append((buf.data[sl.start, :, :]))
                    else:
                        data_all.append(self.backend.sum(buf.data[sl, :, :], axis=0))

                data = self.backend.stack(data_all)
                shape = data.shape

            buf = buf.copy(data=data, shape=shape)
            output_frame.append(buf)
