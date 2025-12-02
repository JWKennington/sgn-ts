from __future__ import annotations

from dataclasses import dataclass

from sgn import validator

from sgnts.base import ArrayBackend, NumpyBackend, TSTransform


@dataclass
class SumIndex(TSTransform):
    """Sum array values over slices in the zero-th dimension.

    Args:
        sl:
            list[slice], the slices to sum over
        backend:
            type[ArrayBackend], the wrapper around array operations.
    """

    sl: list[slice] | None = None
    backend: type[ArrayBackend] = NumpyBackend

    @validator.one_to_one
    def validate(self) -> None:
        assert (
            self.sl is not None
        ), "Slice list (sl) must be provided for SumIndex operation"
        for sl in self.sl:
            assert isinstance(sl, slice)

    def internal(self) -> None:
        """Sum array values over slices."""
        super().internal()

        _, input_frame = self.next_input()
        _, output_frame = self.next_output()

        for buf in input_frame:
            if buf.is_gap:
                data = None
                # NOTE mypy complains about None not being iterable but None should
                # actually be impossible at this point.
                assert (
                    self.sl is not None
                ), "Slice list (sl) should not be None when creating output shape"
                shape = (len(self.sl),) + buf.shape[-2:]
            else:
                data_all = []
                # NOTE mypy complains about None not being iterable but None
                # should actually be impossible at this point.
                assert (
                    self.sl is not None
                ), "Slice list (sl) should not be None during processing"
                for sl in self.sl:
                    if sl.stop - sl.start == 1:
                        data_all.append((buf.data[sl.start, :, :]))
                    else:
                        data_all.append(self.backend.sum(buf.data[sl, :, :], axis=0))

                data = self.backend.stack(data_all)
                shape = data.shape

            buf = buf.replace(data=data, shape=shape)
            output_frame.append(buf)
