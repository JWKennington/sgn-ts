from dataclasses import dataclass

from sgn.base import SourcePad

from sgnts.base import ArrayBackend, NumpyBackend, SeriesBuffer, TSFrame, TSTransform


@dataclass
class SumIndex(TSTransform):
    """Sum array values over slices in the zero-th dimension.

    Args:
        sl:
            list[slice], the slices to sum over
        backend:
            type[ArrayBackend], the wrapper around array operations.
    """

    sl: list[slice] = None
    backend: type[ArrayBackend] = NumpyBackend

    def __post_init__(self):
        super().__post_init__()
        for sl in self.sl:
            assert isinstance(sl, slice)

    def transform(self, pad: SourcePad) -> TSFrame:
        """Sum the data over slices in the zero-th dimension. The zero-th dimension
        will now have a length of len(self.sl).

        Args:
            pad:
                SourcePad, the source pad to produce the transformed frame

        Returns:
            TSFrame, the output TSFrame
        """
        frame = self.preparedframes[self.sink_pads[0]]

        outbufs = []
        for buf in frame:
            if buf.is_gap:
                out = None
            else:
                data = buf.data
                data_all = []
                for sl in self.sl:
                    if sl.stop - sl.start == 1:
                        data_all.append((data[sl.start, :, :]))
                    else:
                        data_all.append(self.backend.sum(data[sl, :, :], dim=0))

                out = self.backend.stack(data_all)

            outbuf = SeriesBuffer(
                offset=buf.offset,
                sample_rate=buf.sample_rate,
                data=out,
                shape=(len(self.sl),) + buf.shape[-2:],
            )
        outbufs.append(outbuf)

        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
