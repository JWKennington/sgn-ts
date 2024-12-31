from dataclasses import dataclass
from functools import wraps

from sgn.base import SourcePad

from sgnts.base import SeriesBuffer, TSFrame, TSTransform


@dataclass
class Amplify(TSTransform):
    """Amplify data by a factor.

    Args:
        factor:
            float, the factor to multiply the data with
    """

    factor: float = 1

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"
        self.sink_pad = self.sink_pads[0]

    # FIXME: wraps are not playing well with mypy.  For now ignore and hope
    # that a future version of mypy will be able to handle this
    @wraps(TSTransform.new)
    def new(self, pad: SourcePad) -> TSFrame:  # type: ignore
        outbufs = []
        # loop over the input data, only amplify non-gap data
        frame = self.preparedframes[self.sink_pad]
        for inbuf in frame:
            if inbuf.is_gap:
                data = None
            else:
                data = inbuf.data * self.factor

            outbuf = SeriesBuffer(
                offset=inbuf.offset,
                sample_rate=inbuf.sample_rate,
                data=data,
                shape=inbuf.shape,
            )
            outbufs.append(outbuf)

        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
