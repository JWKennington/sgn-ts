from dataclasses import dataclass

from sgn import validator
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

    @validator.one_to_one
    def validate(self) -> None:
        pass

    def new(self, pad: SourcePad) -> TSFrame:
        outbufs = []
        # loop over the input data, only amplify non-gap data
        sink_pad = self.sink_pads[0]
        frame = self.preparedframes[sink_pad]
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
