from dataclasses import dataclass

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
        assert len(self.sink_pads) == 1 and len(self.source_pads) == 1, (
            f"Amplify requires exactly one sink pad and one source pad, "
            f"got {len(self.sink_pads)} sink pads and "
            f"{len(self.source_pads)} source pads"
        )
        self.sink_pad = self.sink_pads[0]

    def new(self, pad: SourcePad) -> TSFrame:
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
