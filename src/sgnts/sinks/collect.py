from dataclasses import dataclass

from sgn import CollectSink

from sgnts.base import TSFrame


@dataclass
class TSFrameCollectSink(CollectSink):
    """Sink that collects input SeriesBuffers

    sgn.CollectSink with an additional method `out_frames` that will
    return a dictionary, keyed by sink pad names, where the values are
    single TSFrames containing all buffers collected on the sink pads
    during pipeline operation.

    """

    def __post_init__(self):
        self.extract_data = False
        self.skip_empty = False
        super().__post_init__()

    def out_frames(self) -> dict[str, TSFrame]:
        """The collected frames."""
        out = {}
        for pad_name, frames in self.collects.items():
            buffers = []
            for frame in frames:
                buffers.extend(frame.buffers)
            out[pad_name] = TSFrame(buffers=buffers)
        return out
