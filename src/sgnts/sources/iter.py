"""Frame iterator source element."""

from dataclasses import dataclass

from sgn.base import SourcePad

from sgnts.base import TSFrame, TSSource


@dataclass(kw_only=True)
class TSIterSource(TSSource):
    """A source that iterates through a provided list of TSFrames.

    This is useful for testing, allowing you to provide pre-constructed
    frames (including multi-buffer frames) to a pipeline.

    Args:
        frames:
            list[TSFrame], the frames to iterate through. After the last
            frame is sent, an EOS frame will be sent automatically.
    """

    frames: list[TSFrame]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.frames[-1].EOS = True
        self.last_frame = self.frames[-1]
        self.frame_iter = iter(self.frames)

    def new(self, pad: SourcePad) -> TSFrame:
        try:
            return next(self.frame_iter)
        except StopIteration:
            # Return heartbeats after iterator is exhausted
            return self.last_frame.heartbeat()
