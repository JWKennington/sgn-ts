from dataclasses import dataclass

from sgn.base import SinkPad

from sgnts.base import Offset, TSFrame, TSSink
from sgnts.utils import gpsnow


@dataclass
class NullSeriesSink(TSSink):
    """A series sink that does precisely nothing.

    Args:
        verbose:
            bool, print frames as they pass through the internal pad

    """

    verbose: bool = False

    def process(self, input_frames: dict[SinkPad, TSFrame]) -> None:
        """Print frames if verbose."""
        for sink_pad, frame in input_frames.items():
            if frame.EOS:
                self.mark_eos(sink_pad)
            if self.verbose:
                print(f"{sink_pad.name}:")
                print(f"  {frame}")
                latency = gpsnow() - Offset.tosec(
                    frame.offset + Offset.SAMPLE_STRIDE_AT_MAX_RATE
                )
                print(f"  latency: {latency} s")
