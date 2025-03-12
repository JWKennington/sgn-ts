from dataclasses import dataclass

from sgnts.base import Offset, TSSink
from sgnts.utils import gpsnow


@dataclass
class NullSeriesSink(TSSink):
    """A series sink that does precisely nothing.

    Args:
        verbose:
            bool, print frames as they pass through the internal pad

    """

    verbose: bool = False

    def internal(self) -> None:
        """Print frames if verbose."""
        super().internal()
        for sink_pad in self.sink_pads:
            frame = self.preparedframes[sink_pad]
            if frame.EOS:
                self.mark_eos(sink_pad)
            if self.verbose is True:
                print(frame)
                latency = gpsnow() - Offset.tosec(
                    frame.offset + Offset.SAMPLE_STRIDE_AT_MAX_RATE
                )
                print(f"latency: {latency} s")
