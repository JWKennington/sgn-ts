from dataclasses import dataclass

import numpy as np

from sgnts.base import Time, TSSink


@dataclass
class DumpSeriesSink(TSSink):
    """A sink element that dumps time series data to a txt file.

    Args:
        fname:
            str, output file name
        verbose:
            bool, be verbose
    """

    fname: str = "out.txt"
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        if len(self.sink_pads) != 1:
            # FIXME: When will we use multiple sink pads here?
            # Do we want to support writing multiple frames into the same file?
            raise ValueError("Only supports one sink pad.")

        self.sink_pad = self.sink_pads[0]

        # overwrite existing file
        with open(self.fname, "w"):
            pass

    def write_to_file(self, buf) -> None:
        """Write time series data to txt file.

        Args:
            buf:
                SeriesBuffer, the buffer with time series data to write out
        """
        t0 = buf.t0
        duration = buf.duration
        data = buf.data
        # FIXME: How to write multi-dimensional data?
        data = data.reshape(-1, data.shape[-1])
        ts = np.linspace(
            t0 / Time.SECONDS,
            (t0 + duration) / Time.SECONDS,
            data.shape[-1],
            endpoint=False,
        )
        out = np.vstack([ts, data]).T
        with open(self.fname, "ab") as f:
            np.savetxt(f, out)

    def internal(self) -> None:
        """Write out time-series data."""
        super().internal()
        sink_pad = self.sink_pad
        frame = self.preparedframes[sink_pad]
        if frame.EOS:
            self.mark_eos(sink_pad)
        if self.verbose is True:
            print(frame)
        for buf in frame:
            if not buf.is_gap:
                self.write_to_file(buf)
