from dataclasses import dataclass

import numpy as np
from sgn import validator

from sgnts.base import Time, TSFrame, TSSink
from sgnts.decorators import sink


@dataclass(kw_only=True)
class DumpSeriesSink(TSSink):
    """A sink element that dumps time series data to a txt file.

    Args:
        fname:
            str, output file name
        verbose:
            bool, be verbose
    """

    fname: str
    verbose: bool = False

    def configure(self) -> None:
        # overwrite existing file
        with open(self.fname, "w"):
            pass

    @validator.single_pad
    def validate(self) -> None:
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

    @sink.single_pad
    def process(self, input_frame: TSFrame) -> None:
        """Write out time-series data."""
        if input_frame.EOS:
            self.mark_eos(self.sink_pads[0])
        if self.verbose is True:
            print(input_frame)
        for buf in input_frame:
            if not buf.is_gap:
                self.write_to_file(buf)
