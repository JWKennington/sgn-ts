from dataclasses import dataclass

import numpy as np

from ..base import Time, TSSink


@dataclass
class FakeSeriesSink(TSSink):
    """
    A fake sink element
    """

    print_message: str = "''"
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p: 0 for p in self.sink_pads}

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        super().pull(pad, bufs)
        self.cnt[pad] += 1
        bufs = self.preparedframes[pad]
        if bufs.EOS:
            self.mark_eos(pad)
        if self.verbose is True:
            print(self.cnt[pad], bufs)

    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())


@dataclass
class DumpSeriesSink(TSSink):
    """
    A sink element that dumps time series data to a txt file

    Parameters:
    -----------
    fname: str
        output file name
    """

    fname: str = "out.txt"

    def __post_init__(self):
        super().__post_init__()
        # overwrite existing file
        with open(self.fname, "w") as f:
            pass

    def write_to_file(self, buf):
        t0 = buf.t0
        duration = buf.duration
        data = buf.data
        data = data.reshape(-1, data.shape[-1])
        ts = np.linspace(
            t0 / Time.SECONDS,
            (t0 + duration) / Time.SECONDS,
            data.shape[-1],
            endpoint=False,
        )
        out = np.vstack([ts, data]).T
        with open(self.fname, "a") as f:
            np.savetxt(f, out)

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        super().pull(pad, bufs)
        bufs = self.preparedframes[pad]
        if bufs.EOS:
            self.mark_eos(pad)
        print(bufs)
        for buf in bufs:
            if not buf.is_gap:
                self.write_to_file(buf)
