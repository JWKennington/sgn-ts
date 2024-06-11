from dataclasses import dataclass

import numpy as np
from sgn.sinks import SinkElement

from ..base import Time


@dataclass
class FakeSeriesSink(SinkElement):
    """
    A fake sink element
    """

    print_message: str = "''"

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        if bufs.EOS:
            self.mark_eos(pad)
        print (bufs)
        #outstr = "%s :: " % bufs.metadata['__graph__']
        #for buf in bufs:
        #    outstr += str(buf)
        #print(outstr)

    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())


@dataclass
class DumpSeriesSink(SinkElement):
    """
    A sink element that dumps time series data to a txt file
    """

    fname: str = "out.txt"

    def __post_init__(self):
        super().__post_init__()
        self.f = open(self.fname, "w")

    def write_to_file(self, buf):
        t0 = buf.t0
        duration = buf.duration
        data = buf.data
        ts = np.linspace(
            t0 / Time.SECONDS,
            (t0 + duration) / Time.SECONDS,
            data.shape[-1],
            endpoint=False,
        )
        out = np.vstack([ts, data]).T
        np.savetxt(self.f, out)

    def pull(self, pad, bufs):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        if bufs.EOS:
            self.mark_eos(pad)

        for buf in bufs:
            if buf.data is None:
                print(
                    "buffer flow: ",
                    "%s -> '%s' offset %d time %d "
                    % (
                        bufs.metadata["name"],
                        pad.name,
                        buf.offset,
                        buf.t0,
                    ),
                )
                return
            else:
                print(
                    "buffer flow: ",
                    "%s -> '%s' offset %d time %d shape %s"
                    % (
                        bufs.metadata["name"],
                        pad.name,
                        buf.offset,
                        buf.t0,
                        buf.data.shape,
                    ),
                )
                self.write_to_file(buf)
