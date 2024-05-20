from sgn.sinks import *
from ..base import Time
import numpy as np

@dataclass
class FakeSeriesSink(SinkElement):
    """
    A fake sink element
    """
    def __post_init__(self):
        self.inbuf = None
        super().__post_init__()
        self.at_eos = {p:False for p in self.sink_pads}

    def get_buffer(self, pad, buf):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        self.inbuf = buf
        self.at_eos[pad] = self.inbuf.EOS
        print ("buffer flow: ", "%s -> '%s' offset %d time %d shape %s" % (self.inbuf.metadata["name"], pad.name, self.inbuf.offset, self.inbuf.t0, self.inbuf.data.shape))
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
    fname: str = 'out.txt'

    def __post_init__(self):
        self.inbuf = None
        super().__post_init__()
        self.at_eos = {p:False for p in self.sink_pads}
        self.f = open(self.fname, "w")

    def write_to_file(self, buf):
        t0 = buf.t0
        duration = buf.duration
        data = buf.data
        ts = np.linspace(t0/Time.SECONDS,(t0+duration)/Time.SECONDS,data.shape[-1],endpoint=False)
        out = np.vstack([ts, data]).T
        np.savetxt(self.f, out)

    def get_buffer(self, pad, buf):
        """
        getting the buffer on the pad just modifies the name to show this final
        graph point and the prints it to prove it all works.
        """
        self.inbuf = buf
        self.at_eos[pad] = self.inbuf.EOS
        if self.inbuf.data is None:
            print ("buffer flow: ", "%s -> '%s' offset %d time %d " % (self.inbuf.metadata["name"], pad.name, self.inbuf.offset, self.inbuf.t0))
            return
        else:
            print ("buffer flow: ", "%s -> '%s' offset %d time %d shape %s" % (self.inbuf.metadata["name"], pad.name, self.inbuf.offset, self.inbuf.t0, self.inbuf.data.shape))
            self.write_to_file(buf)

    @property
    def EOS(self):
        """
        If buffers on any sink pads are End of Stream (EOS), then mark this whole element as EOS
        """
        return any(self.at_eos.values())

sinks_registry += ("FakeSeriesSink","DumpSeriesSink")
