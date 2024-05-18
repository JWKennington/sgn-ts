from sgn.sources import *
from .. base import *
import numpy as np

@dataclass
class WhiteSeriesSrc(SourceElement):
    """
    A time-series source that generates random data in one second buffers.
    "num_buffers" is required and sets how many buffers will be created before setting "EOS"
    """
    num_buffers: int = 0
    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p:0 for p in self.source_pads}
        self.offset = {p:0 for p in self.source_pads}
    def new_buffer(self, pad):
        """
        New buffers are created on "pad" with an instance specific count and a
        name derived from the pad name. "EOS" is set if we have surpassed the requested
        number of buffers.
        """
        self.cnt[pad] += 1
        outbuf = SeriesBuffer(offset = self.offset[pad],
                noffset = 16384,
                offset_ref_t0 = 0,
                data = np.random.rand(2048), 
		metadata = {"cnt": self.cnt, "name":"'%s'" % pad.name},
                EOS =  self.cnt[pad] > self.num_buffers )

        self.offset[pad] += 16384

        return outbuf


sources_registry += ("WhiteSeriesSrc",)
