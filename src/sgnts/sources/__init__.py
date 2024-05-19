from sgn.sources import *
from .. base import *
import numpy as np

@dataclass
class RandomSeriesSrc(SourceElement):
    """
    A time-series source that generates random data in one second buffers.

    Parameters:
    -----------
    num_buffers: int 
        is required and sets how many buffers will be created before setting "EOS"
    shape: tuple
        the shape of the data array
    duration: float
        duration of the data buffer, in seconds

    """
    num_buffers: int = 0
    shape: tuple = (2048,)
    duration: float = 1
    signal_type: str = 'white'
    fsin: float = 5

    def __post_init__(self):
        super().__post_init__()
        self.cnt = {p:0 for p in self.source_pads}
        self.offset = {p:0 for p in self.source_pads}

    def create_data(self, offset):
        if self.signal_type == 'white':
            return np.random.rand(*self.shape)
        elif self.signal_type == 'sin' or self.signal_type == 'sine':
            t0 = Offset.offset2sec(offset)
            return np.sin(self.fsin*np.linspace(t0,t0+self.duration,self.shape[-1],endpoint=False))
        else:
            raise ValueError("Unknown signal type")


    def new_buffer(self, pad):
        """
        New buffers are created on "pad" with an instance specific count and a
        name derived from the pad name. "EOS" is set if we have surpassed the requested
        number of buffers.
        """
        self.cnt[pad] += 1
        noffset = int(OFFSET_RATE*self.duration)
        data = self.create_data(self.offset[pad])
        outbuf = SeriesBuffer(offset = self.offset[pad],
                noffset = noffset,
                offset_ref_t0 = 0,
                data = data, 
                metadata = {"cnt": self.cnt, "name":"'%s'" % pad.name},
                EOS =  self.cnt[pad] > self.num_buffers )

        self.offset[pad] += noffset

        return outbuf


sources_registry += ("RandomSeriesSrc",)
