from dataclasses import dataclass
from collections import deque
import numpy as np

from sgn.base import *

from .audioadapter import *
from .buffer import *
from .offset import *
from .time import *
from .slice_tools import *

@dataclass
class _TSTransSink():

    def __post_init__(self):
        self._is_aligned = False
        self.inbufs = {p:deque() for p in self.sink_pads}
        self.preparedframes = {p:None for p in self.sink_pads}
        self.at_EOS = False
        self._last_ts = {p:None for p in self.sink_pads}
        self._last_offset = {p:None for p in self.sink_pads}

    def pull(self, pad, bufs):
        self.at_EOS |= bufs.EOS

        # extend and check the buffers
        self._sanity_check(bufs, pad)
        self.inbufs[pad].extend(bufs)

        # align if possible
        self._align()

        # put in heartbeat buffer if not aligned
        if not self._is_aligned:
            self.preparedframes[pad] = TSFrame(EOS=self.at_EOS, buffers = [SeriesBuffer(offset=0, noffset=0, offset_ref_t0=self.earliest, data=None, is_gap=True)])
        # Else pack all the buffers
        else:
            out = []
            min_latest = self.min_latest
            for b in tuple(self.inbufs[pad]):
                if b <= min_latest:
                    out.append(self.inbufs[pad].popleft())
            if ( buf := self.inbufs[pad].popleft() ) is not None:
                if buf.t0 < min_latest:
                    l,r = buf.split(min_latest)
                    self.inbufs[pad].appendleft(r)
                    out.append(l)
                else:# Yes this condition is silly
                    self.inbufs[pad].appendleft(buf)
            self.preparedframes[pad] = TSFrame(EOS=self.at_EOS, buffers = out)

        if self.timeout(pad):
            raise ValueError("pad %s has timed out" % pad.name)

    def _sanity_check(self, bufs, pad):
        if self._last_ts[pad] is not None and self._last_offset[pad] is not None:
            assert bufs[0].offset == self._last_offset[pad]
            assert bufs[0].end == self._last_ts[pad]
            self._last_offset[pad] = bufs[-1].end_offset
            self._last_ts[pad] = bufs[-1].end

    def _align(self):

        def slice_from_pad(inbufs):
            if len(inbufs) > 0:
                return TSSlice(inbufs[0].t0, inbufs[-1].end)
            else:
                return TSSlice(-1,-1)

        def __can_align(self = self):
            return TSSlice.intersection([slice_from_pad(self.inbufs[p]) for p in self.inbufs])

        if not self._is_aligned and __can_align():
            self._is_aligned = True
            old = self.earliest
            for p in self.inbufs:
                if self.inbufs[p][0].t0 != old: 
                    buf = self.inbufs[p][0].pad_buffer(t0 = old)
                    self.inbufs[p].appendleft(buf)

    def timeout(self, pad):
        assert len(self.inbufs[pad]) > 0
        return self.inbufs[pad][-1].end < (self.latest - self.max_age)

    def latest_by_pad(self, pad):
        return self.inbufs[pad][-1].t0 if self.inbufs[pad] else -1

    def earliest_by_pad(self, pad):
        return self.inbufs[pad][0].t0 if self.inbufs[pad] else -1

    @property
    def latest(self):
        return max(self.latest_by_pad(n) for n in self.inbufs)

    @property
    def earliest(self):
        return min(self.earliest_by_pad(n) for n in self.inbufs)

    @property
    def min_latest(self):
        return min(self.latest_by_pad(n) for n in self.inbufs)

@dataclass
class TSTransform(TransformElement, _TSTransSink):

    max_age: int = None
    pull = _TSTransSink.pull

    def __post_init__(self):
        TransformElement.__post_init__(self)
        _TSTransSink.__post_init__(self)

    def transform(self, pad):
        raise NotImplementedError 

@dataclass
class TSSink(SinkElement, _TSTransSink):

    max_age: int = None
    pull = _TSTransSink.pull

    def __post_init__(self):
        SinkElement.__post_init__(self)
        _TSTransSink.__post_init__(self)

