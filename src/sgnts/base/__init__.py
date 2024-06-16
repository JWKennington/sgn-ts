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
class _TSTransSink:

    def __post_init__(self):
        self._is_aligned = False
        self.inbufs = {p: deque() for p in self.sink_pads}
        self.preparedframes = {p: None for p in self.sink_pads}
        self.at_EOS = False
        self._last_ts = {p: None for p in self.sink_pads}
        self._last_offset = {p: None for p in self.sink_pads}
        self.__pulled = {p: False for p in self.sink_pads}

    def pull(self, pad, bufs):
        self.at_EOS |= bufs.EOS

        # extend and check the buffers
        self._sanity_check(bufs, pad)
        self.inbufs[pad].extend(bufs)
        self.__pulled[pad] = True
        if self.timeout(pad):
            raise ValueError("pad %s has timed out" % pad.name)

        if all(self.__pulled.values()):
            self.__post_pull()

    def __post_pull(self):
        # Reset
        self.__pulled = {p: False for p in self.sink_pads}

        # align if possible
        self._align()

        # put in heartbeat buffer if not aligned
        if not self._is_aligned:
            for pad in self.sink_pads:
                self.preparedframes[pad] = TSFrame(
                    EOS=self.at_EOS,
                    buffers=[
                        SeriesBuffer(
                            offset=self.earliest,
                            sample_rate=self.inbufs[pad][0].sample_rate,
                            data=None,
                            shape=self.inbufs[pad][0].shape[:-1] + (0,)
                        )
                    ],
                )
        # Else pack all the buffers
        else:
            min_latest = self.min_latest
            for pad in self.sink_pads:
                out = []
                for b in tuple(self.inbufs[pad]):
                    if b.end_offset <= min_latest:
                        out.append(self.inbufs[pad].popleft())
                if len(self.inbufs[pad]) > 0:
                    buf = self.inbufs[pad].popleft()
                    if buf.offset < min_latest:
                        l, r = buf.split(min_latest)
                        self.inbufs[pad].appendleft(r)
                        out.append(l)
                    else:  # Yes this condition is silly
                        self.inbufs[pad].appendleft(buf)
                assert len(out) > 0
                self.preparedframes[pad] = TSFrame(EOS=self.at_EOS, buffers=out)


    def _sanity_check(self, bufs, pad):
        if self._last_ts[pad] is not None and self._last_offset[pad] is not None:
            assert bufs[0].offset == self._last_offset[pad]
            assert bufs[0].end == self._last_ts[pad]
            self._last_offset[pad] = bufs[-1].end_offset
            self._last_ts[pad] = bufs[-1].end

    def _align(self):

        def slice_from_pad(inbufs):
            if len(inbufs) > 0:
                return TSSlice(inbufs[0].offset, inbufs[-1].end_offset)
            else:
                return TSSlice(-1, -1)

        def __can_align(self=self):
            return TSSlices(
                [slice_from_pad(self.inbufs[p]) for p in self.inbufs]
            ).intersection()

        if not self._is_aligned and __can_align():
            self._is_aligned = True
            old = self.earliest
            for p in self.inbufs:
                if self.inbufs[p][0].offset != old:
                    buf = self.inbufs[p][0].pad_buffer(off=old)
                    self.inbufs[p].appendleft(buf)

    def timeout(self, pad):
        assert len(self.inbufs[pad]) > 0
        return self.inbufs[pad][-1].end < (self.latest - self.max_age)

    def latest_by_pad(self, pad):
        return self.inbufs[pad][-1].end_offset if self.inbufs[pad] else -1

    def earliest_by_pad(self, pad):
        return self.inbufs[pad][0].offset if self.inbufs[pad] else -1

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


@dataclass
class TSSource(SourceElement):
    """
    A time-series source that generates data in fixed-size buffers.

    Parameters:
    -----------
    t0: float
        start time of first buffer, in seconds
    num_samples: int
        number of samples to produce per Frame. 
        If None, the value from Offset.stridesamples will be used
    rate: int
        the sample rate of the data
    """

    t0: float = 0
    num_samples: int = None
    rate: int = 2048

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.rate, int)
        assert isinstance(self.num_samples, int)
        # FIXME should we be more careful about this?
        self.offset = {p: Offset.fromsec(self.t0 - Offset.offset_ref_t0) for p in self.source_pads}
        if self.num_samples is None:
            self.num_samples = Offset.stridesamples(self.rate)
