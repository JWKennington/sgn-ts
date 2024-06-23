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
    """
    Base class for TSTransforms and TSSinks, will produce aligned frames
    in preparedframes. If overlap or stride is nonzero, will trigger
    the audioadapter to queue data, and make padded or strided frames
    in preparedframes.

    Parameters:
    -----------
    max_age: int
        the max age before timeout, in nanoseconds
    overlap: tuple[int, int]
        the overlap before and after the data segement to process,
        in samples
    stride: int
        the stride to produce, in samples
    pad_zeros_startup: bool
        when overlap is provided, whether to pad zeros in front of the
        first buffer, or wait until there is enough data.
    inrate: int
        the sample rate of the input data
    outrate: int
        the sample rate of output of the transform
    """

    max_age: int = None
    overlap: tuple[int, int] = (0, 0)
    stride: int = 0
    pad_zeros_startup: bool = False
    inrate: int = None
    outrate: int = None

    def __post_init__(self):
        self._is_aligned = False
        self.inbufs = {p: deque() for p in self.sink_pads}
        self.preparedframes = {p: None for p in self.sink_pads}
        self.at_EOS = False
        self._last_ts = {p: None for p in self.sink_pads}
        self._last_offset = {p: None for p in self.sink_pads}
        self.__pulled = {p: False for p in self.sink_pads}
        self.audioadapters = None
        if any(self.overlap) or self.stride:
            # we need audioadapters
            self.audioadapters = {p: Audioadapter() for p in self.sink_pads}
            self.pad_zeros_samples = 0
            if self.pad_zeros_startup is True:
                # at startup, pad zeros in front of the first buffer to
                # serve as history
                self.pad_zeros_samples = self.overlap[0]
            self.preparedoutframes = {p: None for p in self.sink_pads}

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

    def __adapter(self, pad, bufs):
        """
        Use the audioadapter to handle streaming scenarios such
        as padding with overlap before and after the target data,
        and fixed stride outputs.

        The self.preparedframes are padded with the requested
        padding.  This method also produces a self.preparedoutframes,
        that infers the metadata information for the output buffer,
        with the data initialized as None.  Downstream transforms
        can directly use the frames from self.preparedframes for
        computation, and then update the data in the buffer in
        self.preparedoutframes with the buffer.update_data(data)
        method.


        Example 1 upsampling:
        ----------------------
        kernel length = 17
        need to pad 8 samples before and after
        overlap = (8, 8)

        preparedframes:     ________................________
                                        for output
                            pad                     pad
                            samples=8               samples=8


        Example 2 correlation:
        ----------------------
        filter length = 16
        need to pad filter_length - 1 samples
        overlap = (15, 0)

        preparedframes:     ----------------........
                                            for output
                            pad
                            samples=15
        """
        a = self.audioadapters[pad]
        buf0 = bufs[0]

        # push all buffers in the frame into the audioadapter
        for buf in bufs:
            a.push(buf)

        # Check whether we have enough samples to produce a frame
        min_samples = sum(self.overlap) + (self.stride or 1) - self.pad_zeros_samples

        # figure out the offset for preparedframes and preparedoutframes
        offset = a.offset - Offset.fromsamples(self.pad_zeros_samples, buf0.sample_rate)
        outoffset = offset + Offset.fromsamples(self.overlap[0], buf0.sample_rate)
        preparedbufs = []
        print(a.size, min_samples, self.overlap, self.stride, self.pad_zeros_samples)
        if a.size < min_samples:
            # not enough samples to produce output yet
            # make a heartbeat buffer
            shape = buf0.shape[:-1] + (0,)
            outshape = shape
            preparedbufs.append(SeriesBuffer(offset=offset, sample_rate=buf0.sample_rate, data=None, shape=shape))
            # prepare output frames, one buffer per frame
            self.preparedoutframes[pad] = TSFrame(
                buffers=[
                    SeriesBuffer(
                        offset=outoffset,
                        sample_rate=(self.outrate or buf0.sample_rate),
                        data=None,
                        shape=outshape,
                    )
                ],
                EOS=self.at_EOS,
            )
        else:
            # We have enough samples, find out how many samples to copy
            # out of the audioadapter
            # copy all of the samples in the audioadapter
            if self.stride == 0:
                # provide all the data
                num_copy_samples = a.size
                nloop = 1
            else:
                num_copy_samples = min_samples
                nloop = 1 + (a.size - min_samples)//self.stride

            preparedoutbufs = []

            for i in range(nloop):
                if a.is_gap() is True:
                    # the whole audioadapter is a gap
                    data = None
                else:
                    # copy out samples from head of audioadapter
                    data = a.copy_samples(num_copy_samples)
                    if self.pad_zeros_samples > 0:
                        # pad zeros in front of buffer
                        data = pad_func(data, self.pad_zeros_samples)

                # flush out samples from head of audioadapter
                num_flush_samples = num_copy_samples - sum(self.overlap)
                a.flush_samples(num_flush_samples)

                shape = buf0.shape[:-1] + (num_copy_samples + self.pad_zeros_samples,)
                outsamples = num_flush_samples + self.pad_zeros_samples
                if self.outrate is not None and self.inrate is not None:
                    outsamples = int(outsamples * self.outrate / self.inrate)
                outshape = buf0.shape[:-1] + (outsamples,)

                # update next zeros padding
                self.pad_zeros_samples = -min(0, num_flush_samples)
                preparedbufs.append(SeriesBuffer(offset=offset, sample_rate=buf0.sample_rate, data=data, shape=shape))
                preparedoutbufs.append(
                    SeriesBuffer(
                        offset=outoffset,
                        sample_rate=(self.outrate or buf0.sample_rate),
                        data=None,
                        shape=outshape,
                    )
                    )
                offset += Offset.fromsamples(shape[-1], buf0.sample_rate)
                outoffset += Offset.fromsamples(outsamples, (self.outrate or buf0.sample_rate))
                num_copy_samples = sum(self.overlap) + (self.stride or 1) - self.pad_zeros_samples

            # prepare output frames, one buffer per frame
            self.preparedoutframes[pad] = TSFrame(
                buffers=preparedoutbufs,
                EOS=self.at_EOS,
            )

        return preparedbufs

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
                            shape=self.inbufs[pad][0].shape[:-1] + (0,),
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
                if self.audioadapters is not None:
                    out = self.__adapter(pad, out)
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
        return (self.inbufs[pad][-1].end - self.inbufs[pad][0].t0) > self.max_age

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

    pull = _TSTransSink.pull

    def __post_init__(self):
        if self.max_age is None:
            # FIXME is this what we want?
            self.max_age = 100 * Time.SECONDS
        TransformElement.__post_init__(self)
        _TSTransSink.__post_init__(self)

    def transform(self, pad):
        raise NotImplementedError


@dataclass
class TSSink(SinkElement, _TSTransSink):

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
        self.offset = {
            p: Offset.fromsec(self.t0 - Offset.offset_ref_t0) for p in self.source_pads
        }
        if self.num_samples is None:
            self.num_samples = Offset.stridesamples(self.rate)


def pad_func(data, pad_samples):
    npad = [(0, 0)] * data.ndim
    npad[-1] = (pad_samples, 0)
    return np.pad(data, npad, "constant")
