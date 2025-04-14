from __future__ import annotations

import queue
import threading
import time as stime
from collections import deque
from dataclasses import dataclass

# from typing import Optional, Union, Callable, Protocol, Sequence
from typing import Optional, Sequence, Union

import numpy
from sgn.base import SinkElement, SinkPad, SourceElement, SourcePad, TransformElement
from sgn.sources import SignalEOS

from sgnts.base.array_ops import Array, ArrayBackend, NumpyBackend
from sgnts.base.audioadapter import Audioadapter
from sgnts.base.buffer import SeriesBuffer, TSFrame
from sgnts.base.offset import Offset
from sgnts.base.slice_tools import TSSlice, TSSlices
from sgnts.base.time import Time


@dataclass
class AdapterConfig:
    """Config to hold parameters used for the audioadapter in _TSTransSink.

    Args:
        overlap:
            tuple[int, int], the overlap before and after the data segment to process,
            in offsets
        stride:
            int, the stride to produce, in offsets
        pad_zeros_startup:
            bool, when overlap is provided, whether to pad zeros in front of the
            first buffer, or wait until there is enough data.
        skip_gaps:
            bool, produce a whole gap buffer if there are any gaps in the copied data
            segment
        backend:
            type[ArrayBackend], the ArrayBackend wrapper
    """

    overlap: tuple[int, int] = (0, 0)
    stride: int = 0
    pad_zeros_startup: bool = False
    skip_gaps: bool = False
    backend: type[ArrayBackend] = NumpyBackend

    def valid_buffer(self, buf, data: Optional[Union[int, Array]] = 0):
        """
        Return a new buffer corresponding to the non overlapping part of a
        buffer "buf" as defined by this classes overlap properties As a special case,
        if the buffer is shape zero (a heartbeat buffer) a new heartbeat buffer is
        returned with the offsets shifted by overlap[0].
        Otherwise, in order for the buffer to be valid it must be what is expected
        based on the adapter's overlap and stride etc.
        """

        if buf.shape == (0,):
            new_slice = TSSlice(
                buf.slice[0] + self.overlap[0], buf.slice[0] + self.overlap[0]
            )
            return buf.new(new_slice, data=None)
        else:
            expected_shape = (
                Offset.tosamples(self.overlap[0], buf.sample_rate)
                + Offset.tosamples(self.overlap[1], buf.sample_rate)
                + Offset.sample_stride(buf.sample_rate),
            )
            assert buf.shape == expected_shape
            new_slice = TSSlice(
                buf.slice[0] + self.overlap[0], buf.slice[1] - self.overlap[1]
            )
            return buf.new(new_slice, data)


#
# NOTE FIXME. I have tried multiple times to get the _TSTransSink mixin class
# to play well with mypy, but each time just dug a deeper hole. The issue is
# that the mixin class refers to sink_pads which of course don't exist unless
# it is mixed in.  Below you can see remnants of a couple of attemps including
# using a Protocol class (mypy recommendation) and making it a subclass of
# ElementLike. I couldn't get either to work but encourage others to try.
#

# class HasValueProtocol(Protocol):
#    @property
#    def sink_pads(self) -> Sequence: ...
#
#    @sink_pads.setter
#    def sink_pads(self, value: Sequence) -> Sequence: ...


@dataclass
class _TSTransSink:  # (HasValueProtocol):
    # class _TSTransSink(ElementLike)
    """Base class for TSTransforms and TSSinks.

    This will produce aligned frames in preparedframes. If
    adapter_config is provided, will trigger the audioadapter to queue
    data, and make padded or strided frames in preparedframes.

    Args:
        max_age:
            int, the max age before timeout, in nanoseconds
        adapter_config:
            AdapterConfig, holds parameters used for audioadapter behavior

    """

    max_age: int = 100 * Time.SECONDS
    adapter_config: Optional[AdapterConfig] = None
    unaligned: Optional[Sequence[str]] = None

    def __post_init__(self):
        """Initialize the _TSTransSink class."""
        # First, determine which input pads actually require alignment
        if self.unaligned is None:
            self.unaligned_sink_pads = []
        else:
            self.unaligned_sink_pads = [self.snks[name] for name in self.unaligned]
        self.aligned_sink_pads = [
            p for p in self.sink_pads if p not in self.unaligned_sink_pads
        ]

        # Initialize metadata for exempt sink pads
        self.unaligned_data = {p: None for p in self.unaligned_sink_pads}

        # Initialize the alignment metadata for all sink pads that need to be aligned
        self._is_aligned = False
        self.inbufs = {p: Audioadapter() for p in self.aligned_sink_pads}
        self.preparedframes = {p: None for p in self.aligned_sink_pads}
        self.at_EOS = False
        self._last_ts = {p: None for p in self.aligned_sink_pads}
        self._last_offset = {p: None for p in self.aligned_sink_pads}
        self.metadata = {p: None for p in self.aligned_sink_pads}
        self.audioadapters = None
        if self.adapter_config is not None:
            self.overlap = self.adapter_config.overlap
            self.stride = self.adapter_config.stride
            self.pad_zeros_startup = self.adapter_config.pad_zeros_startup
            self.skip_gaps = self.adapter_config.skip_gaps

            # we need audioadapters
            self.audioadapters = {
                p: Audioadapter(backend=self.adapter_config.backend)
                for p in self.aligned_sink_pads
            }
            self.pad_zeros_offset = 0
            if self.pad_zeros_startup is True:
                # at startup, pad zeros in front of the first buffer to
                # serve as history
                self.pad_zeros_offset = self.overlap[0]
            self.preparedoutoffsets = {p: None for p in self.aligned_sink_pads}

    def pull(self, pad: SinkPad, frame: TSFrame) -> None:
        """Pull data from the input pads (source pads of upstream elements) and queue
        data to perform alignment once frames from all pads are pulled.

        Args:
            pad:
                SinkPad, The sink pad that is pulling the frame
            frame:
                Frame, The frame that is pulled to sink pad
        """

        self.at_EOS |= frame.EOS

        # Handle case of a pad that is exempt from alignment
        if pad in self.unaligned_sink_pads:
            # Store most recent data for exempt pads
            self.unaligned_data[pad] = frame
            # TODO maybe add bespoke timeout handling here
            return

        # Handle case of a pad that requires alignment
        # extend and check the buffers
        for buf in frame:
            self.inbufs[pad].push(buf)
        self.metadata[pad] = frame.metadata
        if self.timeout(pad):
            raise ValueError("pad %s has timed out" % pad.name)

    def __adapter(self, pad: SinkPad, frame: TSFrame) -> list[SeriesBuffer]:
        """Use the audioadapter to handle streaming scenarios.

        This will pad with overlap before and after the target output
        data, and produce fixed-stride frames.

        The self.preparedframes are padded with the requested overlap padding. This
        method also produces a self.preparedoutoffsets, that infers the metadata
        information for the output buffer, with the data initialized as None.
        Downstream transforms can directly use the frames from self.preparedframes for
        computation, and then use the offset and noffset information in
        self.preparedoutoffsets to construct the output frame.

        If stride is not provided, the audioadapter will push out as many samples as it
        can. If stride is provided, the audioadapter will wait until there are enough
        samples to produce prepared frames.

        Args:
            pad:
                SinkPad, the sink pad on which to prepare adapted frames
            frame:
                TSFrame, the aligned frame

        Returns:
            list[SeriesBuffers], a list of SeriesBuffers that are adapted according to
            the adapter_config

        Examples:
            upsampling:
                kernel length = 17
                need to pad 8 samples before and after
                overlap_samples = (8, 8)
                stride_samples = 16
                                                for output
                preparedframes:     ________................________
                                                stride
                                    pad         samples=16  pad
                                    samples=8               samples=8


            correlation:
                filter length = 16
                need to pad filter_length - 1 samples
                overlap_samples = (15, 0)
                stride_samples = 8
                                                    for output
                preparedframes:     ----------------........
                                                    stride_samples=8
                                    pad
                                    samples=15

        """
        a = self.audioadapters[pad]
        buf0 = frame[0]
        sample_rate = buf0.sample_rate
        overlap_samples = tuple(Offset.tosamples(o, sample_rate) for o in self.overlap)
        stride_samples = Offset.tosamples(self.stride, sample_rate)
        pad_zeros_samples = Offset.tosamples(self.pad_zeros_offset, sample_rate)

        # push all buffers in the frame into the audioadapter
        for buf in frame:
            a.push(buf)

        # Check whether we have enough samples to produce a frame
        min_samples = sum(overlap_samples) + (stride_samples or 1) - pad_zeros_samples

        # figure out the offset for preparedframes and preparedoutoffsets
        offset = a.offset - self.pad_zeros_offset
        outoffset = offset + self.overlap[0]
        preparedbufs = []
        if a.size < min_samples:
            # not enough samples to produce output yet
            # make a heartbeat buffer
            shape = buf0.shape[:-1] + (0,)
            preparedbufs.append(
                SeriesBuffer(
                    offset=offset, sample_rate=sample_rate, data=None, shape=shape
                )
            )
            # prepare output frames, one buffer per frame
            self.preparedoutoffsets[pad] = [{"offset": outoffset, "noffset": 0}]

        else:
            # We have enough samples, find out how many samples to copy
            # out of the audioadapter
            # copy all of the samples in the audioadapter
            if self.stride == 0:
                # provide all the data
                num_copy_samples = a.size
            else:
                num_copy_samples = min_samples

            outoffsets = []

            segment_has_gap, segment_has_nongap = a.segment_gaps_info(
                (
                    a.offset,
                    a.offset + Offset.fromsamples(num_copy_samples, a.sample_rate),
                )
            )

            if not segment_has_nongap or (self.skip_gaps and segment_has_gap):
                # produce a gap buffer if
                # 1. the whole segment is a gap or
                # 2. there are gaps in the segment and we are skipping gaps
                data = None
            else:
                # copy out samples from head of audioadapter
                data = a.copy_samples(num_copy_samples)
                if self.pad_zeros_offset > 0 and self.adapter_config is not None:
                    # pad zeros in front of buffer
                    data = self.adapter_config.backend.pad(data, (pad_zeros_samples, 0))

            # flush out samples from head of audioadapter
            num_flush_samples = num_copy_samples - sum(overlap_samples)
            if num_flush_samples > 0:
                a.flush_samples(num_flush_samples)

            shape = buf0.shape[:-1] + (num_copy_samples + pad_zeros_samples,)

            # update next zeros padding
            self.pad_zeros_offset = -min(
                0, Offset.fromsamples(num_flush_samples, sample_rate)
            )
            pbuf = SeriesBuffer(
                offset=offset, sample_rate=sample_rate, data=data, shape=shape
            )
            preparedbufs.append(pbuf)
            outnoffset = pbuf.noffset - sum(self.overlap)
            outoffsets.append({"offset": outoffset, "noffset": outnoffset})

            self.preparedoutoffsets[pad] = outoffsets

        return preparedbufs

    def internal(self) -> None:
        """Align buffers from all the sink pads.

        If AdapterConfig is provided, perform the requested
        overlap/stride streaming of frames.

        """
        # align if possible
        self._align()

        # put in heartbeat buffer if not aligned
        if not self._is_aligned:
            # I tried to fix this properly see the notes above the definition
            # of _TSTransSink
            for sink_pad in self.aligned_sink_pads:  # type: ignore
                self.preparedframes[sink_pad] = TSFrame(
                    EOS=self.at_EOS,
                    buffers=[
                        SeriesBuffer(
                            offset=self.earliest,
                            sample_rate=self.inbufs[sink_pad].sample_rate,
                            data=None,
                            shape=self.inbufs[sink_pad].buffers[0].shape[:-1] + (0,),
                        ),
                    ],
                    metadata=self.metadata[sink_pad],
                )
        # Else pack all the buffers
        else:
            min_latest = self.min_latest
            earliest = self.earliest
            # I tried to fix this properly see the notes above the definition
            # of _TSTransSink
            for sink_pad in self.aligned_sink_pads:  # type: ignore
                out = self.inbufs[sink_pad].get_sliced_buffers(
                    (earliest, min_latest), pad_start=True
                )
                if min_latest > self.inbufs[sink_pad].offset:
                    self.inbufs[sink_pad].flush_samples_by_end_offset(min_latest)
                assert len(out) > 0
                if self.adapter_config is not None:
                    out = self.__adapter(sink_pad, out)
                self.preparedframes[sink_pad] = TSFrame(
                    EOS=self.at_EOS,
                    buffers=out,
                    metadata=self.metadata[sink_pad],
                )

    def _align(self) -> None:
        """Align the buffers in self.inbufs."""

        def slice_from_pad(inbufs):
            if len(inbufs) > 0:
                return TSSlice(inbufs.offset, inbufs.end_offset)
            else:
                return TSSlice(-1, -1)

        def __can_align(self=self):
            return TSSlices(
                [slice_from_pad(self.inbufs[p]) for p in self.inbufs]
            ).intersection()

        if not self._is_aligned and __can_align():
            self._is_aligned = True

    def timeout(self, pad: SinkPad) -> bool:
        """Whether pad has timed-out due to oldest buffer exceeding max age.

        Args:
            pad:
                SinkPad, the sink pad to check for timeout

        Returns:
            bool, whether pad has timed-out

        """
        return self.inbufs[pad].end_offset - self.inbufs[pad].offset > Offset.fromns(
            self.max_age
        )

    def latest_by_pad(self, pad: SinkPad) -> int:
        """The latest offset among the queued up buffers in this pad.

        Args:
            pad:
                SinkPad, the requested sink pad

        Returns:
            int, the latest offset in the pad's buffer queue

        """
        return self.inbufs[pad].end_offset if self.inbufs[pad] else -1

    def earliest_by_pad(self, pad) -> int:
        """The earliest offset among the queued up buffers in this pad.

        Args:
            pad:
                SinkPad, the requested sink pad

        Returns:
            int, the earliest offset in the pad's buffer queue

        """
        return self.inbufs[pad].offset if self.inbufs[pad] else -1

    @property
    def latest(self):
        """The latest offset among all the buffers from all the pads."""
        return max(self.latest_by_pad(n) for n in self.inbufs)

    @property
    def earliest(self):
        """The earliest offset among all the buffers from all the pads."""
        return min(self.earliest_by_pad(n) for n in self.inbufs)

    @property
    def min_latest(self):
        """The earliest offset among each pad's latest offset."""
        return min(self.latest_by_pad(n) for n in self.inbufs)


@dataclass
class TSTransform(TransformElement, _TSTransSink):
    """A time-series transform element."""

    # FIXME mypy complains that this takes a TSFrame instead of a Frame.  Not
    # sure what the right fix is.
    # FIXME, I also cannot get type hints to work
    # pull: Callable[[SourcePad, TSFrame], None] = _TSTransSink.pull  # type: ignore
    pull = _TSTransSink.pull  # type: ignore

    def __post_init__(self):
        TransformElement.__post_init__(self)
        _TSTransSink.__post_init__(self)

    def internal(self):
        _TSTransSink.internal(self)

    def new(self, pad: SourcePad) -> TSFrame:
        """The transform function must be provided by the subclass.

        It should take the source pad as an argument and return a new
        TSFrame.

        Args:
            pad:
                SourcePad, The source pad that is producing the transformed frame

        Returns:
            TSFrame, The transformed frame

        """
        raise NotImplementedError


@dataclass
class TSSink(SinkElement, _TSTransSink):
    """A time-series sink element."""

    # FIXME mypy complains that this takes a TSFrame instead of a Frame.  Not
    # sure what the right fix is.
    # FIXME, I also cannot get type hints to work
    # pull: Callable[[SourcePad, TSFrame], None] = _TSTransSink.pull  # type: ignore
    pull = _TSTransSink.pull  # type: ignore

    def __post_init__(self):
        SinkElement.__post_init__(self)
        _TSTransSink.__post_init__(self)

    def internal(self):
        _TSTransSink.internal(self)


@dataclass
class _TSSource(SourceElement, SignalEOS):
    """A time-series source base class. This should not be used directly"""

    def __post_init__(self):
        super().__post_init__()
        self._new_buffer_dict = {}
        self._next_frame_dict = {}

    @property
    def end_offset(self):
        "This should be the precise last offset"
        raise NotImplementedError

    @property
    def start_offset(self):
        "This should be the precise start offset"
        raise NotImplementedError

    def num_samples(self, rate: int) -> int:
        """The number of samples in the sample stride at the requested rate.

        Args:
            rate:
                int, the sample rate

        Returns:
            int, the number of samples

        """
        return Offset.sample_stride(rate)

    @property
    def current_t0(self) -> float:
        """Return the smallest t0 of the current prepared frame, which should
        be the same for all pads when called in the internal method, but maybe
        different otherwise"""
        assert len(self._next_frame_dict) > 0
        return min(f.t0 for f in self._next_frame_dict.values())

    @property
    def current_end(self) -> float:
        """Return the largest end time of the current prepared frame, which
        should be the same for all pads when called in the internal method but maybe
        different otherwise"""
        assert len(self._next_frame_dict) > 0
        return max(f.end for f in self._next_frame_dict.values())

    @property
    def current_end_offset(self) -> float:
        """Return the largest end offset of the current prepared frame, which
        should be the same for all pads when called in the internal method but maybe
        different otherwise"""
        assert len(self._next_frame_dict) > 0
        return max(f.end_offset for f in self._next_frame_dict.values())

    def prepare_frame(
        self,
        pad: SourcePad,
        latest_offset: Optional[int] = None,
        data: Optional[Union[int, Array]] = None,
        EOS: Optional[bool] = None,
        metadata: Optional[dict] = None,
    ) -> TSFrame:
        """Prepare the next TSFrame that the source pad will produce.

        The offset will be advanced by the stride in
        Offset.SAMPLE_STRIDE_AT_MAX_RATE.

        Args:
            pad:
                SourcePad, the source pad to produce the TSFrame
            latest_offset:
                int | None. If given, a buffer will be zero length unless
                latest_offset is >= the expected end offset
            data:
                Optional[int, Array], the data in the buffers
            EOS:
                Optioinal[bool], whether the TSFrame is at EOS
            metadata:
                Optional[dict], the metadata in the TSFrame

        Returns:
            TSFrame, the TSFrame prepared on the source pad

        """
        frame = self._next_frame_dict[pad]
        assert len(frame) == 1

        EOS = (
            (frame[0].end_offset >= self.end_offset or self.signaled_eos())
            if EOS is None
            else (
                EOS or (frame[0].end_offset >= self.end_offset) or self.signaled_eos()
            )
        )

        # See if we need to pass a heartbeat frame
        # If so, return the heartbeat and move on
        if latest_offset is not None:
            assert latest_offset >= frame.offset
            if latest_offset < frame.end_offset:
                return frame.heartbeat(EOS)

        # Otherwise we can make progress with what we have
        frame[0].set_data(data)

        if frame.end_offset > self.end_offset:
            # slice the buffer if the last buffer is not a full stride
            frame.set_buffers(
                [frame[0].sub_buffer(TSSlice(frame[0].offset, self.end_offset))]
            )

        frame.EOS = EOS
        frame.metadata = {} if metadata is None else metadata
        self._next_frame_dict[pad] = next(frame)
        return frame


@dataclass
class TSSource(_TSSource):
    """A time-series source that generates data in fixed-size buffers where the
       user can specify the start time and end time. If you want a data driven
       source consider using TSResourceSource.

    Args:
        t0:
            float, start time of first buffer, in seconds
        end:
            float, end time of the last buffer, in seconds
        duration:
            float, alternative to end option, specify the duration of
            time to be covered in seconds. Cannot be given if end is given.
    """

    t0: float | None = None
    end: float | None = None
    duration: float | None = None

    def __post_init__(self):
        super().__post_init__()

        if self.t0 is None:
            raise ValueError("You must specifiy a t0")

        if self.end is not None and self.duration is not None:
            raise ValueError("may specify either end or duration, not both")

        if self.duration is not None:
            self.end = self.t0 + self.duration

        if self.end is not None:
            assert self.end > self.t0, "end is before t0"

    @property
    def end_offset(self):
        if self.end is None:
            return float("inf")
        return Offset.fromsec(self.end - Offset.offset_ref_t0 / Time.SECONDS)

    @property
    def start_offset(self):
        return Offset.fromsec(self.t0 - Offset.offset_ref_t0 / Time.SECONDS)

    def set_pad_buffer_params(
        self,
        pad: SourcePad,
        sample_shape: tuple[int, ...],
        rate: int,
    ) -> None:
        """Set variables on the pad that are needed to construct SeriesBuffers.

        These should remain constant throughout the duration of the
        pipeline so this method may only be called once.

        Args:
            pad:
                SourcePad, the pad to setup buffers on
            sample_shape:
                tuple[int, ...], the shape of a single sample of the
                data, or put another way, the shape of the data except
                for the last (time) dimension,
                i.e. sample_shape=data.shape[:-1]
            rate:
                int, the sample rate of the data the pad will produce

        """
        # Make sure this has only been called once per pad
        assert pad not in self._new_buffer_dict

        self._new_buffer_dict[pad] = {
            "sample_rate": rate,
            "shape": sample_shape + (self.num_samples(rate),),
        }
        self._next_frame_dict[pad] = TSFrame.from_buffer_kwargs(
            offset=self.start_offset, data=None, **self._new_buffer_dict[pad]
        )


@dataclass
class TSResourceSource(_TSSource):
    """Source class that is entirely data driven by an external resource.

    The resource will gather data in a separate thread.  The user
    must implement the get_data() method and probably doesn't
    need to implement any other methods.  The get_data()
    method must fill a queue with a single buffer of the next data block.
    It must block until the queue is not full.

    Assume we have a service with a client called "server" that has a
    "stream" method that takes a list of channels
    Assume it returns objects called blocks
    with additional metadata e.g.,  "time_ns", "sample_rate", and "data".
    An implementation of get_data() might look like this:

    def get_data(self):
        for stream in server.stream(tuple(self.srcs)):
            for channel, block in stream.items():
                pad = self.srcs[channel]
                buf = SeriesBuffer(
                    offset=Offset.fromns(block.time_ns),
                    data=block.data,
                    sample_rate=block.channel.sample_rate,
                )
                yield pad, buf

    Args:
        start_time: Optional[int] = None
            - If None, implies should start at now and is a real-time
              server
        duration: Optional[int] = None
            - If None, go on forever
        in_queue_timeout: int = 60
            - How long to wait for a buffer from the resource before
              timing out with a fatal error. This needs to be longer
              than the duration of buffers coming from a real-time
              server or it will hang.

    """

    start_time: Optional[int] = None
    duration: Optional[int] = None
    in_queue_timeout: int = 60

    def __post_init__(self):
        super().__post_init__()
        self.__in_queue_length = 100
        self.__is_setup = False
        self.thread = None
        self.__end = None
        if self.duration is None:
            self.duration = numpy.iinfo(numpy.int64).max
            self.__end = numpy.iinfo(numpy.int64).max
        if self.start_time is not None and self.duration is not None:
            self.__end = self.duration + self.start_time

    @property
    def end_time(self):
        """The ending time of the resource"""
        return self.__end

    @property
    def is_setup(self):
        return self.__is_setup

    def sample_shape(self, pad):
        """The channels per sample that a buffer should produce as a tuple
        (since it can be a tensor). For single channels just return ()"""
        return self.first_buffer_properties[pad]["sample_shape"]

    def sample_rate(self, pad):
        """The integer sample rate that a buffer should carry"""
        return self.first_buffer_properties[pad]["sample_rate"]

    @property
    def latest_offset(self):
        """Since the thread is responsible for producing a queue of
        buffers, the latest offest can be derived from those"""
        latest = numpy.iinfo(numpy.int64).min
        for properties in self.latest_buffer_properties.values():
            if properties is not None:
                latest = max(latest, properties["end_offset"])
        return latest

    @property
    def start_offset(self):
        return min(b["offset"] for b in self.first_buffer_properties.values())

    @property
    def end_offset(self):
        if self.end_time is None:
            return float("inf")
        return Offset.fromsec(self.end_time - Offset.offset_ref_t0 / Time.SECONDS)

    @property
    def t0(self):
        """The starting time of the resource in seconds"""
        return Offset.tosec(self.start_offset)

    def setup(self):
        """Initialize the RealTimeDataSource class."""
        if not self.__is_setup:
            self.stop_event = threading.Event()
            self.exception_event = threading.Event()
            self.in_queue = {p: queue.Queue(self.__in_queue_length) for p in self.rsrcs}
            self.out_queue = {p: deque() for p in self.rsrcs}
            self.latest_buffer_properties = {p: None for p in self.rsrcs}
            self.first_buffer_properties = {p: None for p in self.rsrcs}
            self.__is_setup = True
            self.start()

    @property
    def queued_duration(self):
        durations = [d[-1].end - d[0].t0 for d in self.out_queue.values() if d]
        if durations:
            return max(durations)
        else:
            return 0.0

    def get_data(self):
        """User implemented function that pairs with thread_get_data()"""
        raise NotImplementedError

    def thread_get_data(self):
        """This will run in a separate thread. It must fill the self.in_queue
        with SeriesBuffers.  The buffers should be contiguous or else errors will happen
        later"""
        try:
            for pad, buf in self.get_data():
                self.in_queue[pad].put(buf)
                if self.stop_event.is_set():
                    break
        except Exception:
            self.exception_event.set()
            raise

    def __exit__(self):
        self.stop()

    def start(self):
        assert self.__is_setup
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.thread_get_data)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()
            self.thread = None
            self.__is_setup = False

    def _flush_queue(self, q, timeout=0):
        assert timeout >= 1, "timeouts can not be less than one second."
        # how long do we block before checking for resource exceptions
        # FIXME: is should we wait for more or less time?
        sleep = 0.01  # seconds
        # track start time so we know when timeout is reached
        start_time = stime.time()
        # now wait for the elements in the queue, checking for
        # execeptions from the resource while we wait
        while stime.time() - start_time < timeout:
            # check to see if any exceptions have been raise by the
            # resource, and stop the thread and exit is so.
            if self.exception_event.is_set():
                self.stop()
                raise RuntimeError("exception raised in resource thread, aborting.")
            if q.empty():
                stime.sleep(sleep)
            else:
                break
        else:
            self.stop()
            raise ValueError(f"could not read from resource after {timeout} seconds")
        # drain the queue
        out = []
        while not q.empty():
            out.append(q.get(block=False))
        return out

    def get_data_from_queue(self):
        """Retrieve data from the queue with a timeout."""
        for pad in self.out_queue:
            self.out_queue[pad] += self._flush_queue(
                self.in_queue[pad],
                timeout=self.in_queue_timeout,
            )
            self.latest_buffer_properties[pad] = self.out_queue[pad][-1].properties
            if self.first_buffer_properties[pad] is None:
                self.first_buffer_properties[pad] = self.out_queue[pad][0].properties
            # We should have a t0 now
            if self.__end is None and self.duration is not None:
                self.__end = self.t0 + self.duration

    def set_data(self, out_frame, pad):
        """This method will set data on out_frame based on the contents of the
        internal queue"""

        # Check if were are at EOS, if so, stop the thread
        if out_frame.EOS:
            self.stop()

        # If we have been given a zero length frame, just return it. That means
        # we didn't have data at the time the frame was prepared and we should
        # just go with it.
        if out_frame.offset == out_frame.end_offset:
            return out_frame

        # Otherwise create a TSFrame from all the buffers that we have queued up
        in_frame = TSFrame(buffers=self.out_queue[pad])

        # make sure nothing is fishy
        assert out_frame.end_offset <= in_frame.end_offset

        # intersect the TSSource provided output frame with the in_frame
        before, intersection, after = out_frame.intersect(in_frame)

        # Clear the queue
        self.out_queue[pad].clear()

        # and repopulate it with only stuff that is newer than what we just sent.
        if after is not None:
            self.out_queue[pad].extend(after.buffers)

        # It is possible that the out_frame is before the data we have in the
        # queue, if so the intersection will be None. Thats okay, we can just
        # pass along that gap buffer.
        if intersection is None:
            return out_frame

        # make sure to update EOS
        intersection.EOS = out_frame.EOS
        return intersection

    def __set_pad_buffer_params(
        self,
        pad: SourcePad,
    ) -> None:
        # Make sure this has only been called once per pad
        # assert self.resource is not None
        assert pad not in self._new_buffer_dict

        self._new_buffer_dict[pad] = {
            "sample_rate": self.sample_rate(pad),
            "shape": self.sample_shape(pad)
            + (self.num_samples(self.sample_rate(pad)),),
        }
        self._next_frame_dict[pad] = TSFrame.from_buffer_kwargs(
            offset=self.start_offset, data=None, **self._new_buffer_dict[pad]
        )

    def internal(self):
        """Since internal() is gauranteed to be called prior to producing any
        data on a source pad, all setup is done here. First the resource itself is
        setup and the first data is pulled from the resource.  Subsequent calls to
        internal only gets data from the resource if there is not enough data queued up
        to produce a result"""

        # First setup the resource and pull the first data
        if not self.is_setup:
            self.setup()
            self.get_data_from_queue()
            # setup pads if they are not setup.
            # This must happen after the first get data
            for pad in self.rsrcs:
                if pad not in self._new_buffer_dict:
                    self.__set_pad_buffer_params(pad)
        else:
            # check if we need to get more data
            if self.latest_offset < self.current_end_offset:
                self.get_data_from_queue()

    def new(self, pad):
        frame = self.prepare_frame(pad, latest_offset=self.latest_offset)
        frame = self.set_data(frame, pad)
        return frame
