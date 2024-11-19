from __future__ import annotations

from dataclasses import dataclass
from math import isinf
from typing import Optional, Union

from sgn.base import (
    InternalPad,
    SinkElement,
    SinkPad,
    SourceElement,
    SourcePad,
    TransformElement,
)

from sgnts.base.array_ops import Array, ArrayBackend, NumpyBackend
from sgnts.base.audioadapter import Audioadapter
from sgnts.base.buffer import EventBuffer, EventFrame, SeriesBuffer, TSFrame
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


@dataclass
class _TSTransSink:
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

    def __post_init__(self):

        self._is_aligned = False
        self.inbufs = {p: Audioadapter() for p in self.sink_pads}
        self.preparedframes = {p: None for p in self.sink_pads}
        self.at_EOS = False
        self._last_ts = {p: None for p in self.sink_pads}
        self._last_offset = {p: None for p in self.sink_pads}
        self.metadata = {p: None for p in self.sink_pads}
        self.audioadapters = None
        if self.adapter_config is not None:
            self.overlap = self.adapter_config.overlap
            self.stride = self.adapter_config.stride
            self.pad_zeros_startup = self.adapter_config.pad_zeros_startup
            self.skip_gaps = self.adapter_config.skip_gaps

            # we need audioadapters
            self.audioadapters = {
                p: Audioadapter(backend=self.adapter_config.backend)
                for p in self.sink_pads
            }
            self.pad_zeros_offset = 0
            if self.pad_zeros_startup is True:
                # at startup, pad zeros in front of the first buffer to
                # serve as history
                self.pad_zeros_offset = self.overlap[0]
            self.preparedoutoffsets = {p: None for p in self.sink_pads}

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
                # 1. the whole audioadapter is a gap or
                # 2. the whole segment is a gap or
                # 3. there are gaps in the segment and we are skipping gaps
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

    def internal(self, pad: InternalPad) -> None:
        """Align buffers from all the sink pads.

        If AdapterConfig is provided, perform the requested
        overlap/stride streaming of frames.

        """
        # align if possible
        self._align()

        # put in heartbeat buffer if not aligned
        if not self._is_aligned:
            for sink_pad in self.sink_pads:
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
            for sink_pad in self.sink_pads:
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

    pull = _TSTransSink.pull

    def __post_init__(self):
        TransformElement.__post_init__(self)
        _TSTransSink.__post_init__(self)

    def internal(self, pad: InternalPad):
        _TSTransSink.internal(self, pad)

    def transform(self, pad: SourcePad) -> TSFrame:
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

    pull = _TSTransSink.pull

    def __post_init__(self):
        SinkElement.__post_init__(self)
        _TSTransSink.__post_init__(self)

    def internal(self, pad: InternalPad):
        _TSTransSink.internal(self, pad)


@dataclass
class TSSource(SourceElement):
    """A time-series source that generates data in fixed-size buffers.

    Args:
        t0:
            float, start time of first buffer, in seconds
        end:
            float, end time of the last buffer, in seconds

    """

    t0: float = 0
    end: float = float("+inf")

    def __post_init__(self):
        super().__post_init__()
        # FIXME should we be more careful about this?
        # FIXME should this not be different by pad?
        self.offset = {
            p: Offset.fromsec(self.t0 - Offset.offset_ref_t0 / Time.SECONDS)
            for p in self.source_pads
        }
        # FIXME should this be different by pad?
        if not isinf(self.end):
            self.end_offset = Offset.fromsec(
                self.end - Offset.offset_ref_t0 / Time.SECONDS
            )
        else:
            self.end_offset = float("+inf")
        self.__new_buffer_dict = {}

    def num_samples(self, rate: int) -> int:
        """The number of samples in the sample stride at the requested rate.

        Args:
            rate:
                int, the sample rate

        Returns:
            int, the number of samples

        """
        return Offset.sample_stride(rate)

    def set_pad_buffer_params(
        self,
        pad: SourcePad,
        sample_shape: tuple[int, ...],
        rate: int,
    ) -> None:
        """Set variables on the pad that are needed to construct SeriesBuffers.

        These should remain constant throughout the duration of the
        pipeline.

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
        self.__new_buffer_dict[pad] = {
            "sample_rate": rate,
            "shape": sample_shape + (self.num_samples(rate),),
        }

    def prepare_frame(
        self,
        pad: SourcePad,
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
            data:
                Optional[int, Array], the data in the buffers
            EOS:
                Optioinal[bool], whether the TSFrame is at EOS
            metadata:
                Optional[dict], the metadata in the TSFrame

        Returns:
            TSFrame, the TSFrame prepared on the source pad

        """
        buf = SeriesBuffer(
            offset=self.offset[pad], data=data, **self.__new_buffer_dict[pad]
        )
        if buf.end_offset > self.end_offset:
            # slice the buffer if the last buffer is not a full stride
            buf = buf.sub_buffer(TSSlice(buf.offset, self.end_offset))

        if EOS is None:
            EOS = buf.end_offset == self.end_offset
        if metadata is None:
            metadata = {}

        self.offset[pad] += Offset.fromsamples(buf.samples, buf.sample_rate)

        return TSFrame(
            buffers=[buf],
            EOS=EOS,
            metadata=metadata,
        )
