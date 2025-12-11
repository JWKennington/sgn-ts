from __future__ import annotations

import queue
import time as stime
from collections import deque
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, ClassVar, Generic, Optional, Sequence, Type, TypeVar, Union

import numpy
from sgn.base import (
    ElementLike,
    SinkElement,
    SinkPad,
    SourceElement,
    SourcePad,
    TransformElement,
)
from sgn.sources import SignalEOS
from sgn.subprocess import ParallelizeSourceElement, WorkerContext

from sgnts.base.array_ops import Array
from sgnts.base.audioadapter import AdapterConfig, Audioadapter
from sgnts.base.buffer import (
    EventFrame,
    SeriesBuffer,
    TimeSpanFrame,
    TSCollectFrame,
    TSFrame,
)
from sgnts.base.offset import Offset
from sgnts.base.slice_tools import TSSlice, TSSlices
from sgnts.base.time import Time

TSFrameLike = TypeVar("TSFrameLike", bound=TSFrame)


@dataclass
class TimeSeriesMixin(ElementLike, Generic[TSFrameLike]):
    """Mixin that adds time-series capabilities to any SGN element.

    This will produce aligned frames in preparedframes. If the adapter
    is not explicitly disabled, will trigger the audioadapter to queue
    data, and make padded or strided frames in preparedframes.

    This mixin provides:
    - Frame alignment across multiple input pads
    - Optional adapter processing (overlap/stride/gap handling)
    - Timeout detection and EOS handling

    Note:
        Subclasses can customize alignment behavior by setting class-level
        attributes:
          - static_unaligned_sink_pads: Declare which sink pads should not be
            aligned (e.g., EventFrame pads or auxiliary inputs).

    Args:
        max_age:
            int, the max age before timeout, in nanoseconds
        adapter_config:
            AdapterConfig, holds parameters used for audioadapter behavior
        unaligned:
            list[str], the list of unaligned sink pads.

    """

    # Class-level attributes for alignment configuration
    static_unaligned_sink_pads: ClassVar[list[str]] = []

    max_age: int = 100 * Time.SECONDS
    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)
    unaligned: Sequence[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize timeseries state."""
        super().__post_init__()

        # Determine which user-provided input pads require alignment
        unaligned_names = list(self.unaligned) + self.static_unaligned_sink_pads

        # Convert pad names to SinkPad objects and store in instance variable
        self.unaligned_sink_pads = [
            self.snks[name] for name in unaligned_names  # type: ignore[attr-defined]
        ]
        self.aligned_sink_pads = [
            p for p in self.sink_pads if p not in self.unaligned_sink_pads
        ]

        # Initialize metadata for exempt sink pads
        self.unaligned_data: dict[SinkPad, TimeSpanFrame | None] = {
            p: None for p in self.unaligned_sink_pads
        }

        # Initialize the alignment metadata for all sink pads that need to be aligned
        self._is_aligned = False
        self.inbufs = {p: Audioadapter() for p in self.aligned_sink_pads}
        self.preparedframes: dict[SinkPad, TSFrame | None] = {
            p: None for p in self.aligned_sink_pads
        }
        self.aligned_slices: dict[SinkPad, TSSlices | None] = {
            p: None for p in self.aligned_sink_pads
        }
        self.outframes: dict[SourcePad, TimeSpanFrame | None] = {
            p: None for p in self.source_pads
        }
        self.preparedoutoffsets = {"offset": 0, "noffset": 0}
        self.at_EOS = False
        self._last_ts: dict[SinkPad, float | None] = {
            p: None for p in self.aligned_sink_pads
        }
        self._last_offset: dict[SinkPad, int | None] = {
            p: None for p in self.aligned_sink_pads
        }
        self.metadata: dict[SinkPad, dict[Any, Any]] = {
            p: {} for p in self.aligned_sink_pads
        }

        # Initialize default frame types for inputs and outputs
        # These can be overridden by derived classes in configure()
        self.input_frame_types: dict[str, type[TimeSpanFrame]] = {
            name: TSFrame for name in self.sink_pad_names  # type: ignore[attr-defined]
        }
        # Initialize default output frame types (only for elements with source pads)
        # All pads default to TSFrame, elements can override in configure()
        self.output_frame_types: dict[str, type[TimeSpanFrame]] = {
            name: TSFrame for name in getattr(self, "source_pad_names", [])
        }

        # Configure adapter and element-specific attributes
        self.configure()

        self.is_adapter_enabled = self.adapter_config.is_enabled

        # Initialize adapter-specific state only if adapter is enabled
        self.audioadapters = None
        if self.is_adapter_enabled:
            self.overlap = self.adapter_config.overlap
            self.stride = self.adapter_config.stride
            self.pad_zeros_startup = self.adapter_config.pad_zeros_startup
            self.skip_gaps = self.adapter_config.skip_gaps
            self.offset_shift = self.adapter_config.offset_shift

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
        else:
            # No adapter, so no offset shift
            self.offset_shift = 0

        # Call validation hooks
        self.validate()

    def configure(self) -> None:
        """Configure element-specific settings."""
        pass

    def validate(self) -> None:
        """Validate element configuration."""
        pass

    def next_input(self) -> tuple[SinkPad, TSFrame]:
        """Convenience method - get single TSFrame input.

        Equivalent to next_ts_input(). For transforms that only work with TSFrames.

        Returns:
            TSFrame, single input frame
        """
        return self.next_ts_input()

    def next_output(self) -> tuple[SourcePad, TSCollectFrame]:
        """Convenience method - get single TSCollectFrame output.

        Equivalent to next_ts_output(). For transforms that only work with TSFrames.

        Note: Elements using this method must call `.close()` on the collector
        when done populating buffers.

        Returns:
            tuple[SourcePad, TSCollectFrame], pad and collector for output frame
        """
        return self.next_ts_output()

    def next_inputs(self) -> dict[SinkPad, TSFrame]:
        """Convenience method - get all TSFrame inputs.

        Equivalent to next_ts_inputs(). For transforms that only work with TSFrames.

        Returns:
            dict[SinkPad, TSFrame], dictionary of input frames
        """
        return self.next_ts_inputs()

    def next_outputs(self) -> dict[SourcePad, TSCollectFrame]:
        """Convenience method - get all TSCollectFrame outputs.

        Equivalent to next_ts_outputs(). For transforms that only work with TSFrames.

        Note: Elements using this method must call `.close()` on each collector
        when done populating buffers.

        Returns:
            dict[SourcePad, TSCollectFrame], dictionary of collectors for output frames
        """
        return self.next_ts_outputs()

    def next_ts_input(self) -> tuple[SinkPad, TSFrame]:
        """Get single TSFrame input.

        Returns:
            TSFrame, the single TSFrame from inputs

        Raises:
            AssertionError if there is not exactly one TSFrame input
        """
        all_ts = self.next_ts_inputs()
        assert (
            len(all_ts) == 1
        ), f"next_ts_input() requires exactly one TSFrame input, got {len(all_ts)}"
        return next(iter(all_ts.items()))

    def next_ts_inputs(self) -> dict[SinkPad, TSFrame]:
        """Get all TSFrame inputs based on input_frame_types configuration.

        Returns:
            dict[SinkPad, TSFrame], mapping of sink pads to TSFrame inputs
        """
        result: dict[SinkPad, TSFrame] = {}
        for pad in self.sink_pads:
            pad_name = self.rsnks[pad]  # type: ignore[attr-defined]
            if self.input_frame_types.get(pad_name, TSFrame) == TSFrame:
                if pad in self.aligned_sink_pads:
                    frame = self.preparedframes[pad]
                    assert frame is not None
                    result[pad] = frame
                else:
                    unaligned_frame = self.unaligned_data[pad]
                    assert isinstance(unaligned_frame, TSFrame), (
                        f"Expected TSFrame on unaligned pad {pad_name}, "
                        f"got {type(unaligned_frame).__name__}"
                    )
                    result[pad] = unaligned_frame
        return result

    def next_event_input(self) -> tuple[SinkPad, EventFrame]:
        """Get single EventFrame input.

        Returns:
            EventFrame, the single EventFrame from inputs

        Raises:
            AssertionError if there is not exactly one EventFrame input
        """
        all_events = self.next_event_inputs()
        assert len(all_events) == 1, (
            f"next_event_input() requires exactly one EventFrame input, "
            f"got {len(all_events)}"
        )
        return next(iter(all_events.items()))

    def next_event_inputs(self) -> dict[SinkPad, EventFrame]:
        """Get all EventFrame inputs based on input_frame_types configuration.

        Returns:
            dict[SinkPad, EventFrame], mapping of sink pads to EventFrame inputs
        """
        result: dict[SinkPad, EventFrame] = {}
        for pad in self.sink_pads:
            pad_name = self.rsnks[pad]  # type: ignore[attr-defined]
            if self.input_frame_types.get(pad_name) == EventFrame:
                if pad in self.unaligned_sink_pads:
                    frame = self.unaligned_data[pad]
                else:
                    frame = self.preparedframes[pad]
                assert isinstance(frame, EventFrame), (
                    f"Expected EventFrame on pad {pad_name}, "
                    f"got {type(frame).__name__}"
                )
                result[pad] = frame
        return result

    def next_ts_output(self) -> tuple[SourcePad, TSCollectFrame]:
        """Get single TSCollectFrame for output with offsets from preparedoutoffsets.

        Note: The caller must call `.close()` on the collector when done
        populating buffers.

        Returns:
            tuple[SourcePad, TSCollectFrame], pad and collector for the output
            frame

        Raises:
            AssertionError if there is not exactly one TS output pad
        """
        all_ts = self.next_ts_outputs()
        assert (
            len(all_ts) == 1
        ), f"next_ts_output() requires exactly one TS output pad, got {len(all_ts)}"
        return next(iter(all_ts.items()))

    def next_ts_outputs(self) -> dict[SourcePad, TSCollectFrame]:
        """Get all TSCollectFrames for output pads configured as TS outputs.

        Creates TSFrame instances with offset/noffset from preparedoutoffsets,
        then creates TSCollectFrame collectors for atomic buffer population.
        The parent TSFrames are automatically registered in self.outframes.

        Returns:
            dict[SourcePad, TSCollectFrame], mapping of source pads to collectors
        """
        offset = self.preparedoutoffsets["offset"]
        noffset = self.preparedoutoffsets["noffset"]
        at_EOS = any(
            frame.EOS for frame in self.preparedframes.values() if frame is not None
        )

        result: dict[SourcePad, TSCollectFrame] = {}
        for pad in self.source_pads:
            pad_name = self.rsrcs[pad]  # type: ignore[attr-defined]
            if self.output_frame_types.get(pad_name, TSFrame) == TSFrame:
                frame = TSFrame(offset=offset, noffset=noffset, EOS=at_EOS)
                collector = frame.fill()
                result[pad] = collector
                # Automatically register the parent frame in outframes
                self.outframes[pad] = frame
        return result

    def next_event_output(self) -> tuple[SourcePad, EventFrame]:
        """Get single EventFrame for output with offset/noffset from preparedoutoffsets.

        Returns:
            EventFrame, an empty event frame ready to be populated

        Raises:
            AssertionError if there is not exactly one event output pad
        """
        all_events = self.next_event_outputs()
        assert len(all_events) == 1, (
            f"next_event_output() requires exactly one event output pad, "
            f"got {len(all_events)}"
        )
        return next(iter(all_events.items()))

    def next_event_outputs(self) -> dict[SourcePad, EventFrame]:
        """Get all EventFrames for output pads configured as event outputs.

        Creates EventFrame instances with offset/noffset from preparedoutoffsets
        for all source pads configured to produce EventFrame. The frames are
        automatically registered in self.outframes for return.

        Returns:
            dict[SourcePad, EventFrame], mapping of source pads to empty EventFrames
        """
        offset = self.preparedoutoffsets["offset"]
        noffset = self.preparedoutoffsets["noffset"]
        at_EOS = any(
            frame.EOS for frame in self.preparedframes.values() if frame is not None
        )

        result: dict[SourcePad, EventFrame] = {}
        for pad in self.source_pads:
            pad_name = self.rsrcs[pad]  # type: ignore[attr-defined]
            if self.output_frame_types.get(pad_name) == EventFrame:
                frame = EventFrame(offset=offset, noffset=noffset, EOS=at_EOS)
                result[pad] = frame
                # Automatically register in outframes
                self.outframes[pad] = frame
        return result

    def pull(self, pad: SinkPad, frame: TimeSpanFrame) -> None:
        """Pull data and queue for alignment.

        Pull data from the input pads (source pads of upstream elements) and
        queue data to perform alignment once frames from all pads are pulled.

        Args:
            pad:
                SinkPad, The sink pad that is pulling the frame
            frame:
                TimeSpanFrame, The frame that is pulled to sink pad
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

    def _compute_aligned_offset(self, current_offset: int, align_to: int) -> int:
        """Compute aligned offset based on alignment boundary.

        Args:
            current_offset: Current offset in offsets
            align_to: Alignment boundary in offsets

        Returns:
            Aligned offset
        """
        return ((current_offset + align_to - 1) // align_to) * align_to

    def __adapter(self, pad: SinkPad, frame: list[SeriesBuffer]) -> list[SeriesBuffer]:
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
            the adapter config

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
        assert self.audioadapters is not None
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

        # Determine if we're using alignment mode
        use_alignment = (
            self.is_adapter_enabled and self.adapter_config.align_to is not None
        )

        # Apply alignment if configured
        if use_alignment:
            assert self.adapter_config.align_to is not None
            outoffset = self._compute_aligned_offset(
                outoffset,
                self.adapter_config.align_to,
            )

        preparedbufs = []

        # Check if we have enough data
        if use_alignment:
            # For aligned mode, check if we have data up to aligned_offset + stride
            aligned_end = outoffset + self.stride
            has_enough_data = a.end_offset >= aligned_end
        else:
            # Original check based on size
            has_enough_data = a.size >= min_samples

        if not has_enough_data:
            # not enough samples to produce output yet
            # make a heartbeat buffer
            shape = buf0.shape[:-1] + (0,)
            preparedbufs.append(
                SeriesBuffer(
                    offset=offset, sample_rate=sample_rate, data=None, shape=shape
                )
            )
            # prepare output frames, one buffer per frame
            self.preparedoutoffsets = {
                "offset": outoffset + self.offset_shift,
                "noffset": 0,
            }
        else:
            # We have enough samples, retrieve data
            if use_alignment:
                # Retrieve data at exact aligned offset
                aligned_end = outoffset + self.stride
                stride_samples_actual = Offset.tosamples(self.stride, sample_rate)

                # Check for gaps in the aligned segment
                segment_has_gap, segment_has_nongap = a.segment_gaps_info(
                    (outoffset, aligned_end)
                )

                if not segment_has_nongap or (self.skip_gaps and segment_has_gap):
                    # Gap in aligned segment
                    data = None
                else:
                    # Retrieve data at the aligned offset using offset-based slicing
                    data = a.copy_samples_by_offset_segment(
                        (outoffset, aligned_end), pad_start=False
                    )

                # Create output buffer at aligned offset (no padding needed if aligned)
                shape = buf0.shape[:-1] + (
                    stride_samples_actual if data is not None else 0,
                )
                pbuf = SeriesBuffer(
                    offset=outoffset,  # Use aligned offset
                    sample_rate=sample_rate,
                    data=data,
                    shape=shape,
                )
                preparedbufs.append(pbuf)

                # Flush data up to the END of the aligned segment (not the start)
                # This ensures next iteration starts after this segment
                a.flush_samples_by_end_offset(aligned_end)

                # Output offset metadata
                outnoffset = self.stride
                self.preparedoutoffsets = {
                    "offset": outoffset + self.offset_shift,
                    "noffset": outnoffset,
                }

                # No padding offset adjustment needed for aligned mode
                self.pad_zeros_offset = 0

            else:
                # copy all of the samples in the audioadapter
                if self.stride == 0:
                    # provide all the data
                    num_copy_samples = a.size
                else:
                    num_copy_samples = min_samples

                segment_has_gap, segment_has_nongap = a.segment_gaps_info(
                    (
                        a.offset,
                        a.offset + Offset.fromsamples(num_copy_samples, a.sample_rate),
                    )
                )

                # Check if we should preserve buffer boundaries (align_buffers mode)
                if self.adapter_config.align_buffers:
                    # Return sliced buffers without merging, preserving gaps
                    end_offset = a.offset + Offset.fromsamples(
                        num_copy_samples, a.sample_rate
                    )
                    preparedbufs = list(
                        a.get_sliced_buffers((a.offset, end_offset), pad_start=False)
                    )
                    outnoffset = end_offset - a.offset
                    self.preparedoutoffsets = {
                        "offset": a.offset + self.offset_shift,
                        "noffset": outnoffset,
                    }

                    # flush out samples from head of audioadapter
                    num_flush_samples = num_copy_samples - sum(overlap_samples)
                    if num_flush_samples > 0:
                        a.flush_samples(num_flush_samples)

                elif not segment_has_nongap or (self.skip_gaps and segment_has_gap):
                    # produce a gap buffer if
                    # 1. the whole segment is a gap or
                    # 2. there are gaps in the segment and we are skipping gaps
                    data = None

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
                    self.preparedoutoffsets = {
                        "offset": outoffset + self.offset_shift,
                        "noffset": outnoffset,
                    }

                else:
                    # copy out samples from head of audioadapter
                    data = a.copy_samples(num_copy_samples)
                    if self.pad_zeros_offset > 0:
                        # pad zeros in front of buffer
                        data = self.adapter_config.backend.pad(
                            data, (pad_zeros_samples, 0)
                        )

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
                    self.preparedoutoffsets = {
                        "offset": outoffset + self.offset_shift,
                        "noffset": outnoffset,
                    }

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
            for sink_pad in self.aligned_sink_pads:
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
            # Set preparedoutoffsets for heartbeat (zero-length output)
            self.preparedoutoffsets = {
                "offset": self.earliest + self.offset_shift,
                "noffset": 0,
            }
        # Else pack all the buffers
        else:
            min_latest = self.min_latest
            earliest = self.earliest

            rates = set(
                self.inbufs[sink_pad].sample_rate for sink_pad in self.aligned_sink_pads
            )
            off = min_latest - earliest
            for rate in rates:
                factor = Offset.MAX_RATE // rate
                if off % factor:
                    off = off // factor * factor
                    min_latest = earliest + off

            for sink_pad in self.aligned_sink_pads:
                out = list(
                    self.inbufs[sink_pad].get_sliced_buffers(
                        (earliest, min_latest), pad_start=True
                    )
                )
                if min_latest > self.inbufs[sink_pad].offset:
                    self.inbufs[sink_pad].flush_samples_by_end_offset(min_latest)
                assert (
                    len(out) > 0
                ), "No buffers returned from get_sliced_buffers for aligned processing"

                # Apply adapter processing only if adapter is enabled
                if self.is_adapter_enabled:
                    out = self.__adapter(sink_pad, out)

                self.preparedframes[sink_pad] = TSFrame(
                    EOS=self.at_EOS,
                    buffers=out,
                    metadata=self.metadata[sink_pad],
                )

            # Apply buffer alignment if requested
            if self.adapter_config.align_buffers:
                computed_slices = self._compute_aligned_slices()
                for pad, slices in computed_slices.items():
                    self.aligned_slices[pad] = slices

                for pad in self.aligned_sink_pads:
                    aligned_slice = self.aligned_slices[pad]
                    assert aligned_slice is not None
                    frame = self.preparedframes[pad]
                    assert frame is not None
                    # Only align if there are slices (skip if all gaps)
                    if aligned_slice.slices:
                        self.preparedframes[pad] = frame.align(aligned_slice)

            # Set preparedoutoffsets for non-adapter case
            if not self.is_adapter_enabled:
                self.preparedoutoffsets = {
                    "offset": earliest + self.offset_shift,
                    "noffset": min_latest - earliest,
                }

    def _compute_aligned_slices(self) -> dict[SinkPad, TSSlices]:
        """Compute aligned slices for all pads based on minimum sampling rate.

        Extracts slices from prepared frames, finds the minimum
        sampling rate across all pads, and aligns all slices to that rate.
        """
        # Find minimum sampling rate across all aligned pads
        nongap_slices: dict[SinkPad, TSSlices] = {}
        sample_rates = []
        for pad in self.aligned_sink_pads:
            frame = self.preparedframes[pad]
            assert frame is not None
            sample_rates.append(frame.sample_rate)
        min_rate = min(sample_rates)

        # For each pad, extract slices corresponding to non-gaps and align to
        # minimum rate
        for pad in self.aligned_sink_pads:
            frame = self.preparedframes[pad]
            assert frame is not None
            # Extract non-gap buffer slices from the prepared frame
            buffer_slices = [buf.slice for buf in frame.buffers if not buf.is_gap]

            if not buffer_slices:
                # No non-gap buffers, no slices to align
                nongap_slices[pad] = TSSlices([])
                continue

            # align slices to minimum rate
            slices = TSSlices(buffer_slices)
            aligned = slices.align_to_rate(min_rate)
            nongap_slices[pad] = aligned

        all_nongap_slices = TSSlices.intersection_of_multiple(
            list(nongap_slices.values())
        )
        start = self.preparedoutoffsets["offset"]
        end = start + self.preparedoutoffsets["noffset"]
        boundaries = sorted(
            set(
                [
                    start,
                    *list(chain(*[[s.start, s.stop] for s in all_nongap_slices])),
                    end,
                ]
            )
        )
        slice_boundaries = TSSlices(
            [
                TSSlice(b_start, b_stop)
                for b_start, b_stop in zip(boundaries[:-1], boundaries[1:])
            ]
        )

        return {pad: slice_boundaries for pad in self.aligned_sink_pads}

    def _align(self) -> None:
        """Align the buffers in self.inbufs."""

        def slice_from_pad(inbufs):
            if len(inbufs) > 0:
                return TSSlice(inbufs.offset, inbufs.end_offset)
            else:
                return TSSlice(-1, -1)

        def can_align():
            return TSSlices(
                [slice_from_pad(self.inbufs[p]) for p in self.inbufs]
            ).intersection()

        if not self._is_aligned and can_align():
            self._is_aligned = True

    def timeout(self, pad: SinkPad) -> bool:
        """Whether pad has timed-out due to oldest buffer exceeding max age.

        Args:
            pad:
                SinkPad, the sink pad to check for timeout

        Returns:
            True if the pad has timed out

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

    def earliest_by_pad(self, pad: SinkPad) -> int:
        """The earliest offset among the queued up buffers in this pad.

        Args:
            pad:
                SinkPad, the requested sink pad

        Returns:
            int, the earliest offset in the pad's buffer queue

        """
        return self.inbufs[pad].offset if self.inbufs[pad] else -1

    @property
    def latest(self) -> int:
        """The latest offset among all the buffers from all the pads."""
        return max(self.latest_by_pad(n) for n in self.inbufs)

    @property
    def earliest(self) -> int:
        """The earliest offset among all the buffers from all the pads."""
        return min(self.earliest_by_pad(n) for n in self.inbufs)

    @property
    def min_latest(self) -> int:
        """The earliest offset among each pad's latest offset."""
        return min(self.latest_by_pad(n) for n in self.inbufs)

    @property
    def is_aligned(self) -> bool:
        """Check if input frames are currently aligned across all pads.

        Returns:
            True if frames from all input pads have overlapping time ranges
            and can be processed together. False if waiting for more data.
        """
        return self._is_aligned


@dataclass
class TSTransform(TimeSeriesMixin[TSFrame], TransformElement[TimeSpanFrame]):
    """A time-series transform element."""

    def internal(self) -> None:
        """Process frames by calling child class implementation.

        If the child class defines a process() method, it will be called with
        input and output frame dictionaries. Otherwise, child classes should
        override internal() directly.
        """
        super().internal()

        # Check if the element defines a process() method
        if hasattr(self, "process"):
            # Collect all input frames (both TSFrame and EventFrame)
            inframes: dict[SinkPad, TimeSpanFrame] = {}
            inframes.update(self.next_ts_inputs())
            inframes.update(self.next_event_inputs())

            # Collect all output collectors/frames (TSCollectFrame or EventFrame)
            ts_collectors = self.next_ts_outputs()
            outframes: dict[SourcePad, TimeSpanFrame | TSCollectFrame] = {}
            outframes.update(ts_collectors)
            outframes.update(self.next_event_outputs())

            # Call the process method
            self.process(inframes, outframes)  # type: ignore[attr-defined]

            # Close all TS collectors to commit buffers to parent frames
            for collector in ts_collectors.values():
                collector.close()

    def new(self, pad: SourcePad) -> TimeSpanFrame:
        """Return the output frame for the given pad.

        It should take the source pad as an argument and return a new
        TSFrame or EventFrame.

        Args:
            pad:
                SourcePad, The source pad that is producing the transformed frame

        Returns:
            TSFrame or EventFrame, The transformed frame

        """
        frame = self.outframes.get(pad)
        assert frame is not None
        return frame


@dataclass
class TSSink(TimeSeriesMixin[TSFrame], SinkElement[TimeSpanFrame]):
    """A time-series sink element."""

    def internal(self) -> None:
        """Process frames by calling child class implementation.

        If the child class defines a process() method, it will be called with
        input frame dictionaries. Otherwise, child classes should override
        internal() directly.
        """
        super().internal()

        # Check if the element defines a process() method
        if hasattr(self, "process"):
            # Collect all input frames (both TSFrame and EventFrame)
            inframes: dict[SinkPad, TimeSpanFrame] = {}
            inframes.update(self.next_ts_inputs())
            inframes.update(self.next_event_inputs())

            # Call the process method
            self.process(inframes)  # type: ignore[attr-defined]


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
        assert (
            len(self._next_frame_dict) > 0
        ), "_next_frame_dict is empty - no frames available for processing"
        return min(f.t0 for f in self._next_frame_dict.values())

    @property
    def current_end(self) -> float:
        """Return the largest end time of the current prepared frame, which
        should be the same for all pads when called in the internal method but maybe
        different otherwise"""
        assert (
            len(self._next_frame_dict) > 0
        ), "_next_frame_dict is empty - no frames available for processing"
        return max(f.end for f in self._next_frame_dict.values())

    @property
    def current_end_offset(self) -> float:
        """Return the largest end offset of the current prepared frame, which
        should be the same for all pads when called in the internal method but maybe
        different otherwise"""
        assert (
            len(self._next_frame_dict) > 0
        ), "_next_frame_dict is empty - no frames available for processing"
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
        assert (
            len(frame) == 1
        ), "Expected exactly one buffer in frame for single-pad element"

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
            assert latest_offset >= frame.offset, (
                f"Latest offset {latest_offset} cannot be before "
                f"frame offset {frame.offset}"
            )
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
        assert self.t0 is not None
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
        assert (
            pad not in self._new_buffer_dict
        ), f"Pad {pad.name} already exists in _new_buffer_dict - duplicate pad entry"

        self._new_buffer_dict[pad] = {
            "sample_rate": rate,
            "shape": sample_shape + (self.num_samples(rate),),
        }
        self._next_frame_dict[pad] = TSFrame.from_buffer_kwargs(
            offset=self.start_offset, data=None, **self._new_buffer_dict[pad]
        )


@dataclass
class TSResourceSource(ParallelizeSourceElement, _TSSource):
    """Source class that is entirely data driven by an external resource.

    This class uses ParallelizeSourceElement to run data generation in a separate
    worker thread. Subclasses must override the worker_process method
    to define how data is generated in the worker.

    The worker communicates with the main thread via queues provided by
    ParallelizeSourceElement. Data should be sent as (pad, buffer) tuples to
    the output queue using context.output_queue.put((pad, buf)).

    Important: Since the worker starts when entering the Parallelize context
    (before setup() is called), all parameters needed by the worker must be
    added as instance attributes and will be automatically passed to worker_process
    via the parameter extraction mechanism.

    Subclasses should:
    1. Override worker_process to implement data generation
    2. Use context.output_queue.put((pad, buf)) to send data
    3. Check context.should_stop() to know when to exit

    Exception handling follows SGN's improved Parallelize pattern: exceptions in the
    worker are caught by the framework, printed to stdout, and cause the
    worker to terminate. The main thread detects abnormal termination via
    the internal() method.

    Args:
        start_time: Optional[int] = None
            Start time in GPS seconds. Used by subclasses to determine
            when data generation should begin.
        duration: Optional[int] = None
            Duration in nanoseconds. If None, defaults to maximum int64 value.
        in_queue_timeout: int = 60
            Timeout in seconds when waiting for data from the worker.
            Used by get_data_from_queue() in the main thread.

    """

    start_time: Optional[int] = None
    duration: Optional[int] = None
    in_queue_timeout: int = 60
    _use_threading_override: bool = (
        True  # Always use threading for I/O bound data sources
    )

    def __post_init__(self):
        self.queue_maxsize = 100

        self.__is_setup = False
        self.__end = None
        if self.duration is None:
            self.duration = numpy.iinfo(numpy.int64).max
            self.__end = numpy.iinfo(numpy.int64).max
        if self.start_time is not None and self.duration is not None:
            self.__end = self.duration + self.start_time

        # Initialize parent classes - IMPORTANT: Order matters!
        # _TSSource must come first because it creates self.source_pads and self.srcs
        # ParallelizeSourceElement must come after because it extracts self.srcs
        # for worker
        _TSSource.__post_init__(self)
        ParallelizeSourceElement.__post_init__(self)

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
        props = self.first_buffer_properties[pad]
        assert props is not None
        return props["sample_shape"]

    def sample_rate(self, pad):
        """The integer sample rate that a buffer should carry"""
        props = self.first_buffer_properties[pad]
        assert props is not None
        return props["sample_rate"]

    @property
    def latest_offset(self):
        """Since the worker is responsible for producing a queue of
        buffers, the latest offset can be derived from those"""
        latest = numpy.iinfo(numpy.int64).min
        for properties in self.latest_buffer_properties.values():
            if properties is not None:
                latest = max(latest, properties["end_offset"])
        return latest

    @property
    def start_offset(self):
        offsets = [
            b["offset"] for b in self.first_buffer_properties.values() if b is not None
        ]
        return min(offsets)

    @property
    def end_offset(self):
        if self.end_time is None:
            return float("inf")
        return Offset.fromsec(self.end_time - Offset.offset_ref_t0 / Time.SECONDS)

    @property
    def t0(self):
        """The starting time of the resource in seconds"""
        return Offset.tosec(self.start_offset)

    def setup(self) -> None:
        """Initialize the TSResourceSource data structures."""
        if not self.__is_setup:
            self.buffer_queue: dict[SourcePad, deque[SeriesBuffer]] = {
                p: deque() for p in self.rsrcs
            }
            self.latest_buffer_properties = {p: None for p in self.rsrcs}
            self.first_buffer_properties = {p: None for p in self.rsrcs}
            self.__is_setup = True

    @property
    def queued_duration(self):
        durations = [d[-1].end - d[0].t0 for d in self.buffer_queue.values() if d]
        if durations:
            return max(durations)
        else:
            return 0.0

    def _get_data_from_worker(
        self, timeout: int = 60
    ) -> dict[SourcePad, list[SeriesBuffer]]:
        """Get data from the worker via ParallelizeSourceElement's queue."""
        data_by_pad: dict[SourcePad, list[SeriesBuffer]] = {p: [] for p in self.rsrcs}
        start_time = stime.time()

        # Collect data from worker until we have data for all pads or timeout
        while stime.time() - start_time < timeout:
            # Check if worker has terminated abnormally before trying to get data
            self.check_worker_terminated()
            try:
                # Get data from worker queue (provided by ParallelizeSourceElement)
                item = self.out_queue.get(timeout=0.1)

                pad, buf = item
                data_by_pad[pad].append(buf)

                # Check if we have at least one buffer for each pad
                if all(data_by_pad[p] for p in self.rsrcs):
                    break

            except queue.Empty:
                # No data available yet, continue waiting
                continue
        else:
            self.check_worker_terminated()
            # Timeout reached
            raise ValueError(f"Could not read from resource after {timeout} seconds")

        return data_by_pad

    def get_data_from_queue(self):
        """Retrieve data from the worker with a timeout."""
        # Get data from worker
        data_by_pad = self._get_data_from_worker(timeout=self.in_queue_timeout)

        # Add data to output queues
        for pad, buffers in data_by_pad.items():
            self.buffer_queue[pad].extend(buffers)

            if buffers:  # If we got any buffers for this pad
                buffer_queue = self.buffer_queue[pad]
                self.latest_buffer_properties[pad] = buffer_queue[-1].properties
                if self.first_buffer_properties[pad] is None:
                    self.first_buffer_properties[pad] = buffer_queue[0].properties

        # We should have a t0 now
        if self.__end is None and self.duration is not None:
            self.__end = self.t0 + self.duration

    def set_data(self, out_frame, pad):
        """This method will set data on out_frame based on the contents of the
        internal queue"""

        # Check if we are at EOS, if so, set the flag
        if out_frame.EOS:
            self.at_eos = True

        # If we have been given a zero length frame, just return it. That means
        # we didn't have data at the time the frame was prepared and we should
        # just go with it.
        if out_frame.offset == out_frame.end_offset:
            return out_frame

        # Otherwise create a TSFrame from all the buffers that we have queued up
        in_frame = TSFrame(buffers=list(self.buffer_queue[pad]))

        # make sure nothing is fishy
        assert out_frame.end_offset <= in_frame.end_offset, (
            f"Output frame end_offset {out_frame.end_offset} extends beyond "
            f"input frame end_offset {in_frame.end_offset}"
        )

        # intersect the TSSource provided output frame with the in_frame
        before, intersection, after = out_frame.intersect(in_frame)

        # Clear the queue
        self.buffer_queue[pad].clear()

        # and repopulate it with only stuff that is newer than what we just sent.
        if after is not None:
            self.buffer_queue[pad].extend(after.buffers)

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
        assert (
            pad not in self._new_buffer_dict
        ), f"Pad {pad.name} already exists in _new_buffer_dict - duplicate pad entry"

        self._new_buffer_dict[pad] = {
            "sample_rate": self.sample_rate(pad),
            "shape": self.sample_shape(pad)
            + (self.num_samples(self.sample_rate(pad)),),
        }
        self._next_frame_dict[pad] = TSFrame.from_buffer_kwargs(
            offset=self.start_offset, data=None, **self._new_buffer_dict[pad]
        )

    @staticmethod
    def worker_process(context: WorkerContext, *args: Any, **kwargs: Any) -> None:
        """Override this method in subclasses to implement data generation.

        This method runs in a separate worker (process or thread) and should:
        1. Generate data from the external resource
        2. Send (pad, buffer) tuples via context.output_queue.put((pad, buf))
        3. Check context.should_stop() to know when to exit

        Args:
            context: WorkerContext with access to queues and events
            *args: Automatically extracted instance attributes
            **kwargs: Automatically extracted instance attributes with defaults
        """
        raise NotImplementedError("Subclasses must implement worker_process method")

    def internal(self):
        """Since internal() is guaranteed to be called prior to producing any
        data on a source pad, all setup is done here. First the resource itself is
        setup and the first data is pulled from the resource. Subsequent calls to
        internal only gets data from the resource if there is not enough data queued up
        to produce a result"""

        # Check if worker has terminated abnormally
        super().internal()

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


def make_ts_element(sgn_element_class: Type) -> Type:
    """Factory to create TS-enabled versions of SGN elements.

    This provides a simple way to add TS capabilities to existing SGN elements
    so they can connect to TS pipelines. Uses a basic AdapterConfig() that works
    for most general-purpose applications.

    Args:
        sgn_element_class: SGN element class to enhance

    Returns:
        New class that combines SGN element with TS capabilities
    """

    @dataclass
    class TSEnabledElement(TimeSeriesMixin, sgn_element_class):
        """Dynamically created TS-enabled element."""

        # Use basic adapter config that works for general TS connectivity
        adapter_config: AdapterConfig = field(default_factory=AdapterConfig)

        def new(self, pad):
            """Default implementation of new() for factory-created elements."""
            return self.outframes.get(pad)

    # Set a meaningful name for the new class
    TSEnabledElement.__name__ = f"TS{sgn_element_class.__name__}"
    TSEnabledElement.__qualname__ = f"TS{sgn_element_class.__qualname__}"

    return TSEnabledElement
