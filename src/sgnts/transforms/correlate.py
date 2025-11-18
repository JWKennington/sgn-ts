from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy
import scipy
from sgn.base import SinkPad, SourcePad

from sgnts.base import (
    AdapterConfig,
    Array,
    Event,
    EventBuffer,
    EventFrame,
    Offset,
    SeriesBuffer,
    TSFrame,
    TSTransform,
)
from sgnts.base.slice_tools import TIME_MAX


@dataclass
class Correlate(TSTransform):
    """Correlates input data with filters

    Args:
        filters:
            Array, the filter to correlate over
        latency:
            int, the latency of the filter in samples
    """

    sample_rate: int = -1
    filters: Optional[Array] = None
    latency: int = 0

    def __post_init__(self):
        # FIXME: read sample_rate from data
        assert (
            self.filters is not None
        ), "Filters must be provided during initialization"
        assert self.sample_rate != -1, "Sample rate must be specified (not -1)"
        self.shape = self.filters.shape
        if self.adapter_config is None:
            self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (
            Offset.fromsamples(self.shape[-1] - 1, self.sample_rate),
            0,
        )
        self.adapter_config.pad_zeros_startup = False
        super().__post_init__()
        assert len(self.aligned_sink_pads) == 1 and len(self.source_pads) == 1, (
            f"Correlate requires exactly one aligned sink pad and one "
            f"source pad, got {len(self.aligned_sink_pads)} aligned sink "
            f"pads and {len(self.source_pads)} source pads"
        )

    def corr(self, data: Array) -> Array:
        """Correlate an array of data with an array of filters.

        Args:
            data:
                Array, the data to correlate with the filters

        Returns:
            Array, the result of the correlation
        """
        assert self.filters is not None, "Filters must not be None during correlation"
        if len(self.filters.shape) == 1:
            return scipy.signal.correlate(data, self.filters, mode="valid")

        # Skip the reshape for now
        os = []
        shape = self.shape
        self.filters = self.filters.reshape(-1, shape[-1])
        for j in range(self.shape[0]):
            os.append(scipy.signal.correlate(data, self.filters[j], mode="valid"))
        return numpy.vstack(os).reshape(shape[:-1] + (-1,))

    def new(self, pad: SourcePad) -> TSFrame:
        outbufs = []
        outoffset = self.preparedoutoffsets
        frames = self.preparedframes[self.sink_pads[0]]
        for buf in frames:
            assert buf.sample_rate == self.sample_rate, (
                f"Buffer sample rate {buf.sample_rate} doesn't match "
                f"correlator sample rate {self.sample_rate}"
            )
            if buf.is_gap:
                data = None
            else:
                # FIXME: Are there multi-channel correlation in numpy or scipy?
                # FIXME: consider multi-dimensional filters
                data = self.corr(buf.data)
            outbufs.append(
                SeriesBuffer(
                    offset=outoffset["offset"]
                    - Offset.fromsamples(self.latency, self.sample_rate),
                    sample_rate=buf.sample_rate,
                    data=data,
                    shape=(
                        self.shape[:-1]
                        + (Offset.tosamples(outoffset["noffset"], buf.sample_rate),)
                        if data is None
                        else data.shape
                    ),
                )
            )
        return TSFrame(buffers=outbufs, EOS=frames.EOS)


@dataclass
class AdaptiveCorrelate(Correlate):
    """Adaptive Correlate filter performs a correlation over a time-dependent set of
    filters. When the filters are updated, the correlation is performed over both the
    existing filters and the new filters, then combined using a window function.

    Notes:
        Update frequency:
            Only 2 sets of filters are supported at this time. This is equivalent
            to requiring that filters can only be updated once per stride. Attempting
            to pass more than one update per stride will raise an error.
        Update duration:
            The filter update is performed across the entire stride. There is not
            presently support for more time-domain control of start/stop times for
            the blending of filters.

    Args:
        filter_sink_name:
            str, the name of the sink pad to pull data from
        filters:
            Array, the filter to correlate over

    Raises:
        ValueError:
            Raises a value error if more than one filter update is passed per stride
    """

    filter_sink_name: str = "filters"

    def __post_init__(self) -> None:
        """Setup the adaptive FIR filter"""
        # Setup empty deque for storing filters
        self.filter_deque: Deque[EventFrame] = deque()

        # Argument validation
        self._validate_filters_pad()

        # Call the parent's post init, this will setup all the appropriate pads
        super().__post_init__()

        # Set the initial filters
        event = Event(offset=0, data=self.filters)
        buf = EventBuffer(
            offset=0,
            noffset=TIME_MAX,
            data=[event],
        )
        frame = EventFrame(data=[buf])
        self.filter_deque.append(frame)

    def _validate_filters_pad(self):
        """Validate the filter sink pad before initializing the filter"""
        # Make sure the filter sink name is not already in use
        assert self.filter_sink_name not in self.sink_pad_names, (
            f"Filter sink name '{self.filter_sink_name}' already exists "
            f"in sink_pad_names: {self.sink_pad_names}"
        )

        # Check that if unaligned pads are specified, that the filter sink name MUST
        # be one of them, if not included then add
        if self.unaligned is not None:
            if self.filter_sink_name not in self.unaligned:
                self.unaligned = list(self.unaligned) + [self.filter_sink_name]
        else:
            self.unaligned = [self.filter_sink_name]

        # Add the filter sink name to the sink pad names
        self.sink_pad_names = list(self.sink_pad_names) + [self.filter_sink_name]

    @staticmethod
    def _extract_filter(item: EventBuffer | EventFrame) -> Array:
        """Extract the filter from an event buffer or frame."""
        if len(item.events) > 1:
            msg = "found more than one event in {item}, " "cannot extract filter."
            raise ValueError(msg)
        event = item.events[0]
        return event.data

    @property
    def filters_cur(self) -> EventFrame:
        """Get the current filters"""
        return self.filter_deque[0]

    @property
    def filters_new(self) -> Optional[EventFrame]:
        """Get the new filters"""
        if len(self.filter_deque) > 1:
            return self.filter_deque[1]

        return None

    @property
    def is_adapting(self) -> bool:
        """Check if the adaptive filter is adapting"""
        return self.filters_new is not None

    def can_adapt(self, frame: TSFrame) -> bool:
        """Check if the buffer can be adapted"""
        if not self.is_adapting:
            return False

        if frame.is_gap:
            return False

        # The below check is unnecessary except for Mypy
        assert (
            self.filters_new is not None
        ), "filters_new should not be None when can_adapt returns True"
        # Check that the frame overlaps the new filter slice
        new_slice = self.filters_new.slice
        frame_slice = frame.slice
        overlap = new_slice & frame_slice
        return overlap.isfinite()

    def pull(self, pad: SinkPad, frame: TSFrame) -> None:
        # Pull the data from the sink pad
        super().pull(pad, frame)

        # If the pad is the special filter sink pad, then update filter
        # metadata values
        if pad.name == self.snks[self.filter_sink_name].name:
            # Assume frame is an EventFrame containing a single Event
            event_frame = self.unaligned_data[pad]
            assert isinstance(event_frame, EventFrame)
            new_filter = self._extract_filter(event_frame)

            # If the buffer is null, then short circuit
            if new_filter is None:
                return

            # Redundant check, but more generalizable?
            if len(self.filter_deque) > 1:
                raise ValueError("Only one filter update per stride is supported")

            # Check that the new filters have the same shape as the existing filters
            if (
                self.filters_cur is not None
                and self._extract_filter(self.filters_cur).shape != new_filter.shape
            ):
                raise ValueError(
                    "New filters must have the same shape as existing filters"
                )

            # Set the new filters
            self.filter_deque.append(event_frame)

    def new(self, pad: SourcePad) -> TSFrame:
        # Get a aligned buffer to see if overlaps with new filters
        frame = self.preparedframes[self.sink_pads[0]]

        if self.can_adapt(frame):
            # Call the parent's new method for each set of filters
            assert self.filters_cur is not None
            self.filters = self._extract_filter(self.filters_cur)
            res_cur = super().new(pad)  # type: ignore  # not recognizing self

            # Change the state of filters
            assert self.filters_new is not None
            self.filters = self._extract_filter(self.filters_new)
            res_new = super().new(pad)  # type: ignore  # not recognizing self

            # Combine data with window functions

            # remove the new filters to indicate adaptation is complete
            self.filter_deque.popleft()

            # Compute window functions. Window functions
            # will be piecewise functions for the corresponding
            # intersection of the filter slice and data slice
            # where the window function is 0.0 before the intersection
            # and 1.0 after the intersection, and cos^2 in between
            N = res_cur[0].data.shape[-1]
            win_new = (scipy.signal.windows.cosine(2 * N, sym=True) ** 2)[:N]
            win_cur = 1.0 - win_new

            data = win_cur * res_cur[0].data + win_new * res_new[0].data

        else:
            res_new = super().new(pad)
            if res_new.is_gap:
                data = None
            else:
                data = res_new.buffers[0].data

        # Return the new frame
        assert data is None or isinstance(data, numpy.ndarray)  # assert for typing
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=res_new[0].offset,
                    data=data,
                    sample_rate=res_new.sample_rate,
                    shape=res_new.shape if data is None else data.shape,
                )
            ],
            EOS=res_new.EOS,
        )
        return frame
