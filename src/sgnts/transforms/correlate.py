from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy
import scipy
from sgn import validator
from sgn.base import SinkPad

from sgnts.base import (
    Array,
    Event,
    EventBuffer,
    EventFrame,
    Offset,
    TimeSpanFrame,
    TSCollectFrame,
    TSFrame,
    TSTransform,
)
from sgnts.base.slice_tools import TIME_MAX


@dataclass(kw_only=True)
class Correlate(TSTransform):
    """Correlates input data with filters

    Args:
        sample_rate:
            int, the sample rate of the input data
        filters:
            Optional[Array], the filter to correlate over. Default is None,
            which will be treated as the null initial condition (produce gap
            buffers until filters are provided in the case of AdaptiveCorrelate, or
            just produce gap buffers in the case of Correlate). This is done to prevent
            requiring non DRY initial condition/behavior specification for Correlate
            vs AdaptiveCorrelate, since the latter has a sink pad for filters which
            will be given a default value by the first frame.
        latency:
            int, the latency of the filter in samples
    """

    sample_rate: int
    filters: Optional[Array] = None
    latency: int = 0

    def configure(self) -> None:
        # If filters are not set yet (e.g., AdaptiveCorrelate startup), use a
        # placeholder shape and zero overlap; downstream corr() won't be called
        # until filters are provided.
        if self.filters is None:
            self.shape = (1,)
            overlap_samples = 0
        else:
            self.shape = self.filters.shape
            overlap_samples = max(0, self.shape[-1] - 1)

        # apply latency offset shift: negative shift moves output backward in time
        self.adapter_config.alignment(
            overlap=(Offset.fromsamples(overlap_samples, self.sample_rate), 0),
            shift=-Offset.fromsamples(self.latency, self.sample_rate),
        )
        self.adapter_config.on_startup(pad_zeros=False)

        self.sink_pad = self.sink_pads[0]
        self.source_pad = self.source_pads[0]

    @validator.one_to_one
    def validate(self) -> None:
        pass

    def corr(self, data: Array) -> Array:
        """Correlate an array of data with an array of filters.

        Args:
            data:
                Array, the data to correlate with the filters

        Returns:
            Array, the result of the correlation
        """
        if self.filters is None:
            raise ValueError("Cannot correlate without filters")

        if len(self.filters.shape) == 1:
            return scipy.signal.correlate(data, self.filters, mode="valid")

        # Skip the reshape for now
        os = []
        shape = self.shape
        self.filters = self.filters.reshape(-1, shape[-1])
        for j in range(self.shape[0]):
            os.append(scipy.signal.correlate(data, self.filters[j], mode="valid"))
        return numpy.vstack(os).reshape(shape[:-1] + (-1,))

    def _transform(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Helper to correlate input with current filters and populate output.

        Args:
            input_frame: Input frame to process
            output_frame: Output frame to populate
        """
        for buf in input_frame:
            assert buf.sample_rate == self.sample_rate, (
                f"Buffer sample rate {buf.sample_rate} doesn't match "
                f"correlator sample rate {self.sample_rate}"
            )
            if buf.is_gap or self.filters is None:
                data = None
                output_samples = Offset.tosamples(output_frame.noffset, buf.sample_rate)
                shape = self.shape[:-1] + (output_samples,)
                is_gap = True
            else:
                # FIXME: Are there multi-channel correlation in numpy or scipy?
                # FIXME: consider multi-dimensional filters
                data = self.corr(buf.data)
                shape = data.shape
                is_gap = False

            buf = buf.copy(
                offset=output_frame.offset,
                data=data,
                shape=shape,
                is_gap=is_gap,
            )
            output_frame.append(buf)

    def internal(self) -> None:
        super().internal()

        _, output_collector = self.next_output()
        _, input_frame = self.next_input()
        self._transform(input_frame, output_collector)
        output_collector.close()


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
        Startup behavior (no explicit initial conditions):
            This element no longer accepts explicit initial PSD or initial filters.
            On startup, it will emit gap buffers (no data) until a filter bank is
            received on the dedicated filters sink pad (filter_sink_name). The first
            received filter set becomes the current filters. Subsequent updates are
            blended over a stride as described below.

    Args:
        filter_sink_name:
            str, the name of the sink pad to pull data from

    Raises:
        ValueError:
            Raises a value error if more than one filter update is passed per stride
    """

    filter_sink_name: str = "filters"
    ignore_rapid_updates: bool = False
    verbose: bool = False

    @property
    def static_sink_pads(self) -> list[str]:  # type: ignore[override]
        """Add the filter sink pad as an static sink pad."""
        return [self.filter_sink_name]

    @property
    def static_unaligned_sink_pads(self) -> list[str]:  # type: ignore[override]
        """Declare that the filter sink pad is unaligned."""
        return [self.filter_sink_name]

    def configure(self) -> None:
        super().configure()
        self.filter_pad = self.snks[self.filter_sink_name]

        # Set the input frame type for the filter pad to EventFrame
        self.input_frame_types[self.filter_sink_name] = EventFrame

        # Setup empty deque for storing filters
        self.filter_deque: Deque[EventFrame] = deque()

        # If an initial filter bank is provided (legacy/compat), accept it.
        if self.filters is not None:
            assert not isinstance(self.filters, EventFrame), "Filters should be Array"
            # Create a dummy event with the filter data and put it in an EventFrame
            event_frame = EventFrame(
                data=[
                    EventBuffer(
                        offset=0,
                        noffset=TIME_MAX,
                        data=[self.filters],
                    )
                ]
            )
            self.filter_deque.append(event_frame)

    def validate(self) -> None:
        assert len(self.aligned_sink_pads) == 1 and len(self.source_pads) == 1, (
            f"Correlate requires exactly one aligned sink pad and one "
            f"source pad, got {len(self.aligned_sink_pads)} aligned sink "
            f"pads and {len(self.source_pads)} source pads"
        )
        assert self.sample_rate != -1, "Sample rate must be specified (not -1)"

    @staticmethod
    def _extract_filter(item: EventBuffer | EventFrame) -> Array:
        """Extract the filter from an event buffer or frame."""
        if len(item.events) > 1:
            msg = "found more than one event in {item}, " "cannot extract filter."
            raise ValueError(msg)
        event = item.events[0]
        if isinstance(event, Event):
            return event.data
        return event

    @property
    def filters_cur(self) -> Optional[EventFrame]:
        """Get the current filters"""
        if len(self.filter_deque) == 0:
            return None
        return self.filter_deque[0]

    @property
    def filters_new(self) -> EventFrame | None:
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
        if self.filters_cur is None:
            return False

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

    def pull(self, pad: SinkPad, frame: TimeSpanFrame) -> None:
        # Pull the data from the sink pad
        super().pull(pad, frame)

        if frame.is_gap:
            return

        # If the pad is the special filter sink pad, then update filter
        # metadata values
        if pad.name == self.filter_pad.name:
            _, input_frame = self.next_event_input()
            new_filter = self._extract_filter(input_frame)

            # If the buffer is null, then short circuit
            if new_filter is None:
                return

            # Redundant check, but more generalizable?
            if len(self.filter_deque) > 1:
                if self.ignore_rapid_updates:
                    if self.verbose:
                        warnings.warn(
                            f"Ignoring rapid filter update at" f" {input_frame.start}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    return
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
            self.filter_deque.append(input_frame)

    def internal(self) -> None:
        """Override internal to perform correlation with both current and
        new filters when adapting, and just current filters when not adapting.
        """
        # Check if we're adapting without needing frames
        if not self.is_adapting:
            # Just delegate to parent Correlate implementation
            super().internal()
            return

        # If we're adapting, we need to do custom processing
        # Call base TSTransform internal for alignment/preparation
        TSTransform.internal(self)

        # Get aligned buffer to see if overlaps with new filters
        _, input_frame = self.next_input()
        _, output_collector = self.next_output()

        if self.can_adapt(input_frame):
            if self.verbose:
                print(f"Adapting to new filters at {input_frame.slice}")
            # Correlate with current filters
            assert self.filters_cur is not None
            self.filters = self._extract_filter(self.filters_cur)

            for buf in input_frame:
                assert not buf.is_gap
                data_cur = self.corr(buf.data)

                # Change the state of filters
                assert self.filters_new is not None
                self.filters = self._extract_filter(self.filters_new)
                data_new = self.corr(buf.data)

                # Combine data with window functions

                # Compute window functions. Window functions
                # will be piecewise functions for the corresponding
                # intersection of the filter slice and data slice
                # where the window function is 0.0 before the intersection
                # and 1.0 after the intersection, and cos^2 in between
                N = data_cur.shape[-1]
                win_new = (scipy.signal.windows.cosine(2 * N, sym=True) ** 2)[:N]
                win_cur = 1.0 - win_new

                data = win_cur * data_cur + win_new * data_new
                shape = data.shape

                buf = buf.copy(offset=output_collector.offset, data=data, shape=shape)
                output_collector.append(buf)

            # Remove the new filters to indicate adaptation is complete
            self.filter_deque.popleft()
        else:
            # We're adapting but this frame doesn't overlap with the new filter
            # Just do normal correlation with current filters
            assert self.filters_cur is not None
            self.filters = self._extract_filter(self.filters_cur)
            self._transform(input_frame, output_collector)

        # Close the collector to commit buffers
        output_collector.close()
