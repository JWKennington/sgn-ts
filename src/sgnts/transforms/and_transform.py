from __future__ import annotations

from dataclasses import dataclass, field

from sgn import validator
from sgn.base import SinkPad

from sgnts.base import (
    ArrayBackend,
    NumpyBackend,
    SeriesBuffer,
    TSCollectFrame,
    TSFrame,
    TSSlice,
    TSSlices,
    TSTransform,
)
from sgnts.decorators import transform


@dataclass
class ANDTransform(TSTransform):
    """Perform logical AND operation across multiple input streams based on gap status.

    This transform takes multiple input streams with arbitrary sample rates and
    produces a single output stream at the maximum sample rate among all inputs.

    Output behavior:
    - Where ALL inputs have non-gap data: output is 1 (logical AND)
    - Where ANY input has a gap: output is a gap buffer (data=None)

    This approach is semantically correct as gaps represent "no valid data" rather
    than zeros. Downstream transforms can decide how to handle gaps (e.g., convert
    to zeros if needed).

    The transform uses TSSlice set logic to determine overlapping regions and
    efficiently creates output buffers using the split method with contiguous=True.

    Args:
        backend:
            type[ArrayBackend], the wrapper around array operations
        output_shape:
            Optional[tuple[int, ...]], shape of output samples (excluding time
            dimension).
            If None, defaults to scalar output ()
    """

    backend: type[ArrayBackend] = NumpyBackend
    output_shape: tuple[int, ...] = field(default_factory=tuple)

    def configure(self) -> None:
        # Explicitly disable adapter to prevent gap filling
        self.adapter_config.enable = False

    @validator.many_to_one
    def validate(self) -> None:
        pass

    @transform.many_to_one
    def process(
        self, input_frames: dict[SinkPad, TSFrame], output_frame: TSCollectFrame
    ) -> None:
        """Generate output frame with AND logic across all inputs.

        Output contains buffers with 1s where all inputs have data,
        and gap buffers where any input has gaps.
        """
        frames = list(input_frames.values())

        # Use the maximum sample rate among all inputs for output
        max_rate = max(f.sample_rate for f in frames)

        # Collect non-gap slices from each input
        all_nongap_slices = [
            TSSlices([buf.slice for buf in frame if not buf.is_gap]) for frame in frames
        ]

        # Find intersection of all non-gap regions (where ALL inputs have data)
        intersection_slices = TSSlices.intersection_of_multiple(all_nongap_slices)

        # Create initial buffer spanning entire frame with 1s
        frame_slice = TSSlice(frames[0].offset, frames[0].end_offset)
        full_buffer = SeriesBuffer.fromoffsetslice(
            frame_slice,
            sample_rate=max_rate,
            data=1,  # Buffer filled with ones
            channels=self.output_shape,
        )

        # Split buffer at intersection boundaries:
        # - Keeps 1s where we have intersections (all inputs have data)
        # - Creates gap buffers (data=None) between intersections
        # If no intersections exist, create a single gap buffer
        output_buffers = (
            full_buffer.split(intersection_slices, contiguous=True)
            if intersection_slices.slices
            else [full_buffer.new()]  # new() with no data creates a gap buffer
        )

        output_frame.extend(output_buffers)
