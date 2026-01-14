from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sgn.base import SinkPad

from sgnts.base import (
    AdapterConfig,
    Offset,
    SeriesBuffer,
    TSCollectFrame,
    TSFrame,
    TSTransform,
)
from sgnts.decorators import transform


@dataclass
class BitVector(TSTransform):
    """Generate buffer/gap state as integer values for N input streams.

    Takes N input streams and produces a single output stream containing
    single-channel integer values. Each value is calculated by interpreting
    the buffer/gap state of all inputs as a binary number.

    Unlike ANDTransform which outputs gaps where any input has a gap,
    BitGen always outputs buffers (never gaps), while also preserving
    information about the gap/buffer status of each input.

    Output characteristics:
        - Sample rate: Configurable via output_rate parameter
        - Data type: uint8 (unsigned 8-bit integer)
        - Shape: (1, num_samples) - single channel with integer values
        - Always produces buffers, never gaps
        - State is based on last arrived frame for each input

    For example:
        3 inputs with different gap patterns at rates [64, 128, 256] Hz
        Input 0: buffer at t=0.1-0.5s, gap elsewhere
        Input 1: buffer at t=0.3-0.7s, gap elsewhere
        Input 2: gap everywhere
        With output_rate=128:
        At t=0.05s: 0 (binary 000 = decimal 0)
        At t=0.2s: 4 (binary 100 = decimal 4)
        At t=0.4s: 6 (binary 110 = decimal 6)
        At t=0.6s: 2 (binary 010 = decimal 2)

    Args:
        output_rate:
            int, the sample rate for the output stream in Hz.
            Default: None (uses minimum rate among all inputs)

    """

    output_rate: Optional[int] = None

    def configure(self) -> None:
        """Initialize transform with buffer alignment enabled."""
        self.adapter_config = AdapterConfig(align_buffers=True)

    @transform.many_to_one
    def process(
        self, input_frames: dict[SinkPad, TSFrame], output_frame: TSCollectFrame
    ) -> None:
        """Generate output frame with buffer/gap state as integer values.

        Output contains buffers with single-channel integer values where:
        - Each integer represents the binary state of all inputs
        - Bit i (from left) is 1 if input i has a buffer, 0 if gap
        - Output never contains gap buffers

        Algorithm:
            1. Get all aligned frames (aligned to same boundaries via align_buffers)
            2. Determine output sample rate
            3. All frames have same number of buffers after alignment
            4. For each buffer index (time region):
               a. Check each input independently for buffer/gap state
               b. Create state vector [1, 0, 1, ...] indicating state
               c. Convert binary state vector to single integer
               d. Broadcast integer across all samples in region
               e. Create output buffer
            5. Append all output buffers to output_frame
        """
        N = len(input_frames)

        # determine output sample rate
        if self.output_rate is None:
            # use minimum rate among inputs if not configured
            output_rate = min(f.sample_rate for f in input_frames.values())
        else:
            output_rate = self.output_rate

        # after align_buffers, all frames have same number of buffers at same boundaries
        # iterate through buffers by index
        for input_buffers in zip(*input_frames.values()):
            # for each input, independently check if it has a buffer or gap
            state_vector = np.array(
                [int(not buf.is_gap) for buf in input_buffers], dtype=np.uint8
            )

            # convert binary state vector to integer
            # e.g., [1, 1, 1] -> 7, [1, 0, 1] -> 5
            int_value = np.dot(state_vector, 2 ** np.arange(N - 1, -1, -1))

            # all buffers at this index have same time span (aligned)
            num_samples = Offset.tosamples(input_buffers[0].noffset, output_rate)

            # create data array: (1, num_samples)
            # single row containing int_value repeated for all samples
            data = np.full((1, num_samples), int_value, dtype=np.uint32)

            # create output buffer for this region
            buffer = SeriesBuffer(
                offset=input_buffers[0].offset,
                sample_rate=output_rate,
                data=data,
                shape=(1, num_samples),
            )
            output_frame.append(buffer)
