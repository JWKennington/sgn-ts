"""NAry transforms."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sgn import validator
from sgn.base import SinkPad

from sgnts.base import SeriesBuffer, TSCollectFrame, TSFrame, TSTransform
from sgnts.decorators import transform


@dataclass
class NaryTransform(TSTransform):
    """N-ary transform. Takes N inputs and applies a function to them
    frame by frame.

    Args:
        op:
            Callable, the operation to apply to the inputs. Must take N
            arguments, where N is the number of sink pads, and return a
            single output.
    """

    op: Callable | None = None

    @validator.many_to_one
    def validate(self) -> None:
        assert self.op is not None, "op must be provided"
        self._validate_op()

    def apply(self, *buffers: SeriesBuffer) -> SeriesBuffer:
        """Apply the operator to the given sequence of buffers"""
        # Check if there are any gaps
        if any(buf.is_gap for buf in buffers):
            data = None
        else:
            assert self.op is not None
            data = self.op(*[buf.data for buf in buffers])

        return SeriesBuffer(
            data=data,
            offset=buffers[0].offset,
            sample_rate=buffers[0].sample_rate,
            shape=buffers[0].shape,
        )

    @transform.many_to_one
    def process(
        self, input_frames: dict[SinkPad, TSFrame], output_frame: TSCollectFrame
    ) -> None:
        """Process multiple input frames to single output."""
        input_buffers = [frame.buffers for frame in input_frames.values()]

        # Check all prepared frames have same number of buffers, this
        # is to make sure that zip doesn't silently drop any buffers
        assert all(len(b) == len(input_buffers[0]) for b in input_buffers), (
            "Prepared frames have different number "
            "of buffers, expected same number of "
            "buffers for all sink pads, got:"
            f" {[len(b) for b in input_buffers]}"
        )

        # Apply the operator to zipped groups of buffers
        for buffers in zip(*input_buffers):
            buf = self.apply(*buffers)
            output_frame.append(buf)

    def _validate_op(self):
        """Validate the given operator to make sure it
        has the right number of arguments
        """
        sig = inspect.signature(self.op)

        # Check if the operator has var positional arguments,
        # meaning that it can accept and arbitrary number of arguments,
        # so we don't need to check if the number of pads is compatible
        if not any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        ):
            assert len(sig.parameters) == len(self.aligned_sink_pads), (
                "Operator must take arguments matching number of aligned pads. "
                f"Got {len(sig.parameters)} arguments, "
                f"expected {len(self.aligned_sink_pads)}"
            )


@dataclass
class Multiply(NaryTransform):
    """Multiply transform"""

    def configure(self) -> None:
        self.op = _multiply


def _multiply(*arrays):
    """Multiple op"""
    output = arrays[0]
    for arr in arrays[1:]:
        output = output * arr
    return output


@dataclass
class Real(NaryTransform):
    """Extract Real component of single input"""

    def configure(self) -> None:
        self.op = _real


def _real(*arrays):
    """Multiple op"""
    assert len(arrays) == 1, f"Real operator only takes one input, got {len(arrays)}"
    return np.real(arrays[0])
