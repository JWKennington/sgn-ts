"""NAry transforms."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional

import numpy as np
from sgn.base import SourcePad

from sgnts.base import SeriesBuffer, TSFrame, TSTransform


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

    op: Optional[Callable] = None

    def __post_init__(self):
        """Checks"""
        super().__post_init__()
        # Check op is not None
        assert self.op is not None, "op must be provided"

        # Check only 1 output pad
        assert len(self.source_pads) == 1

        # Validate the operator and pads
        self._validate_op()

        # Extra attrs
        self._data = None

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

    @wraps(TSTransform.new)
    def new(self, pad: SourcePad) -> TSFrame:  # type: ignore
        """New method"""
        # Get all prepared frames
        prepped_pad_buffers = [
            self.preparedframes[snk].buffers for snk in self.sink_pads
        ]

        # Check all prepared frames have same number of buffers, this
        # is to make sure that zip doesn't silently drop any buffers
        assert all(
            len(b) == len(prepped_pad_buffers[0]) for b in prepped_pad_buffers
        ), (
            "Prepared frames have different number "
            "of buffers, expected same number of "
            "buffers for all sink pads, got:"
            f" {[len(b) for b in prepped_pad_buffers]}"
        )

        # Apply the operator to zipped groups of buffers
        bufs = [self.apply(*b) for b in zip(*prepped_pad_buffers)]

        # Assemble the frame and return
        return TSFrame(buffers=bufs, EOS=self.at_EOS)


@dataclass
class Multiply(NaryTransform):
    """Multiply transform"""

    def __post_init__(self):
        """Post init"""
        # Force the operator to be multiplication
        self.op = _multiply
        super().__post_init__()


def _multiply(*arrays):
    """Multiple op"""
    output = arrays[0]
    for arr in arrays[1:]:
        output = output * arr
    return output


@dataclass
class Real(NaryTransform):
    """Extract Real component of single input"""

    def __post_init__(self):
        """Post init"""
        # Force the operator to be multiplication
        self.op = _real
        super().__post_init__()


def _real(*arrays):
    """Multiple op"""
    assert len(arrays) == 1, f"Real operator only takes one input, got {len(arrays)}"
    return np.real(arrays[0])
