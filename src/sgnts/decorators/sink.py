"""Decorators for simplifying sink implementations."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from sgn.base import SinkPad

    from sgnts.base import TSFrame, TSSink

T = TypeVar("T", bound="TSSink")


def single_pad(
    method: Callable[[T, TSFrame], None],
) -> Callable[[T, dict[SinkPad, TSFrame]], None]:
    """Decorator for simple single-pad sinks.

    Transforms the simpler signature::

        def process(self, input_frame: TSFrame) -> None

    Into the dict-based signature expected by TSSink::

        def process(self, input_frames: dict[SinkPad, TSFrame]) -> None

    Usage::

        from sgnts.decorators import sink

        class MySink(TSSink):
            @sink.single_pad
            def process(self, input_frame: TSFrame) -> None:
                # Process single input frame
                pass
    """

    @wraps(method)
    def wrapper(
        self: T,
        input_frames: dict[SinkPad, TSFrame],
    ) -> None:
        assert len(self.sink_pads) == 1, (
            f"@sink.single_pad requires exactly one sink pad, "
            f"got {len(self.sink_pads)}"
        )
        return method(self, input_frames[self.sink_pads[0]])

    return wrapper
