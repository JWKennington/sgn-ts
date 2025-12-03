"""Decorators for simplifying transform implementations."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from sgn.base import SinkPad, SourcePad

    from sgnts.base import TSFrame, TSTransform

T = TypeVar("T", bound="TSTransform")


def one_to_one(
    method: Callable[[T, TSFrame, TSFrame], None],
) -> Callable[[T, dict[SinkPad, TSFrame], dict[SourcePad, TSFrame]], None]:
    """Decorator for simple one-to-one transforms.

    Transforms the simpler signature::

        def process(self, input_frame: TSFrame, output_frame: TSFrame) -> None

    Into the dict-based signature expected by TSTransform::

        def process(self, input_frames: dict[SinkPad, TSFrame],
                    output_frames: dict[SourcePad, TSFrame]) -> None

    Usage::

        from sgnts.decorators import transform

        class MyTransform(TSTransform):
            @transform.one_to_one
            def process(self, input_frame: TSFrame, output_frame: TSFrame) -> None:
                for buf in input_frame:
                    output_frame.append(buf)
    """

    @wraps(method)
    def wrapper(
        self: T,
        input_frames: dict[SinkPad, TSFrame],
        output_frames: dict[SourcePad, TSFrame],
    ) -> None:
        assert len(self.sink_pads) == 1 and len(self.source_pads) == 1, (
            f"@transform.one_to_one requires exactly one sink and source pad, "
            f"got {len(self.sink_pads)} sink pads and "
            f"{len(self.source_pads)} source pads"
        )
        return method(
            self, input_frames[self.sink_pads[0]], output_frames[self.source_pads[0]]
        )

    return wrapper


def many_to_one(
    method: Callable[[T, dict[SinkPad, TSFrame], TSFrame], None],
) -> Callable[[T, dict[SinkPad, TSFrame], dict[SourcePad, TSFrame]], None]:
    """Decorator for many-to-one transforms.

    Transforms the simpler signature::

        def process(
            self, input_frames: dict[SinkPad, TSFrame], output_frame: TSFrame
        ) -> None

    Into the dict-based signature expected by TSTransform::

        def process(
            self, input_frames: dict[SinkPad, TSFrame],
            output_frames: dict[SourcePad, TSFrame]
        ) -> None

    Usage::

        from sgnts.decorators import transform

        class MyTransform(TSTransform):
            @transform.many_to_one
            def process(
                self, input_frames: dict[SinkPad, TSFrame], output_frame: TSFrame
            ) -> None:
                # Process multiple inputs to single output
                pass
    """

    @wraps(method)
    def wrapper(
        self: T,
        input_frames: dict[SinkPad, TSFrame],
        output_frames: dict[SourcePad, TSFrame],
    ) -> None:
        assert len(self.source_pads) == 1, (
            f"@transform.many_to_one requires exactly one source pad, "
            f"got {len(self.source_pads)}"
        )
        return method(self, input_frames, output_frames[self.source_pads[0]])

    return wrapper
