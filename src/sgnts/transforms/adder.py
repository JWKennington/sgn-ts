from __future__ import annotations

from dataclasses import dataclass

from sgn import validator
from sgn.base import SinkPad

from sgnts.base import (
    ArrayBackend,
    NumpyBackend,
    SeriesBuffer,
    TSCollectFrame,
    TSFrame,
    TSTransform,
)
from sgnts.decorators import transform


@dataclass
class Adder(TSTransform):
    """Add up all the frames from all the sink pads.

    Args:
        backend:
            type[ArrayBackend], the wrapper around array operations
        addslices_map:
            Optional[dict[str, tuple[slice, ...]], a mapping of sink_pad_names to a
            tuple of slice objects, representing array index slices in each dimension
            except the last. Suppose there are two sink pads "sink_pad_name1" and
            "sink_pad_name2", and data1 is the data from sink_pad_name1, and data2 is
            the data from sink_pad_name2, and addslices_map = {"sink_pad_name2":
            (slice(2, 6), slice(0, 8))}, then this element will perform the following
            operation:

                out = data1[slice(2, 6), slice(0, 8), :] + data2
    """

    backend: type[ArrayBackend] = NumpyBackend
    addslices_map: dict[str, tuple[slice, ...]] | None = None

    @validator.many_to_one
    def validate(self) -> None:
        pass

    @transform.many_to_one
    def process(
        self, input_frames: dict[SinkPad, TSFrame], output_frame: TSCollectFrame
    ) -> None:
        """Add up all the frames from all the sink pads."""
        frames = list(input_frames.values())

        # Sanity check frames
        assert (
            len({f.sample_rate for f in frames}) == 1
        ), "Sample rate of frames must be the same"
        assert len({f.offset for f in frames}) == 1, "Frames must be aligned"
        assert len({f.end_offset for f in frames}) == 1, "Frames must be aligned"

        if self.addslices_map is None:
            assert (
                len({f.shape for f in frames}) == 1
            ), "Shape of frames must be the same"
        else:
            assert (
                len({f.shape[-1] for f in frames}) == 1
            ), "Size of last dimension must be the same"

        if all(frame.is_gap for frame in frames):
            # Return a gap buffer if all frames are gaps
            out = None
            shape = frames[0].shape
        else:
            # use the first frame as basis
            if len(frames[0]) == 1:
                out = frames[0][0].filleddata(self.backend.zeros)
            else:
                out = self.backend.cat(
                    [buf.filleddata(self.backend.zeros) for buf in frames[0]],
                    axis=-1,
                )
            shape = out.shape
            # add to the first frame
            for i, f in enumerate(frames[1:]):
                i0 = 0
                for buf in f:
                    if not buf.is_gap:
                        if self.addslices_map is None:
                            out[..., i0 : i0 + buf.samples] += buf.data
                        else:
                            slices = self.addslices_map[self.sink_pad_names[i + 1]] + (
                                slice(i0, i0 + buf.samples),
                            )
                            out[slices] += buf.data

                    i0 += buf.samples

        output_frame.append(
            SeriesBuffer(
                offset=frames[0].offset,
                sample_rate=frames[0].sample_rate,
                data=out,
                shape=shape,
            )
        )
