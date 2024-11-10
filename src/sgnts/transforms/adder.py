from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sgn.base import SourcePad

from sgnts.base import ArrayBackend, NumpyBackend, SeriesBuffer, TSFrame, TSTransform


@dataclass
class Adder(TSTransform):
    """Add up all the frames from all the sink pads.

    Args:
        backend:
            type[ArrayBackend], the wrapper around array operations
        coeff_map:
            Optional[dict[str, float]], a mapping of sink pad names to coefficients.
            Suppose coeff_map = {"sink_pad_name1": 2, "sink_pad_name2": 4}, and data1
            is the data from sink_pad_name1, and data2 is the data from sink_pad_name2,
            then this element will perform the following operation:

                out = 2 * data1 + 4 * data2

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
    coeff_map: Optional[dict[str, float]] = None
    addslices_map: Optional[dict[str, tuple[slice, ...]]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.coeff_map is not None:
            assert self.sink_pad_names == tuple(self.coeff_map.keys())

    def transform(self, pad: SourcePad) -> TSFrame:
        """Add up all the frames from all the aligned sink pads.

        Args:
            pad:
                SourcePad, the pad to output the transform element

        Returns:
            TSFrame, the output TSFrame
        """
        frames = [self.preparedframes[self.snks[snk]] for snk in self.sink_pad_names]

        # Sanity check frames
        assert len(set(f.sample_rate for f in frames)) == 1, (
            "Sample rate of frames" " must be the same"
        )
        assert len(set(f.offset for f in frames)) == 1, "Frames must be aligned"
        assert len(set(f.end_offset for f in frames)) == 1, "Frames must be aligned"

        if self.addslices_map is None:
            assert len(set(f.shape for f in frames)) == 1, (
                "Shape of frames must be" "the same"
            )
        else:
            assert len(set(f.shape[-1] for f in frames)) == 1, (
                "Size of last" " dimension must be the same"
            )

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
                    [buf.filleddata(self.backend.zeros) for buf in frames[0]], axis=-1
                )
            if self.coeff_map is not None:
                out *= self.coeff_map[self.sink_pad_names[0]]
            shape = out.shape
            # add to the first frame
            for i, f in enumerate(frames[1:]):
                if self.coeff_map is not None:
                    coeff = self.coeff_map[self.sink_pad_names[i + 1]]
                else:
                    coeff = 1
                i0 = 0
                for buf in f:
                    if not buf.is_gap:
                        if self.addslices_map is None:
                            out[..., i0 : i0 + buf.samples] += buf.data * coeff
                        else:
                            slices = self.addslices_map[self.sink_pad_names[i + 1]] + (
                                slice(i0, i0 + buf.samples),
                            )
                            out[slices] += buf.data * coeff

                    i0 += buf.samples

        return TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=frames[0].offset,
                    sample_rate=frames[0].sample_rate,
                    data=out,
                    shape=shape,
                )
            ],
            EOS=frames[0].EOS,
            metadata=frames[0].metadata,
        )
