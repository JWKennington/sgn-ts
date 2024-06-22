from dataclasses import dataclass

import numpy as np

from sgn.base import TransformElement

from ..base import Audioadapter, Offset, SeriesBuffer, TSFrame


@dataclass
class Sync(TransformElement):
    """
    Synchronizes buffers

    Parameters:
    -----------
    mode: str
        Supports the following modes:
        (1) pad: pads missing data to match the oldest data across buffers
        (2) drop: drop old data, only data common to all buffers will be produced
    pad_names_map: dict
        link map between source pad and sink pad within this transform,
        pad_names_map = {'src1_pad':'sink1_pad','src2_pad':'sink2_pad',...}
    """

    mode: str = None
    pad_names_map: dict[str, str] = None

    def __post_init__(self):
        self.inbufs = {}
        self.audioadapters = {}
        self.segments = {}
        self.outbufs = {}
        self.source_pad_names = self.pad_names_map.keys()
        self.sink_pad_names = self.pad_names_map.values()
        self.pad_map = {}
        # rename pad_map
        for k, v in self.pad_names_map.items():
            self.pad_map["%s:src:%s" % (self.name, k)] = "%s:sink:%s" % (self.name, v)

        super().__post_init__()
        for sink_pad in self.sink_pads:
            self.audioadapters[sink_pad.name] = Audioadapter()

    def pull(self, pad, bufs):
        self.inbufs[pad] = bufs
        for buf in bufs:
            self.audioadapters[pad.name].push(buf)
        self.segments[pad.name] = self.audioadapters[
            pad.name
        ].get_available_offset_segment()

    def transform(self, pad):
        frames = list(self.inbufs.values())
        EOS = any(b.EOS for b in frames)

        sink_pad = self.pad_map[pad.name]

        # Check if buffers are aligned in time
        oldsegs = [seg[0] for seg in self.segments.values()]
        newsegs = [seg[1] for seg in self.segments.values()]
        aligned = len(set(oldsegs)) == 1
        A = self.audioadapters[sink_pad]

        if aligned:
            output_segment = (min(oldsegs), min(newsegs))
            noffset = output_segment[1] - output_segment[0]
            seg = self.segments[sink_pad]
            overlap = (max(output_segment[0], seg[0]), min(output_segment[1], seg[1]))
            data = A.copy_samples_by_offset_segment(overlap, pad_zeros=True)
            A.flush_samples_by_end_offset_segment(overlap[1])
        else:
            if self.mode == "pad":
                output_segment = (min(oldsegs), min(newsegs))
                noffset = output_segment[1] - output_segment[0]
                # FIXME: are there cases where noffset is negative?
                seg = self.segments[sink_pad]
                # find overlap
                overlap = (
                    max(output_segment[0], seg[0]),
                    min(output_segment[1], seg[1]),
                )
                if overlap[1] <= overlap[0]:
                    data = None
                else:
                    data = A.copy_samples_by_offset_segment(overlap, pad_zeros=True)
                    A.flush_samples_by_end_offset_segment(overlap[1])
            elif self.mode == "drop":
                output_segment = (max(oldsegs), min(newsegs))
                noffset = output_segment[1] - output_segment[0]
                if noffset <= 0:
                    # produce empty buffers
                    data = None
                else:
                    seg = self.segments[sink_pad]
                    overlap = (
                        max(output_segment[0], seg[0]),
                        min(output_segment[1], seg[1]),
                    )
                    data = A.copy_samples_by_offset_segment(overlap, pad_zeros=True)
                    self.audioadapters[sink_pad].flush_samples_by_end_offset_segment(
                        overlap[1]
                    )
            else:
                raise ValueError("Unknown mode")
        outbuf = SeriesBuffer(
            offset=output_segment[0],
            sample_rate=frames[0].sample_rate,
            data=data,
            shape=frames[0].shape[:-1]
            + (Offset.tosamples(max(noffset, 0), A.sample_rate),),
        )
        # self.outbufs.pop(sink_pad)
        return TSFrame(buffers=[outbuf], EOS=EOS)
