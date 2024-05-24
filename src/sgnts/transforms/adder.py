from dataclasses import dataclass, field

from sgn.base import TransformElement

from ..base import Audioadapter, SeriesBuffer


@dataclass
class Adder(TransformElement):
    """
    Add frombuf.data to tobuf.data

    Parameters:
    -----------
    frombuf_pad: str
        The pad name from which data will be added.
        Data on the "frombuf_pad" will be added to the data on the "tobuf_pad".
    tobuf_pad: str
        The pad name to which data will be added.
    rescale: float
        Rescale factor of frombuf data. tobuf.data += rescale*frombuf.data
    addslice: slice
        Used when the user only wants to add to a subset of channels in tobuf.data,
        frombuf.data is only added to tobuf.data[addslice,...]
    """
    frombuf_pad: str = None
    tobuf_pad: str = None
    rescale: float = 1
    addslice: slice = field(default_factory=lambda: slice(None))

    def __post_init__(self):
        self.inbufs = {}
        self.audioadapters = {}
        super().__post_init__()
        for sink_pad in self.sink_pads:
            self.audioadapters[sink_pad] = Audioadapter()

    def pull(self, pad, bufs):
        self.inbufs[pad] = bufs
        for buf in bufs:
            # There is a list of bufs, push sequentially
            self.audioadapters[pad].push(buf)

    def transform(self, pad):
        EOS = any(b[-1].EOS for b in self.inbufs.values())
        metadata = {
            "cnt:%s" % b[-1].metadata["name"]: b[-1].metadata["cnt"]
            for b in self.inbufs.values()
        }
        metadata["name"] = "(%s) -> '%s'" % (
            "+".join(b[-1].metadata["name"] for b in self.inbufs.values()),
            pad.name,
        )

        fromA = self.audioadapters[self.frombuf_pad]
        toA = self.audioadapters[self.tobuf_pad]

        seg_from = fromA.get_available_offset_segment()
        seg_to = toA.get_available_offset_segment()

        offset_ref_t0 = fromA.buffers[0].offset_ref_t0

        # Will only produce an output buffer with sum of the data in the overlap_segment
        overlap_segment = (max(seg_from[0], seg_to[0]), min(seg_from[1], seg_to[1]))
        noffset = overlap_segment[1] - overlap_segment[0]
        offset = overlap_segment[0]
        if noffset < 0:
            # FIXME
            return
        elif noffset == 0:
            return [SeriesBuffer(
                offset=offset,
                noffset=0,
                offset_ref_t0=offset_ref_t0,
                data=None,
                is_gap=True,
                metadata=metadata,
                EOS=EOS,
            )]
        else:
            # Check if all gaps
            if fromA.is_gap() and toA.is_gap():
                return [SeriesBuffer(
                    offset=offset,
                    noffset=noffset,
                    offset_ref_t0=offset_ref_t0,
                    data=None,
                    # FIXME: should we output zeros array?
                    is_gap=True,
                    metadata=metadata,
                    EOS=EOS,
                )]
            elif fromA.is_gap():
                # FIXME: just output toA
                return
            elif toA.is_gap():
                # FIXME: just output fromA
                return

            fromdata, _, _ = fromA.copy_samples_by_offset_segment(overlap_segment)
            todata, _, _ = toA.copy_samples_by_offset_segment(overlap_segment)
            # if (
            #    not (frombuf.is_gap and tobuf.is_gap)
            #    and fromdata is not None
            #    and fromdata.shape[-1] > 0
            # ):
            # assert tobuf.offset + tobuf.noffset == frombuf.offset + frombuf.noffset, (
            #    f"end offset does not match {frombuf.sample_rate=}"
            #    f" {tobuf.sample_rate=} {(tobuf.offset + tobuf.noffset)=}"
            #    f" {(frombuf.offset + frombuf.noffset)=}"
            # )
            # if 0 in tobuf.data.stride():
            #    # actually allocate memory of expanded tensor
            #    tobuf.data = tobuf.data.clone()
            # tobuf.data[self.addslice,  -fromshape[-1] :] += fromdata * self.rescale
            # FIXME: figure out how to make addslice more general
            todata += fromdata * self.rescale

            outbuf = SeriesBuffer(
                offset=offset,
                noffset=noffset,
                offset_ref_t0=offset_ref_t0,
                data=todata,
                metadata=metadata,
                EOS=EOS,
            )

            fromA.flush_samples_by_end_offset_segment(overlap_segment[1])
            toA.flush_samples_by_end_offset_segment(overlap_segment[1])

            return [outbuf]
