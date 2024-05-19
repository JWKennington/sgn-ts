from . import *
from .. base import *

@dataclass
class Adder(TransformElement):
    addslice: slice = slice(None)
    rescale: float = 1
    frombuf_pad: str = None
    tobuf_pad: str = None

    def __post_init__(self):
        self.inbuf = {}
        self.audioadapters = {}

        super().__post_init__()

    def get_buffer(self, pad, buf):
        self.inbuf[pad] = buf
        #self.inbuf[pad].metadata = {'name':pad.name,'cnt':{pad.name:1}}
        if pad not in self.audioadapters:
            self.audioadapters[pad] = Audioadapter()
        self.audioadapters[pad].push(buf)

    def transform_buffer(self, pad):
        """
        Add frombuf.data to tobuf.data

        Arguments:
        ----------
        frombuf: SeriesBuffer
            The buffer to add from
        tobuf: SeriesBuffer
            The buffer to add to
        """
        #frombuf = self.inbuf[self.frombuf_pad]
        #tobuf = self.inbuf[self.tobuf_pad]

        EOS = any(b.EOS for b in self.inbuf.values())
        metadata = {"cnt:%s" % b.metadata['name']:b.metadata['cnt'] for b in self.inbuf.values()}
        metadata["name"] = "%s -> '%s'" % ("+".join(b.metadata["name"] for b in self.inbuf.values()), pad.name)

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
            return SeriesBuffer(offset = offset, noffset = 0, offset_ref_t0 = offset_ref_t0, data = None, is_gap = True, metadata = metadata, EOS=EOS)
        else:

            # Check if all gaps
            if fromA.is_gap() and toA.is_gap():
                return SeriesBuffer(offset = offset, noffset = noffset, offset_ref_t0 = offset_ref_t0, data = None, is_gap = True, metadata = metadata, EOS=EOS)
            elif fromA.is_gap():
                # FIXME
                return 
            elif toA.is_gap():
                # FIXME
                return 

            fromdata, _, _ = fromA.copy_samples_by_offset_segment(overlap_segment)
            todata, _, _ = toA.copy_samples_by_offset_segment(overlap_segment)
            #if (
            #    not (frombuf.is_gap and tobuf.is_gap)
            #    and fromdata is not None
            #    and fromdata.shape[-1] > 0
            #):
            #assert tobuf.offset + tobuf.noffset == frombuf.offset + frombuf.noffset, (
            #    f"end offset does not match {frombuf.sample_rate=}"
            #    f" {tobuf.sample_rate=} {(tobuf.offset + tobuf.noffset)=}"
            #    f" {(frombuf.offset + frombuf.noffset)=}"
            #)
            #if 0 in tobuf.data.stride():
            #    # actually allocate memory of expanded tensor
            #    tobuf.data = tobuf.data.clone()
            #tobuf.data[self.addslice,  -fromshape[-1] :] += fromdata * self.rescale
            todata += fromdata * self.rescale

            outbuf = SeriesBuffer(offset = offset, noffset = noffset, offset_ref_t0 = offset_ref_t0, data = todata, metadata=metadata, EOS=EOS)

            fromA.flush_samples_by_end_offset_segment(overlap_segment[1])
            toA.flush_samples_by_end_offset_segment(overlap_segment[1])

            return outbuf

transforms_registry += ("Adder",)
