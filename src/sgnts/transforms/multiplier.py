from dataclasses import dataclass, field
from ..base import Audioadapter, SeriesBuffer, TSFrame, TSTransform
import numpy as np

@dataclass
class Multiplier(TSTransform):
    rescale: float = 1
    num_samples: int = 1
    def __post_init__(self):
        super().__post_init__()
        self.inbuf = {}
        self.audioadapters = {}

    def pull(self, pad, buf):
        self.inbuf[pad] = buf
        if pad not in self.audioadapters:
            self.audioadapters[pad.name] = Audioadapter()
        for n in buf:
            self.audioadapters[pad.name].push(n)

    def transform(self, pad):
        EOS = any(b.EOS for b in self.inbuf.values())
        sample_rate = self.audioadapters[str(self.sink_pads[0].name)].buffers[0].sample_rate 
        metadata = {"cnt:%s" % b.metadata['name']:b.metadata['cnt'] for b in self.inbuf.values()}
        metadata["name"] = "%s -> '%s'" % ("*".join(b.metadata["name"] for b in self.inbuf.values()), pad.name)
        
        #makes two dictionaries containing respective offset bounderies
        minsegs = {str(n.name): [self.audioadapters[str(n.name)].get_available_offset_segment()[0]] for n in self.sink_pads} 
        maxsegs = {str(n.name): [self.audioadapters[str(n.name)].get_available_offset_segment()[1]] for n in self.sink_pads} 

        # Will only produce an output buffer with sum of the data in the overlap_segment
        overlap_segment = (max(minsegs.values())[0], min(maxsegs.values())[0]) #finds the overlap of the offsets that we are working with
        noffset = overlap_segment[1] - overlap_segment[0]
        offset = overlap_segment[0]
        if noffset <= 0: #for when there is no overlap in the offsets
            return TSFrame(
                    buffers=[
                    SeriesBuffer(
                        offset=offset,
                        sample_rate=sample_rate,
                        data=None,
                        is_gap=True
                        )
                    ],
                    EOS=EOS,
                    metadata=metadata
                    )
        else:
            bothgaps = all(self.audioadapters[str(n.name)].is_gap() for n in self.sink_pads)
            # Check if all gaps
            if bothgaps:
                return TSFrame(
                        buffers=[
                        SeriesBuffer(
                            offset=offset,
                            sample_rate=sample_rate,
                            data=None,
                            is_gap=True
                            )
                        ],
                        EOS=EOS,
                        metadata=metadata
                        )
            data=list(1 for i in range(self.num_samples))
            samples = (self.audioadapters[str(n.name)].copy_samples_by_offset_segment(overlap_segment) for n in self.sink_pads)
            for sample in samples:
                for point in range(len(sample)):
                    data[point] *= sample[point]
            data=np.array(data) #data must be in tuple format for export and list format for calculateion

            return TSFrame(
                    buffers=[
                    SeriesBuffer(
                        offset=offset,
                        sample_rate=sample_rate,
                        data=data,
                        )
                    ],
                    EOS=EOS,
                    metadata=metadata
                    )
