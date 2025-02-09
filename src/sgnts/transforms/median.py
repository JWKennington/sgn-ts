from dataclasses import dataclass

import numpy

from sgnts.base import AdapterConfig, Offset, TSFrame, TSTransform


@dataclass
class Median(TSTransform):
    """Computes a running median over the previous and afterward
    "overlap_offsets". The output will be at the same sample rate as the input"""

    overlap_offsets: tuple[int, int] = (1024, 1024)

    def __post_init__(self):
        # FIXME: When this option is available, fill the gap buffers with nan's
        # (instead of 0's)
        self.adapter_config = AdapterConfig(
            overlap=self.overlap_offsets,
            pad_zeros_startup=False,
            stride=Offset.SAMPLE_STRIDE_AT_MAX_RATE,
        )
        super().__post_init__()
        # This element is written to assume one channel, one source pad and one sink pad
        assert len(self.source_pads) == len(self.sink_pads) == 1

    def running_median(self, inbuf, outbuf):
        num_samples = Offset.tosamples(
            self.overlap_offsets[0] + self.overlap_offsets[1], inbuf.sample_rate
        )
        if outbuf.shape != 0:
            for i in range(len(outbuf.data)):
                outbuf.data[i] = numpy.nanmedian(inbuf.data[i : i + num_samples])
        return outbuf

    def new(self, pad):
        frame = self.preparedframes[self.sink_pads[0]]
        # We expect all frames to have exactly one buffer
        assert len(frame) == 1
        if frame[0].duration > 0:
            outbuf = self.running_median(
                frame[0], self.adapter_config.valid_buffer(frame[0])
            )
        else:
            return frame
        return TSFrame(buffers=[outbuf], EOS=frame.EOS)
