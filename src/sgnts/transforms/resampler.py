from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from ..base import Audioadapter, SeriesBuffer, TSTransform, TSFrame, Offset


@dataclass
class Resampler(TSTransform):
    """
    Up/down samples time-series data

    Assumptions:
    ------------
    - There is only one sink pad and one source pad
    """

    def __post_init__(self):
        factor = self.outrate / self.inrate
        self.factor = factor
        self.next_out_offset = None
        # self.audioadapter = Audioadapter()

        if self.outrate < self.inrate:
            # downsample parameters
            self.half_length = int(32 / factor)
            self.kernel_length = self.half_length * 2 + 1
            self.thiskernel = self.downkernel(factor)
        else:
            # upsample parameters
            self.half_length = 8
            self.kernel_length = self.half_length * 2 + 1
            self.thiskernel = self.upkernel(factor)

        self.overlap = (self.half_length, self.half_length)

        super().__post_init__()

        self.pad_length = self.half_length

        assert (
            len(self.sink_pads) == 1 and len(self.source_pads) == 1
        ), "only one sink_pad and one source_pad is allowed"

    def downkernel(self, factor: float):
        """
        Compute the kernel for downsampling
        """
        kernel_length = int(2 * self.half_length + 1)

        # the domain should be the kernel_length divided by two
        c = kernel_length // 2
        x = np.arange(-c, c + 1)
        vecs = np.sinc(x * factor) * np.sinc(x / c)
        norm = np.linalg.norm(vecs) / factor**0.5
        vecs = vecs / norm

        return vecs.reshape(1, -1)

    def upkernel(self, factor: float):
        """
        Compute the kernel for upsampling
        """
        factor = int(factor)

        kernel_length = int(2 * self.half_length * factor + 1)
        sub_kernel_length = int(2 * self.half_length + 1)

        # the domain should be the kernel_length divided by two
        c = kernel_length // 2
        x = np.arange(-c, c + 1)
        out = np.sinc(x / factor) * np.sinc(x / c)
        out = np.pad(out, (0, factor - 1))
        # FIXME: check if interleave same as no interleave
        vecs = out.reshape(-1, factor).T[:, ::-1]

        return vecs.reshape(int(factor), 1, sub_kernel_length)

    def resample(self, data0, outshape):
        data = data0.reshape(-1, data0.shape[-1])

        if self.factor > 1:
            # upsample
            os = []
            for i in range(int(self.factor)):
                os.append(correlate(data, self.thiskernel[i], mode="valid"))
            out = np.vstack(os)
            out = np.moveaxis(out, -1, -2)
        else:
            # downsample
            # FIXME: implement a strided correlation, rather than doing unnecessary calculations
            out = correlate(data, self.thiskernel, mode="valid")[
                ..., :: int(1 / self.factor)
            ]
        return out.reshape(outshape)

    def transform(self, pad):
        frame = self.preparedframes[self.sink_pads[0]]
        outframe = self.preparedoutframes[self.sink_pads[0]]
        if frame.shape[-1] > 0 and not frame.is_gap:
            for buf, outbuf in zip(frame, outframe):
                outdata = self.resample(buf.data, outbuf.shape)
                outbuf.update_data(outdata)

        return outframe
