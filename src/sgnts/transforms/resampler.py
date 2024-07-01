from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from ..base import (
    Audioadapter,
    SeriesBuffer,
    TSTransform,
    TSFrame,
    Offset,
    AdapterConfig,
)

UP_HALF_LENGTH = 8
DOWN_HALF_LENGTH = 32

@dataclass
class Resampler(TSTransform):
    """
    Up/down samples time-series data

    Parameters:
    -----------
    inrate: int
        sample rate of the input frames
    outrate: int
        sample rate of the output frames

    Assumptions:
    ------------
    - There is only one sink pad
    """

    inrate: int = None
    outrate: int = None

    def __post_init__(self):
        factor = self.outrate / self.inrate
        self.factor = factor
        self.next_out_offset = None
        # self.audioadapter = Audioadapter()

        if self.outrate < self.inrate:
            # downsample parameters
            self.half_length = int(DOWN_HALF_LENGTH / factor)
            self.kernel_length = self.half_length * 2 + 1
            self.thiskernel = self.downkernel(factor)
        else:
            # upsample parameters
            self.half_length = UP_HALF_LENGTH
            self.kernel_length = self.half_length * 2 + 1
            self.thiskernel = self.upkernel(factor)

        if self.adapter_config is None:
            self.adapter_config = AdapterConfig()
        self.adapter_config.overlap = (self.half_length, self.half_length)
        self.adapter_config.pad_zeros_startup = True

        super().__post_init__()

        self.pad_length = self.half_length

        assert (
            len(self.sink_pads) == 1
        ), "only one sink_pad"

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
        outoffsets = self.preparedoutoffsets[self.sink_pads[0]]

        outbufs = []
        if frame.shape[-1] == 0:
            outbufs.append(
                SeriesBuffer(
                    offset=outoffsets[0]["offset"],
                    sample_rate=self.outrate,
                    data=None,
                    shape=frame.shape,
                )
            )
        else:
            for i, buf in enumerate(frame):
                shape = frame.shape[:-1] + (
                    Offset.tosamples(outoffsets[i]["noffset"], self.outrate),
                )
                if buf.is_gap:
                    data = None
                else:
                    data = self.resample(buf.data, shape)
                outbufs.append(
                    SeriesBuffer(
                        offset=outoffsets[i]["offset"],
                        sample_rate=self.outrate,
                        data=data,
                        shape=shape,
                    )
                )

        return TSFrame(buffers=outbufs, EOS=frame.EOS, metadata=frame.metadata)
