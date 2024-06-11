from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from ..base import OFFSET_RATE, Audioadapter, SeriesBuffer, TransformElement, TSFrame


@dataclass
class Resampler(TransformElement):
    """
    Up/down samples time-series data

    Parameters:
    -----------
    inrate: int
        sample rate of input data
    outrate: int
        sample rate of output data

    Assumptions:
    ------------
    - There is only one sink pad and one source pad
    """

    inrate: int = None
    outrate: int = None

    def __post_init__(self):
        factor = self.outrate / self.inrate
        self.factor = factor
        self.next_out_offset = None
        self.audioadapter = Audioadapter()

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

        self.pad_length = self.half_length
        self.offset_ref_t0 = None

        super().__post_init__()
        assert len(self.sink_pads) == 1 and len(self.source_pads) == 1, (
        "only one sink_pad and one source_pad is allowed")

    def pull(self, pad, bufs):
        """
        assumes there is only one sink pad, if the user wants 
        to resample multitple channels of data, 
        connect multiple resampler elements
        """
        self.inbufs = bufs
        for buf in bufs:
            assert buf.sample_rate == self.inrate, (
                f"data sample rate: {buf.sample_rate}"
                f" does not match resampler sample rate: {self.inrate}"
            )
            self.audioadapter.push(buf)
            if self.next_out_offset is None:
                # start offset counter with the offset of the very first buffer
                self.next_out_offset = buf.offset

            if self.offset_ref_t0 is None:
                # start offset counter with the offset of the very first buffer
                self.offset_ref_t0 = buf.offset_ref_t0

    def zeros_buffer(self, shape):
        return np.zeros(shape)

    def pad_func(self, inputs_padded):
        npad = [(0, 0)] * inputs_padded.ndim
        npad[-1] = (self.pad_length, 0)
        return np.pad(inputs_padded, npad, "constant")

    def get_output_length(self):
        """
        Needs half_length of data on each side, will use all the samples available 
        in the audioadapter
        """
        # Pretend that we have a half_length set of samples if we are at a discont
        pretend_samps = self.pad_length
        numinsamps = self.audioadapter.size + pretend_samps
        nout = int((numinsamps - self.kernel_length + 1) * self.factor)
        if nout < 0:
            nout = 0

        return numinsamps, nout

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

    def resample(self, data0, output_shape):
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

        out = out.reshape(output_shape)

        return out

    def transform(self, pad):
        """
        Up/down samples buffers
        """
        inbufs = self.inbufs

        EOS = inbufs.EOS
        #metadata = inbufs.metadata
        # if metadata is None:
        metadata = {}
        metadata["cnt:%s" % inbufs.metadata["name"]] = inbufs.metadata["cnt"]
        metadata["cnt"] = inbufs.metadata["cnt"]
        metadata["name"] = "%s -> '%s'" % (
            inbufs.metadata["name"],
            pad.name,
        )

        #if inbuf.duration == 0:
        #    inbuf.metadata = metadata
        #    inbuf.EOS = EOS
        #    return inbuf

        # if inbuf.data.dim() == 1:
        #    inbuf.data = inbuf.data.unsqueeze(0)

        A = self.audioadapter
        channels = A.channels
        out_offset = self.next_out_offset

        # If it's the first data segment, pad with zeros in front.
        sampsin, output_length = self.get_output_length()

        if output_length == 0:
            # TODO: consider more general cases
            return TSFrame(buffers=[SeriesBuffer(
                offset=out_offset,
                noffset=0,
                offset_ref_t0=self.offset_ref_t0,
                data=None,
                is_gap=True,
            )], metadata=metadata, EOS=EOS)



        noffset = int(output_length * OFFSET_RATE / self.outrate)

        # shift the next output buffer's offset starting point
        self.next_out_offset += noffset

        asize = A.size
        if A.is_gap() is True:
            # Produce a single gap buffer
            data = self.zeros_buffer(channels + (output_length,))
            flush_nsamples = asize - self.half_length * 2
            self.pad_length = -min(0, flush_nsamples)
            A.flush_samples(flush_nsamples)

            return TSFrame(buffers=[SeriesBuffer(
                offset=out_offset,
                noffset=noffset,
                offset_ref_t0=self.offset_ref_t0,
                data=data,
                is_gap=True,
            )], metadata=metadata, EOS=EOS)
        else:
            inputs_padded, _, copied_nongap = A.copy_samples(asize)
            if self.pad_length > 0:
                # if we need to pad half length of zeros in front
                inputs_padded = self.pad_func(inputs_padded)

            # TODO: check what happens when buffer size is a half integer number
            flush_nsamples = asize - self.half_length * 2
            self.pad_length = -min(0, flush_nsamples)
            out = self.resample(inputs_padded, channels + (output_length,))

            # flush samples from audioadapter
            # leave some leftover samples to pad infront of next buffer
            A.flush_samples(flush_nsamples)
            outbuf = SeriesBuffer(
                offset=out_offset,
                noffset=noffset,
                offset_ref_t0=self.offset_ref_t0,
                data=out,
                is_gap=(not copied_nongap),
            )
            assert (
                outbuf.sample_rate == self.outrate
            ), f"{outbuf.sample_rate}, {self.outrate}"
            return TSFrame(buffers=[outbuf], metadata=metadata, EOS=EOS)
