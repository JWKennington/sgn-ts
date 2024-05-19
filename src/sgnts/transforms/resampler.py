from . import *
from .. base import *
from .. base.audioadapter import Audioadapter
import numpy as np
from scipy.signal import correlate

@dataclass
class Resample(TransformElement):
    """
    Up/down samples time-series data
    """
    inrate: int = None
    outrate: int = None

    def __post_init__(self):
        self.inbuf = {}

        self.PI = np.pi
        self.sin = np.sin
        self.norm = np.linalg.norm

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

        super().__post_init__()

    def get_buffer(self, pad, buf):
        self.inbuf[pad] = buf

    def zeros(self, shape):
        return np.zeros(shape)

    def zeros_buffer(self, shape):
        return np.zeros(shape)

    def pad_func(self, inputs_padded):
        npad = [(0, 0)] * inputs_padded.ndim
        npad[-1] = (self.pad_length, 0)
        return np.pad(inputs_padded, npad, "constant")


    def get_output_length(self, samps: int, factor: float):
        """
        Needs half_length of data on each side
        """
        # Pretend that we have a half_length set of samples if we are at a discont
        pretend_samps = self.pad_length
        numinsamps = self.audioadapter.size + pretend_samps
        nout = int((numinsamps - self.kernel_length + 1) * factor)
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

        vecs = self.zeros(shape=(kernel_length,))
        for i in range(kernel_length):
            x = int(i - c)
            if x == 0:
                vecs[i] = 1
            else:
                x1 = self.PI * x * factor
                x2 = self.PI * x / c
                vecs[i] = self.sin(x1) / (x1) * self.sin(x2) / (x2)

        norm = self.norm(vecs) / factor**0.5

        vecs = vecs / norm

        return vecs.reshape(1,-1)

    def resample(self, data0, output_length, channels):
        data = data0.reshape(-1, data0.shape[-1])

        if self.factor > 1:
            # upsample
            pass
        else:
            # FIXME: implement a strided correlation, rather than doing unnecessary calculations
            out = correlate(data, self.thiskernel, mode='valid')[...,::int(1/self.factor)]

        out = out.reshape(channels+(output_length,))

        return out


    def transform_buffer(self, pad):
        """
        The transform buffer just update the name to show the graph history.
        Useful for proving it works.  "EOS" is set if any input buffers are at EOS.
        """
        inbuf = self.inbuf[self.sink_pads[0]]

        EOS = any(b.EOS for b in self.inbuf.values())
        metadata = {"cnt:%s" % b.metadata['name']:b.metadata['cnt'] for b in self.inbuf.values()}
        metadata["name"] = "%s -> '%s'" % ("+".join(b.metadata["name"] for b in self.inbuf.values()), pad.name)

        if inbuf.duration == 0:
            inbuf.metadata = metadata
            inbuf.EOS = EOS
            return inbuf

        assert inbuf.sample_rate == self.inrate, (
                f"data sample rate: {inbuf.sample_rate}"
                f" does not match resampler sample rate: {self.inrate}"
        )

        #if inbuf.data.dim() == 1:
        #    inbuf.data = inbuf.data.unsqueeze(0)

        audioadapter = self.audioadapter
        audioadapter.push(inbuf)

        channels = inbuf.data.shape[:-1]

        if self.next_out_offset is None:
            self.next_out_offset = inbuf.offset

        #metadata = inbuf.metadata
        out_offset = self.next_out_offset
        offset_ref_t0 = inbuf.offset_ref_t0

        # If it's the first data segment, pad with zeros in front.
        sampsin, output_length = self.get_output_length(inbuf.size, self.factor)

        if output_length == 0:
            # TODO: consider more general cases
            return SeriesBuffer(
                offset = out_offset, noffset =0, offset_ref_t0=offset_ref_t0, data=None, metadata=metadata, is_gap=True, EOS=EOS
            )

        noffset = int(output_length * OFFSET_RATE / self.outrate)
        self.next_out_offset += noffset
        asize = audioadapter.size
        if audioadapter.is_gap() is True:
            # Produce a single gap buffer
            data = self.zeros_buffer(channels + (output_length,))
            flush_nsample = asize - self.half_length * 2
            self.pad_length = -min(0, flush_nsamples)
            audioadapter.flush_samples(flush_nsamples)

            return SeriesBuffer(
                offset=out_offset, noffset=noffset, offset_ref_t0=offset_ref_t0, data=data, metadata=metadata, is_gap=True, EOS=EOS
            )
        else:
            inputs_padded, _, copied_nongap = audioadapter.copy_samples(asize)
            if self.pad_length > 0:
                # if we need to pad half length of zeros in front
                inputs_padded = self.pad_func(inputs_padded)

            # TODO: check what happens when buffer size is a half integer number
            flush_nsamples = asize - self.half_length * 2
            self.pad_length = -min(0, flush_nsamples)
            out = self.resample(inputs_padded, output_length, channels)

            # flush samples from audioadapter 
            # leave some leftover samples to pad infront of next buffer
            audioadapter.flush_samples(flush_nsamples)
            outbuf = SeriesBuffer(offset=out_offset, noffset=noffset, offset_ref_t0=offset_ref_t0, data=out, metadata=metadata, is_gap = (not copied_nongap), EOS=EOS)
            assert (
                outbuf.sample_rate == self.outrate
            ), f"{outbuf.sample_rate}, {self.outrate}"
            return outbuf

transforms_registry += ("Resample",)
