from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch
from scipy.signal import correlate
from sgn.base import SourcePad
from torch.nn.functional import conv1d as Fconv1d

from sgnts.base import AdapterConfig, Offset, SeriesBuffer, TSFrame, TSTransform
from sgnts.base.array_ops import (
    Array,
    ArrayBackend,
    NumpyArray,
    NumpyBackend,
    TorchArray,
    TorchBackend,
)

UP_HALF_LENGTH = 8
DOWN_HALF_LENGTH = 32


@dataclass
class Resampler(TSTransform):
    """Up/down samples time-series data

    Args:
        inrate:
            int, sample rate of the input frames
        outrate:
            int, sample rate of the output frames
        backend:
            type[ArrayBackend], default NumpyBackend, a wrapper around array operations
    """

    inrate: int = -1
    outrate: int = -1
    backend: type[ArrayBackend] = NumpyBackend

    def __post_init__(self):
        assert self.inrate in Offset.ALLOWED_RATES
        assert self.outrate in Offset.ALLOWED_RATES
        self.next_out_offset = None

        if self.outrate < self.inrate:
            # downsample parameters
            factor = self.inrate // self.outrate
            self.half_length = int(DOWN_HALF_LENGTH * factor)
            self.kernel_length = self.half_length * 2 + 1
            self.thiskernel = self.downkernel(factor)
        elif self.outrate > self.inrate:
            # upsample parameters
            factor = self.outrate // self.inrate
            self.half_length = UP_HALF_LENGTH
            self.kernel_length = self.half_length * 2 + 1
            self.thiskernel = self.upkernel(factor)
        else:
            # same rate
            raise ValueError("Inrate {self.inrate} is the same as outrate {outrate}")

        if self.backend == TorchBackend:
            # Convert the numpy kernel to torch tensors
            if self.outrate < self.inrate:
                # downsample
                self.thiskernel = torch.from_numpy(self.thiskernel).view(1, 1, -1)
            else:
                # upsample
                sub_kernel_length = int(2 * self.half_length + 1)
                self.thiskernel = torch.tensor(self.thiskernel.copy()).view(
                    self.outrate // self.inrate, 1, sub_kernel_length
                )
            self.thiskernel = self.thiskernel.to(TorchBackend.DEVICE).to(
                TorchBackend.DTYPE
            )
            self.resample = self.resample_torch
        else:
            self.resample = self.resample_numpy

        if self.adapter_config is None:
            self.adapter_config = AdapterConfig(backend=self.backend)
        else:
            assert self.adapter_config.backend == self.backend
        self.adapter_config.overlap = (
            Offset.fromsamples(self.half_length, self.inrate),
            Offset.fromsamples(self.half_length, self.inrate),
        )
        self.adapter_config.pad_zeros_startup = True

        super().__post_init__()

        self.pad_length = self.half_length

        assert len(self.sink_pads) == 1, "only one sink_pad"
        assert len(self.source_pads) == 1, "only one source_pad"
        self.sink_pad = self.sink_pads[0]

    def downkernel(self, factor: int) -> Array:
        """Compute the kernel for downsampling. Modified from gstlal_interpolator.c

        This is a sinc windowed sinc function kernel
        The baseline kernel is defined as

        g[k] = sin(pi / f * (k-c)) / (pi / f * (k-c)) * (1 - (k-c)^2 / c / c)   k != c
        g[k] = 1                                                                k = c

        Where:

            f: downsample factor, must be power of 2, e.g., 2, 4, 8, ...
            c: defined as half the full kernel length

        You specify the half filter length at the target rate in samples,
        the kernel length is then given by:

            kernel_length = half_length_at_original_rate * 2 * f + 1


        Args:
            factor:
                int, factor = inrate/outrate

        Returns:
            Array, the downsampling kernel
        """
        kernel_length = int(2 * self.half_length + 1)

        # the domain should be the kernel_length divided by two
        c = kernel_length // 2
        x = np.arange(-c, c + 1)
        vecs = np.sinc(x / factor) * np.sinc(x / c)
        norm = np.linalg.norm(vecs) * factor**0.5
        vecs = vecs / norm

        return vecs.reshape(1, -1)

    def upkernel(self, factor: int) -> Array:
        """Compute the kernel for upsampling. Modified from gstlal_interpolator.c

        This is a sinc windowed sinc function kernel
        The baseline kernel is defined as

        $$\\begin{align}
        g(k) &= \\sin(\\pi / f * (k-c)) /
                (\\pi / f * (k-c)) * (1 - (k-c)^2 / c / c)  & k != c \\\\
        g(k) &= 1 & k = c
        \\end{align}$$

        Where:

            f: interpolation factor, must be power of 2, e.g., 2, 4, 8, ...
            c: defined as half the full kernel length

        You specify the half filter length at the original rate in samples,
        the kernel length is then given by:

            kernel_length = half_length_at_original_rate * 2 * f + 1

        Interpolation is then defined as a two step process.  First the
        input data is zero filled to bring it up to the new sample rate,
        i.e., the input data, x, is transformed to x' such that:

        x'[i] = x[i/f]	if (i%f) == 0
              = 0       if (i%f) > 0

        y[i] = sum_{k=0}^{2c+1} x'[i-k] g[k]

        Since more than half the terms in this series would be zero, the
        convolution is implemented by breaking up the kernel into f separate
        kernels each 1/f as large as the originalcalled z, i.e.,:

        z[0][k/f] = g[k*f]
        z[1][k/f] = g[k*f+1]
        ...
        z[f-1][k/f] = g[k*f + f-1]

        Now the convolution can be written as:

        y[i] = sum_{k=0}^{2c/f+1} x[i/f] z[i%f][k]

        which avoids multiplying zeros.  Note also that by construction the
        sinc function has its zeros arranged such that z[0][:] had only one
        nonzero sample at its center. Therefore the actual convolution is:

        y[i] = x[i/f]					if i%f == 0
        y[i] = sum_{k=0}^{2c/f+1} x[i/f] z[i%f][k]	otherwise


        Args:
            factor:
                int, factor = outrate/inrate

        Returns:
            Array, the upsampling kernel
        """
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

    def resample_numpy(
        self, data0: NumpyArray, outshape: tuple[int, ...]
    ) -> NumpyArray:
        """Correlate the data with the kernel.

        Args:
            data0:
                Array, the data to be up/downsampled
            outshape:
                tuple[int, ...], the shape of the output array

        Returns:
            Array, the resulting array of the up/downsamping
        """
        data = data0.reshape(-1, data0.shape[-1])

        if self.outrate > self.inrate:
            # upsample
            os = []
            for i in range(self.outrate // self.inrate):
                os.append(correlate(data, self.thiskernel[i], mode="valid"))
            out = np.vstack(os)
            out = np.moveaxis(out, -1, -2)
        else:
            # downsample
            # FIXME: implement a strided correlation, rather than doing unnecessary
            # calculations
            out = correlate(data, self.thiskernel, mode="valid")[
                ..., :: self.inrate // self.outrate
            ]
        return out.reshape(outshape)

    def resample_torch(
        self, data0: TorchArray, output_shape: tuple[int, ...]
    ) -> TorchArray:
        """Correlate the data with the kernel.

        Args:
            data0:
                TorchArray, the data to be up/downsampled
            outshape:
                tuple[int, ...], the shape of the output array

        Returns:
            TorchArray, the resulting array of the up/downsamping
        """
        # FIXME: should this be in ArrayBackend?
        # FIXME: include memeory format
        data = data0.view(-1, 1, data0.shape[-1])
        thiskernel = self.thiskernel
        assert (
            data.dtype == thiskernel.dtype
        ), f"{self.name} {data.dtype} {thiskernel.dtype}"

        if self.outrate > self.inrate:  # upsample
            out = Fconv1d(data, thiskernel)
            out = out.mT.reshape(data.shape[0], -1)
        else:  # downsample
            out = Fconv1d(data, thiskernel, stride=self.inrate // self.outrate)
            out = out.squeeze(1)

        return out.view(output_shape)

    # FIXME: wraps are not playing well with mypy.  For now ignore and hope
    # that a future version of mypy will be able to handle this
    @wraps(TSTransform.new)
    def new(self, pad: SourcePad) -> TSFrame:  # type: ignore
        frame = self.preparedframes[self.sink_pad]
        assert frame.sample_rate == self.inrate
        outoffsets = self.preparedoutoffsets[self.sink_pad]

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
