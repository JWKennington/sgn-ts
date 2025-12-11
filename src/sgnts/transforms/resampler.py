from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate
from sgn import validator

from sgnts.base import Offset, SeriesBuffer, TSCollectFrame, TSFrame, TSTransform
from sgnts.decorators import transform

# Try to import torch, but don't fail if it's not available
try:
    import torch
    from torch.nn.functional import conv1d as Fconv1d

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import fast C-based gstlal upsampling
try:
    from sgnl_cpu_interp import upsample_transposed

    GSTLAL_AVAILABLE = True
except ImportError:
    GSTLAL_AVAILABLE = False
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


@dataclass(kw_only=True)
class Resampler(TSTransform):
    """Up/down samples time-series data

    Args:
        inrate:
            int, sample rate of the input frames
        outrate:
            int, sample rate of the output frames
        backend:
            type[ArrayBackend], default NumpyBackend, a wrapper around array operations
        gstlal_norm:
            boolean: If true it will normalize consistent with SGNL
            filter matching. If false it have a slightly more accurate normalization
        use_gstlal_cpu_upsample:
            boolean: If true, use fast C-based gstlal implementation for upsampling

    """

    inrate: int
    outrate: int
    backend: type[ArrayBackend] = NumpyBackend
    gstlal_norm: bool = True
    use_gstlal_cpu_upsample: bool = False

    def configure(self) -> None:
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
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is not installed. Install it with 'pip install "
                    "sgn-ts[torch]'"
                )

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

        self.adapter_config.backend = self.backend
        self.adapter_config.overlap = (
            Offset.fromsamples(self.half_length, self.inrate),
            Offset.fromsamples(self.half_length, self.inrate),
        )
        self.adapter_config.on_startup(pad_zeros=True)

        self.pad_length = self.half_length

    @validator.one_to_one
    def validate(self) -> None:
        assert (
            self.inrate in Offset.ALLOWED_RATES
        ), f"Input rate {self.inrate} not in ALLOWED_RATES: {Offset.ALLOWED_RATES}"
        assert (
            self.outrate in Offset.ALLOWED_RATES
        ), f"Output rate {self.outrate} not in ALLOWED_RATES: {Offset.ALLOWED_RATES}"

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
        if self.gstlal_norm:
            norm = np.linalg.norm(vecs) * factor**0.5
        else:
            norm = sum(vecs)
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

    def upsample_gstlal(self, data):
        """Upsample using gstlal implementation.

        Handles both numpy arrays and torch tensors.

        Args:
            data: Input data (numpy array or torch tensor), shape (-1, n_samples)

        Returns:
            Upsampled data (same type as input), not reshaped
        """
        # Check if input is torch tensor
        is_torch = TORCH_AVAILABLE and torch.is_tensor(data)
        if is_torch:
            # Convert torch -> numpy
            torch_device = data.device
            torch_dtype = data.dtype
            data_np = data.cpu().numpy()
        else:
            data_np = data

        # Call gstlal
        factor = self.outrate // self.inrate
        out_np = upsample_transposed(
            data_np, factor=factor, half_length=self.half_length
        )

        # Convert back to torch if needed
        if is_torch:
            out = torch.from_numpy(out_np).to(torch_device).to(torch_dtype)
        else:
            out = out_np

        return out

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
            if self.use_gstlal_cpu_upsample and GSTLAL_AVAILABLE:
                # Use fast C-based gstlal implementation
                out = self.upsample_gstlal(data)
            else:
                # Fall back to scipy correlate
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
        self, data0: TorchArray, outshape: tuple[int, ...]
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
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install it with 'pip install sgn-ts[torch]'"
            )

        if self.outrate > self.inrate:  # upsample
            if self.use_gstlal_cpu_upsample and GSTLAL_AVAILABLE:
                # Use gstlal (handles torch->numpy->torch conversion)
                data = data0.view(-1, data0.shape[-1])
                out = self.upsample_gstlal(data)
                return out.view(outshape)
            else:
                # Use PyTorch conv1d
                data = data0.view(-1, 1, data0.shape[-1])
                thiskernel = self.thiskernel

                # Convert data to match kernel's dtype if necessary
                if data.dtype != thiskernel.dtype:
                    data = data.to(thiskernel.dtype)

                out = Fconv1d(data, thiskernel)
                out = out.mT.reshape(data.shape[0], -1)
                return out.view(outshape)
        else:  # downsample
            data = data0.view(-1, 1, data0.shape[-1])
            thiskernel = self.thiskernel

            # Convert data to match kernel's dtype if necessary
            if data.dtype != thiskernel.dtype:
                data = data.to(thiskernel.dtype)

            out = Fconv1d(data, thiskernel, stride=self.inrate // self.outrate)
            out = out.squeeze(1)

        return out.view(outshape)

    @transform.one_to_one
    def process(self, input_frame: TSFrame, output_frame: TSCollectFrame) -> None:
        """Resample input frame to output sample rate."""
        assert input_frame.sample_rate == self.inrate, (
            f"Frame sample rate {input_frame.sample_rate} doesn't match "
            f"resampler input rate {self.inrate}"
        )

        if input_frame.shape[-1] == 0:
            buf = SeriesBuffer(
                offset=output_frame.offset,
                sample_rate=self.outrate,
                data=None,
                shape=input_frame.shape,
            )
            output_frame.append(buf)
        else:
            for buf in input_frame:
                shape = input_frame.shape[:-1] + (
                    Offset.tosamples(output_frame.noffset, self.outrate),
                )
                if buf.is_gap:
                    data = None
                else:
                    data = self.resample(buf.data, shape)
                buf = buf.copy(
                    offset=output_frame.offset,
                    sample_rate=self.outrate,
                    data=data,
                    shape=shape,
                )
                output_frame.append(buf)
