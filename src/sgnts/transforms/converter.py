from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgn import validator
from sgn.base import SinkPad, SourcePad

from sgnts.base import TSCollectFrame, TSFrame, TSTransform

# Try to import torch, but don't fail if it's not available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class Converter(TSTransform):
    """Change the data type or the device of the data.

    Args:
        backend:
            str, the backend to convert the data to. Supported backends:
            ['numpy'|'torch']
        dtype:
            str, the data type to convert the data to. Supported dtypes:
            ['float32'|'float16']
        device:
            str, the device to convert the data to. Suppored devices:
            if backend = 'numpy', only supports device = 'cpu', if backend = 'torch',
            supports device = ['cpu'|'cuda'|'cuda:<GPU number>'] where <GPU number> is
            the GPU device number.
    """

    backend: str = "numpy"
    dtype: str = "float32"
    device: str = "cpu"

    def configure(self) -> None:
        if self.backend == "numpy":
            if self.device != "cpu":
                raise ValueError("Converting to numpy only supports device as cpu")
        elif self.backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is not installed. Install it with 'pip install "
                    "sgn-ts[torch]'"
                )

            if isinstance(self.dtype, str):
                if self.dtype == "float64":
                    self.dtype = torch.float64  # type: ignore[assignment]
                elif self.dtype == "float32":
                    self.dtype = torch.float32  # type: ignore[assignment]
                elif self.dtype == "float16":
                    self.dtype = torch.float16  # type: ignore[assignment]
                else:
                    raise ValueError(
                        "Supported torch data types: float64, float32, float16"
                    )
            elif isinstance(self.dtype, torch.dtype):
                pass
            else:
                raise ValueError("Unknown dtype")
        else:
            raise ValueError("Supported backends: 'numpy' or 'torch'")

        self.pad_map = {
            src_pad: self.snks[src_pad.pad_name] for src_pad in self.source_pads
        }

    @validator.pad_names_match
    def validate(self) -> None:
        pass

    def process(
        self,
        input_frames: dict[SinkPad, TSFrame],
        output_frames: dict[SourcePad, TSCollectFrame],
    ) -> None:
        """Convert data type and device."""
        # process each source pad's corresponding sink pad
        for pad in self.source_pads:
            frame = input_frames[self.pad_map[pad]]
            out: None | np.ndarray | torch.Tensor
            for buf in frame:
                if buf.is_gap:
                    out = None
                else:
                    data = buf.data
                    if self.backend == "numpy":
                        if isinstance(data, np.ndarray):
                            # numpy to numpy
                            out = data.astype(self.dtype, copy=False)
                        elif isinstance(data, torch.Tensor):
                            # torch to numpy
                            out = (
                                data.detach()
                                .cpu()
                                .numpy()
                                .astype(self.dtype, copy=False)
                            )
                        else:
                            raise ValueError("Unsupported data type")
                    else:
                        if not TORCH_AVAILABLE:
                            raise ImportError(
                                "PyTorch is not installed. Install it with 'pip "
                                "install sgn-ts[torch]'"
                            )

                        if isinstance(data, np.ndarray):
                            # numpy to torch
                            out = torch.from_numpy(data).to(self.dtype).to(self.device)
                        elif hasattr(torch, "Tensor") and isinstance(
                            data, torch.Tensor
                        ):
                            # torch to torch
                            out = data.to(self.dtype).to(self.device)
                        else:
                            raise ValueError("Unsupported data type")

                buf = buf.copy(data=out)
                output_frames[pad].append(buf)
