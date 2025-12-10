#!/usr/bin/env python3
import pytest

from sgn.apps import Pipeline
from sgn.sinks import NullSink

from sgnts.base import TSTransform
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Converter

torch = pytest.importorskip("torch")


class FakeArray:
    """An object that looks like an array but isn't numpy or torch"""

    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape) if isinstance(shape, tuple) else 1


class BreakData(TSTransform):
    """Test helper that breaks data by setting it to an unsupported type."""

    def configure(self):
        self.adapter_config.enable = True

    def internal(self) -> None:
        super().internal()
        _, input_frame = self.next_input()
        _, output_frame = self.next_output()
        for buf in input_frame:
            new_buf = buf.copy(data=FakeArray(buf.shape))
            output_frame.append(new_buf)


def test_invalid_converter():
    with pytest.raises(ValueError):
        Converter(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            device="fpga",
        )
    with pytest.raises(ValueError):
        Converter(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            device="cpu",
            backend="blah",
        )
    Converter(
        name="trans1",
        source_pad_names=("H1",),
        sink_pad_names=("H1",),
        device="cpu",
        backend="torch",
        dtype="float64",
    )
    Converter(
        name="trans1",
        source_pad_names=("H1",),
        sink_pad_names=("H1",),
        device="cpu",
        backend="torch",
        dtype="float32",
    )
    Converter(
        name="trans1",
        source_pad_names=("H1",),
        sink_pad_names=("H1",),
        device="cpu",
        backend="torch",
        dtype="float16",
    )
    Converter(
        name="trans1",
        source_pad_names=("H1",),
        sink_pad_names=("H1",),
        device="cpu",
        backend="torch",
        dtype=torch.float16,
    )
    with pytest.raises(ValueError):
        Converter(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            device="cpu",
            backend="torch",
            dtype="blah",
        )
    with pytest.raises(ValueError):
        Converter(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            device="cpu",
            backend="torch",
            dtype=None,
        )


def test_broken_converter_2():
    pipeline = Pipeline()

    inrate = 256

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            signal_type="sin",
            fsin=3,
            ngap=2,
            end=8,
        ),
        BreakData(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
        ),
        Converter(
            name="trans2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
        ),
        NullSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "trans2:snk:H1": "trans1:src:H1",
            "snk1:snk:H1": "trans2:src:H1",
        },
    )
    with pytest.raises(ValueError):
        pipeline.run()


def test_broken_converter_1():
    pipeline = Pipeline()

    inrate = 256

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            signal_type="sin",
            fsin=3,
            ngap=2,
            end=8,
        ),
        BreakData(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
        ),
        Converter(
            name="trans2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            backend="torch",
        ),
        NullSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "trans2:snk:H1": "trans1:src:H1",
            "snk1:snk:H1": "trans2:src:H1",
        },
    )
    with pytest.raises(ValueError):
        pipeline.run()


def test_converter():

    pipeline = Pipeline()

    inrate = 256

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            signal_type="sin",
            fsin=3,
            ngap=2,
            end=8,
        ),
        Converter(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
        ),
        Converter(
            name="trans2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            backend="torch",
        ),
        Converter(
            name="trans3",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            backend="torch",
        ),
        Converter(
            name="trans4",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
        ),
        NullSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "trans2:snk:H1": "trans1:src:H1",
            "trans3:snk:H1": "trans2:src:H1",
            "trans4:snk:H1": "trans3:src:H1",
            "snk1:snk:H1": "trans4:src:H1",
        },
    )

    pipeline.run()


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_converter_unsupported_data_type(backend):
    """Test that converter raises ValueError for unsupported data types"""

    class FakeArray:
        """An object that looks like an array but isn't numpy or torch"""

        def __init__(self, shape):
            self.shape = shape
            self.ndim = len(shape)

    class BreakData(TSTransform):
        def new(self, pad):
            frame = self.preparedframes[self.sink_pads[0]]
            for buf in frame:
                if not buf.is_gap:
                    fake = FakeArray(buf.shape)
                    buf.data = fake
            return frame

    pipeline = Pipeline()

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=256,
            signal_type="sin",
            ngap=0,  # No gaps so we get data
            end=2,
        ),
        BreakData(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
        ),
        Converter(
            name="trans2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            backend=backend,
        ),
        NullSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "trans2:snk:H1": "trans1:src:H1",
            "snk1:snk:H1": "trans2:src:H1",
        },
    )
    with pytest.raises(ValueError, match="Unsupported data type"):
        pipeline.run()


if __name__ == "__main__":
    test_converter()
