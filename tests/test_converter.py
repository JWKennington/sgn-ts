#!/usr/bin/env python3
import pytest
import torch

from sgn.apps import Pipeline
from sgn.sinks import NullSink

from sgnts.base import TSTransform
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Converter


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

    class BreakData(TSTransform):
        def new(self, pad):
            for buf in self.preparedframes[self.sink_pads[0]]:
                buf.data = "blah"
            return self.preparedframes[self.sink_pads[0]]

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

    class BreakData(TSTransform):
        def new(self, pad):
            for buf in self.preparedframes[self.sink_pads[0]]:
                buf.data = "blah"
            return self.preparedframes[self.sink_pads[0]]

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


if __name__ == "__main__":
    test_converter()
