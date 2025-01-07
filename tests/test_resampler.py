#!/usr/bin/env python3
import pytest
from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Resampler, Converter
from sgnts.base.array_ops import TorchBackend
from sgnts.base import AdapterConfig, TSTransform


def test_valid_resampler():
    with pytest.raises(ValueError):
        Resampler(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=1,
            outrate=1,
        )
    Resampler(
        name="trans1",
        source_pad_names=("H1",),
        sink_pad_names=("H1",),
        inrate=2,
        outrate=1,
        backend=TorchBackend,
    )
    Resampler(
        name="trans1",
        source_pad_names=("H1",),
        sink_pad_names=("H1",),
        inrate=1,
        outrate=2,
        backend=TorchBackend,
    )
    Resampler(
        name="trans1",
        source_pad_names=("H1",),
        sink_pad_names=("H1",),
        inrate=1,
        outrate=2,
        adapter_config=AdapterConfig,
    )


def test_torch_resampler():

    pipeline = Pipeline()

    #
    #   ------------
    #  | src1       |
    #   ------------
    #     H1 | SR1
    #   ------------
    #  | GapFirst   |
    #   ------------
    #     H1 | SR1
    #   ------------
    #  | Converter  |
    #   ------------
    #     H1 | SR1
    #   ------------
    #  | Resampler  |
    #   ------------
    #     H1 | SR2
    #   ------------
    #  | Resampler  |
    #   ------------
    #    H1 | SR1
    #   ------------
    #  | snk1       |
    #   ------------

    inrate = 256
    outrate = 64

    class GapFirstData(TSTransform):
        cnt = 0

        def new(self, pad):
            self.cnt += 1
            if self.cnt < 5:
                for buf in self.preparedframes[self.sink_pads[0]]:
                    buf.data = None
            return self.preparedframes[self.sink_pads[0]]

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            rate=inrate,
            signal_type="sin",
            fsin=3,
            ngap=2,
            end=16,
        ),
        GapFirstData(
            name="gap",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
        ),
        Converter(
            name="conv",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            backend="torch",
        ),
        Resampler(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=inrate,
            outrate=outrate,
            backend=TorchBackend,
        ),
        Resampler(
            name="trans2",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=outrate,
            outrate=inrate,
            backend=TorchBackend,
        ),
        DumpSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
            fname="out.txt",
            verbose=True,
        ),
        link_map={
            "gap:snk:H1": "src1:src:H1",
            "conv:snk:H1": "gap:src:H1",
            "trans1:snk:H1": "conv:src:H1",
            "trans2:snk:H1": "trans1:src:H1",
            "snk1:snk:H1": "trans2:src:H1",
        },
    )

    pipeline.run()


def test_resampler():

    pipeline = Pipeline()

    #
    #       ----------   H1   -------
    #      | src1     | ---- | snk2  |
    #       ----------   SR1  -------
    #              \
    #           H1  \ SR2
    #           ------------
    #          | Resampler  |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

    inrate = 256
    outrate = 64

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
        Resampler(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=inrate,
            outrate=outrate,
        ),
        DumpSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
            fname="out.txt",
            verbose=True,
        ),
        DumpSeriesSink(
            name="snk2",
            sink_pad_names=("H1",),
            fname="in.txt",
            verbose=True,
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "snk1:snk:H1": "trans1:src:H1",
            "snk2:snk:H1": "src1:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_resampler()
