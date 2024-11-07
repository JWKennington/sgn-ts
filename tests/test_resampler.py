#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink, FakeSeriesSink
from sgnts.sources import FakeSeriesSrc
from sgnts.transforms import Resampler
from sgnts.base import AdapterConfig


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
        FakeSeriesSrc(
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
            adapter_config=AdapterConfig(
                pad_zeros_startup=True,
            ),
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
            "trans1:sink:H1": "src1:src:H1",
            "snk1:sink:H1": "trans1:src:H1",
            "snk2:sink:H1": "src1:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_resampler()
