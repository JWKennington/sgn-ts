#!/usr/bin/env python3

from sgn.apps import Pipeline
from sgn.sinks import NullSink

from sgnts.sources import FakeSeriesSource
from sgnts.transforms import Amplify


def test_amplify():

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \ SR2
    #           ------------
    #          | Amplify    |
    #           ------------
    #                 \
    #             H1   \ SR2
    #             ---------
    #            | snk1    |
    #             ---------

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
        Amplify(
            name="trans1",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            factor=2,
        ),
        NullSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={
            "trans1:snk:H1": "src1:src:H1",
            "snk1:snk:H1": "trans1:src:H1",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_amplify()
