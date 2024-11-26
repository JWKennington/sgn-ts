#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.base import AdapterConfig, Offset
from sgnts.sources import FakeSeriesSource
from sgnts.sinks import NullSeriesSink
from sgnts.transforms import Adder, Resampler


def test_adder():

    pipeline = Pipeline()
    max_age = 1000000000000

    #
    #       ----------   H1   -------------
    #      | src1     | ---- | downsample  |
    #       ----------   SR1  -------------
    #             |              |
    #             |              |
    #             |           H1 | SR2
    #             |     ------------
    #          H1 |    | upsample   |
    #         SR1 |     ------------
    #             |        |
    #             |     H1 | SR1
    #             |        |
    #             |        |
    #             -----------
    #            |   add     |
    #             -----------
    #                   |
    #                H1 | SR1
    #             -----------
    #            |   snk1    |
    #             -----------
    #

    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names=("H1",),
            end=4,
            rate=2048,
            signal_type="sin",
            sample_shape=(2,),
        ),
        Resampler(
            name="down",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=2048,
            outrate=512,
        ),
        Resampler(
            name="up",
            source_pad_names=("H1",),
            sink_pad_names=("H1",),
            inrate=512,
            outrate=2048,
        ),
        Adder(
            name="add",
            source_pad_names=("A",),
            sink_pad_names=("A", "B"),
            max_age=max_age,
            adapter_config=AdapterConfig(stride=Offset.fromsec(2)),
            addslices_map={"A": (slice(0, 2),), "B": (slice(0, 2),)},
        ),
        NullSeriesSink(name="snk1", sink_pad_names=("H1",), verbose=True),
        link_map={
            "down:snk:H1": "src1:src:H1",
            "up:snk:H1": "down:src:H1",
            "add:snk:A": "up:src:H1",
            "add:snk:B": "src1:src:H1",
            "snk1:snk:H1": "add:src:A",
        },
    )

    pipeline.run()


if __name__ == "__main__":
    test_adder()
