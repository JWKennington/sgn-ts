#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink
from sgnts.sources import FakeSeriesSrc
from sgnts.transforms import Resampler


def test_resampler(capsys):

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
    duration = 1

    pipeline.insert(
        FakeSeriesSrc(
            name="src1",
            source_pad_names=("H1",),
            num_buffers=2,
            shape=(int(inrate * duration),),
            duration=duration,
            signal_type="sin",
            fsin=3,
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
        ),
        DumpSeriesSink(
            name="snk2",
            sink_pad_names=("H1",),
            fname="in.txt",
        ),
        link_map={
            "trans1:sink:H1": "src1:src:H1",
            "snk1:sink:H1": "trans1:src:H1",
            "snk2:sink:H1": "src1:src:H1",
        },
    )

    pipeline.run()
    if capsys is not None:
        captured = capsys.readouterr()
        assert (
            captured.out.strip()
            == """
buffer flow:  'src1:src:H1' -> 'snk2:sink:H1' offset 0 time 0 shape (256,)
buffer flow:  'src1:src:H1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 0 time 0 shape (32,)
buffer flow:  'src1:src:H1' -> 'snk2:sink:H1' offset 16384 time 1000000000 shape (256,)
buffer flow:  'src1:src:H1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 8192 time 500000000 shape (64,)
buffer flow:  'src1:src:H1' -> 'snk2:sink:H1' offset 32768 time 2000000000 shape (256,)
buffer flow:  'src1:src:H1' -> 'trans1:src:H1' -> 'snk1:sink:H1' offset 24576 time 1500000000 shape (64,)
""".strip()
        )


if __name__ == "__main__":
    test_resampler(None)
