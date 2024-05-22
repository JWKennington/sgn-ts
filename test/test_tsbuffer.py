#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSrc


def test_tsgraph(capsys):

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \
    #           ------------
    #          | snk1      |
    #           ------------

    pipeline.insert(
        FakeSeriesSrc(
            name="src1",
            source_pad_names=("H1",),
            num_buffers=2,
            shape=(2048,),
            duration=1,
            signal_type="white",
        ),
        FakeSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
        ),
        link_map={"snk1:sink:H1": "src1:src:H1"},
    )

    pipeline.run()
    if capsys is not None:
        captured = capsys.readouterr()
        assert (
            captured.out.strip()
            == """
buffer flow:  'src1:src:H1' -> 'snk1:sink:H1' offset 0 time 0 shape (2048,)
buffer flow:  'src1:src:H1' -> 'snk1:sink:H1' offset 16384 time 1000000000 shape (2048,)
buffer flow:  'src1:src:H1' -> 'snk1:sink:H1' offset 32768 time 2000000000 shape (2048,)
""".strip()
        )


if __name__ == "__main__":
    test_tsgraph(None)
