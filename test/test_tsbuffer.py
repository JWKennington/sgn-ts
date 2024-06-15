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
            rate=2048,
            num_samples=2048,
            signal_type="white",
            random_seed=1234
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
1 -> src1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=0, offset_end=16384, samples=2048, duration=1000000000, data=[0.19151945 ... 0.72562624])
2 -> src1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=16384, offset_end=32768, samples=2048, duration=1000000000, data=[0.5880075  ... 0.88689441])
3 -> src1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=32768, offset_end=49152, samples=2048, duration=1000000000, data=[0.93690654 ... 0.04236331])
""".strip()
        )


if __name__ == "__main__":
    test_tsgraph(None)
