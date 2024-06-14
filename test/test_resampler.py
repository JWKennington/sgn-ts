#!/usr/bin/env python3

from sgn.apps import Pipeline

from sgnts.sinks import DumpSeriesSink, FakeSeriesSink
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
            num_buffers=5,
            rate=inrate,
            duration=duration,
            signal_type="sin",
            fsin=3,
            ngap=2,
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
            fname="out_gap.txt"
        ),
        DumpSeriesSink(
            name="snk2",
            sink_pad_names=("H1",),
            fname="in_gap.txt"
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
-> src1:src:H1 -> trans1:sink:H1 -> snk2:sink:H1  ::
	SeriesBuffer(offset=0, noffset=16384, size=256, duration=1000000000, data=[0.         ... 0.15271153])
-> trans1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=0, noffset=8192, size=32, duration=500000000, data=[0.00452957 ... 0.99831967])
-> src1:src:H1 -> trans1:sink:H1 -> snk2:sink:H1  ::
	SeriesBuffer(offset=16384, noffset=16384, size=256, duration=1000000000, data=None)
-> trans1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=8192, noffset=16384, size=64, duration=1000000000, data=[ 1.00275514e+00 ... -1.57143307e-05])
-> src1:src:H1 -> trans1:sink:H1 -> snk2:sink:H1  ::
	SeriesBuffer(offset=32768, noffset=16384, size=256, duration=1000000000, data=[-0.2794155  ...  0.42276725])
-> trans1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=24576, noffset=16384, size=64, duration=1000000000, data=[7.40148683e-17 ... 9.25553230e-01])
-> src1:src:H1 -> trans1:sink:H1 -> snk2:sink:H1  ::
	SeriesBuffer(offset=49152, noffset=16384, size=256, duration=1000000000, data=None)
-> trans1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=40960, noffset=16384, size=64, duration=1000000000, data=[ 9.42946394e-01 ... -4.20381775e-05])
-> src1:src:H1 -> trans1:sink:H1 -> snk2:sink:H1  ::
	SeriesBuffer(offset=65536, noffset=16384, size=256, duration=1000000000, data=[-0.53657292 ...  0.65914558])
-> trans1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=57344, noffset=16384, size=64, duration=1000000000, data=[-7.40148683e-17 ...  7.79057752e-01])
-> src1:src:H1 -> trans1:sink:H1 -> snk2:sink:H1  ::
	SeriesBuffer(offset=81920, noffset=16384, size=256, duration=1000000000, data=None)
-> trans1:src:H1 -> snk1:sink:H1  ::
	SeriesBuffer(offset=73728, noffset=16384, size=64, duration=1000000000, data=[ 8.08023076e-01 ... -6.50132872e-05])
""".strip()
        )


if __name__ == "__main__":
    test_resampler(None)
