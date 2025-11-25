from sgn.apps import Pipeline
from sgnts.base import Time
from sgnts.sinks import TSFrameCollectSink
from sgnts.sources import FakeSeriesSource


def test_collect():
    pipeline = Pipeline()

    rate = 2048
    duration = 20

    source = FakeSeriesSource(
        name="source",
        source_pad_names=("src",),
        rate=rate,
        ngap=0,
        signal_type="const",
        const=17.1,
        end=duration,
    )

    collect = TSFrameCollectSink(
        name="sink",
        sink_pad_names=["snk"],
    )

    pipeline.insert(
        source,
        collect,
        link_map={
            collect.snks["snk"]: source.srcs["src"],
        },
    )
    pipeline.run()

    out_frame = collect.out_frames()["snk"]
    assert out_frame.start == 0
    assert out_frame.end == 20 * Time.SECONDS
    assert out_frame.duration == 20 * Time.SECONDS
    assert out_frame.sample_rate == rate
    assert out_frame.samples == duration * rate
    assert len(out_frame.filleddata()) == duration * rate
    assert all(out_frame.filleddata() == 17.1)
