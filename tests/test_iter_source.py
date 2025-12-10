"""Tests for TSIterSource."""

import numpy
from sgn.apps import Pipeline

from sgnts.base import Offset, SeriesBuffer, TSFrame
from sgnts.sinks import NullSeriesSink
from sgnts.sources import TSIterSource


def test_iter_source_with_frames():
    """Test TSIterSource with provided frames."""
    sample_rate = 256

    # Create some test frames
    buffers1 = [
        SeriesBuffer(
            offset=Offset.fromsamples(0, sample_rate),
            sample_rate=sample_rate,
            data=numpy.ones(100),
            shape=(100,),
        ),
    ]
    buffers2 = [
        SeriesBuffer(
            offset=Offset.fromsamples(100, sample_rate),
            sample_rate=sample_rate,
            data=numpy.ones(100) * 2,
            shape=(100,),
        ),
    ]

    frame1 = TSFrame(buffers=buffers1)
    frame2 = TSFrame(buffers=buffers2)

    source = TSIterSource(
        name="src",
        source_pad_names=["out"],
        frames=[frame1, frame2],
        t0=0,
    )

    sink = NullSeriesSink(name="sink", sink_pad_names=["in"])

    pipeline = Pipeline()
    pipeline.connect(source, sink)
    pipeline.run()

    # Should have received both frames plus EOS
    # The last frame should have been marked as EOS


def test_iter_source_single_frame():
    """Test TSIterSource with a single frame."""
    sample_rate = 256

    buffer = SeriesBuffer(
        offset=Offset.fromsamples(0, sample_rate),
        sample_rate=sample_rate,
        data=numpy.ones(100),
        shape=(100,),
    )

    frame = TSFrame(buffers=[buffer])

    source = TSIterSource(
        name="src",
        source_pad_names=["out"],
        frames=[frame],
        t0=0,
    )

    sink = NullSeriesSink(name="sink", sink_pad_names=["in"])

    pipeline = Pipeline()
    pipeline.connect(source, sink)
    pipeline.run()

    # Single frame should be marked as EOS
