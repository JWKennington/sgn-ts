"""Test offset alignment with multi-rate data processing"""

import numpy
from sgnts.base import TSFrame, SeriesBuffer
from sgnts.sinks import NullSeriesSink


def test_simple_multi_rate_alignment():
    """Test alignment when processing data at different sample rates"""
    # Create a sink that receives data at different rates
    sink = NullSeriesSink(
        name="sink",
        sink_pad_names=["foo", "bar"],
    )

    # Create frames with different sample rates to test alignment
    frame1 = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=16384,
                shape=(1001,),
                data=numpy.zeros(1001),
            )
        ]
    )

    frame2 = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=16,
                shape=(1,),
                data=numpy.zeros(1),
            )
        ]
    )

    # Pull frames to the sink
    sink.pull(pad=sink.snks["foo"], frame=frame1)
    sink.pull(pad=sink.snks["bar"], frame=frame2)

    # Process the frames - test passes if no exception is raised
    sink.internal()
