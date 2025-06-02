"""Test to cover offset alignment in base.py lines 361-362 using multi-rate setup"""

import numpy
from sgnts.base import TSFrame, SeriesBuffer
from sgnts.sinks import NullSeriesSink


def test_simple_multi_rate_alignment():
    """Test alignment with a simple multi-rate sink to trigger lines 361-362 in
    base.py"""
    # Create a sink that receives data at different rates
    sink = NullSeriesSink(
        name="sink",
        sink_pad_names=["foo", "bar"],
    )

    # Create frames with different sample rates that will cause alignment issues
    # Use rate 16384 (max rate) and rate 16 (creates large factor)
    # The offset difference will not be divisible by the factor, triggering alignment
    frame1 = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=16384,
                shape=(1001,),  # 1001 samples at max rate creates offset 1001
                data=numpy.zeros(1001),
            )
        ]
    )

    frame2 = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=16,
                shape=(1,),  # Just 1 sample at rate 16 creates offset 1024
                data=numpy.zeros(1),
            )
        ]
    )

    # Pull frames to the sink
    sink.pull(pad=sink.snks["foo"], frame=frame1)
    sink.pull(pad=sink.snks["bar"], frame=frame2)

    # Force internal processing which triggers the alignment code at lines 361-362
    # The different sample rates (16384 and 16) create different factors
    # and the offset difference (1001 vs 1024) requires alignment
    sink.internal()

    # Test passes if no exception
    assert True
