#!/usr/bin/env python3

from dataclasses import dataclass
import time
import numpy

import pytest
from sgn.apps import Pipeline
from sgn.sources import SignalEOS
from sgnts.base import TSResourceSource, TSFrame, Offset
from sgnts.base.buffer import SeriesBuffer
from sgnts.sinks import NullSeriesSink
from sgnts.utils import gpsnow


#
# NOTE this mocks e.g., an arrakis server
#
@dataclass
class DataServer:
    block_duration: int = 2
    simulate_skip_data: bool = False
    simulate_hang: int = 0
    simulate_exception: bool = False

    description = {
        "H1:FOO": {"rate": 2048, "sample-shape": ()},
        "L1:FOO": {"rate": 2048, "sample-shape": ()},
    }

    def stream(self, channels, start=None, end=None):
        assert not (set(channels) - set(self.description))
        t0 = int(gpsnow()) - 1.0 if start is None else start
        time.sleep(self.simulate_hang)
        if self.simulate_exception:
            raise ValueError("Simulated worker exception")
        while True:
            out = {}
            if end is not None and t0 >= end:
                return
            for channel in channels:
                sample_shape, rate = (
                    self.description[channel]["sample-shape"],
                    self.description[channel]["rate"],
                )
                shape = sample_shape + (self.block_duration * rate,)
                out[channel] = {
                    "t0": t0,
                    "data": numpy.random.randn(*shape),
                    "rate": rate,
                    "sample_shape": sample_shape,
                }
            t0 += self.block_duration
            # Simulate a data skip if requested
            if self.simulate_skip_data:
                t0 += 2
            # simulate real-time if start is None
            if start is None:
                time.sleep(max(0, t0 - gpsnow()))
            yield out


@dataclass
class FakeLiveSource(TSResourceSource):
    simulate_skip_data: bool = False
    block_duration: int = 4
    simulate_hang: int = 0
    simulate_exception: bool = False

    def __post_init__(self):
        super().__post_init__()

    def worker_process(
        self,
        context,
        srcs,
        start_time,
        end_time,
        block_duration,
        simulate_skip_data,
        simulate_hang,
        simulate_exception,
    ):
        """Worker process implementation for data generation."""
        # Create the DataServer in the worker context (like original MR)
        server = DataServer(
            block_duration=block_duration,
            simulate_skip_data=simulate_skip_data,
            simulate_hang=simulate_hang,
            simulate_exception=simulate_exception,
        )

        for stream in server.stream(srcs, start_time, end_time):
            # Check if we should stop
            if context.should_stop():
                break

            for channel, block in stream.items():
                pad = srcs[channel]
                buf = SeriesBuffer(
                    offset=Offset.fromsec(block["t0"]),
                    data=block["data"],
                    sample_rate=block["rate"],
                )
                # Send (pad, buffer) tuple to output queue
                context.output_queue.put((pad, buf))


def test_resource_source():

    pipeline = Pipeline()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        duration=10,
        block_duration=4,
    )
    snk = NullSeriesSink(
        name="snk",
        sink_pad_names=("H1",),
        verbose=True,
    )
    pipeline.insert(
        src,
        snk,
        link_map={snk.snks["H1"]: src.srcs["H1:FOO"]},
    )

    with SignalEOS():
        pipeline.run()


def test_resource_fail():

    pipeline = Pipeline()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:BAR",),
        duration=10,
        block_duration=4,
    )
    snk = NullSeriesSink(
        name="snk",
        sink_pad_names=("H1",),
        verbose=True,
    )
    pipeline.insert(
        src,
        snk,
        link_map={snk.snks["H1"]: src.srcs["H1:BAR"]},
    )

    with pytest.raises(RuntimeError):
        pipeline.run()


def test_resource_hang():

    pipeline = Pipeline()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        duration=10,
        block_duration=4,
        simulate_hang=2,
        in_queue_timeout=1,
    )
    snk = NullSeriesSink(
        name="snk",
        sink_pad_names=("H1",),
        verbose=True,
    )
    pipeline.insert(
        src,
        snk,
        link_map={snk.snks["H1"]: src.srcs["H1:FOO"]},
    )

    with pytest.raises(ValueError, match="Could not read from resource after"):
        pipeline.run()


def test_resource_worker_exception():
    """Test that worker exceptions are properly propagated"""

    pipeline = Pipeline()

    src = FakeLiveSource(
        name="src",
        source_pad_names=("H1:FOO",),
        duration=10,
        block_duration=4,
        simulate_exception=True,
    )
    snk = NullSeriesSink(
        name="snk",
        sink_pad_names=("H1",),
        verbose=True,
    )
    pipeline.insert(
        src,
        snk,
        link_map={snk.snks["H1"]: src.srcs["H1:FOO"]},
    )

    # Should raise RuntimeError indicating worker stopped before EOS
    with pytest.raises(RuntimeError, match="worker stopped before EOS") as exc_info:
        pipeline.run()

    # Verify the original exception is preserved in the exception chain
    # The chain might be RuntimeError -> RuntimeError -> ValueError
    current_exc = exc_info.value
    found_original = False

    # Walk through the exception chain to find the original ValueError
    while current_exc is not None:
        if isinstance(current_exc, ValueError) and "Simulated worker exception" in str(
            current_exc
        ):
            found_original = True
            break
        current_exc = current_exc.__cause__

    assert found_original, (
        f"Expected to find ValueError with 'Simulated worker"
        f"exception' in chain, final cause: {exc_info.value.__cause__}"
    )


# TSResourceSource unit tests (moved from tests/core/test_base.py)


def test_tsresourcesource_duration_properties():
    """Test duration-related public properties and behavior"""
    # Test with explicit duration - verify public properties work correctly
    src = TSResourceSource(
        start_time=0,
        duration=Offset.fromsamples(100, sample_rate=1),
        source_pad_names=["O1"],
    )
    assert src.duration == 1638400  # 100 samples at 1 Hz
    assert src.end_time == 1638400  # start_time + duration

    # Test with no duration - should use max value
    src_infinite = TSResourceSource(
        start_time=0,
        source_pad_names=["O1"],
    )
    assert src_infinite.duration == numpy.iinfo(numpy.int64).max
    assert src_infinite.end_time == numpy.iinfo(numpy.int64).max

    # Test with duration but no start_time - end_time should be None,
    # end_offset should be inf
    src_no_start = TSResourceSource(
        start_time=None,
        duration=100,
        source_pad_names=["O1"],
    )
    assert src_no_start.duration == 100
    assert src_no_start.start_time is None
    assert src_no_start.end_time is None
    assert src_no_start.end_offset == float("inf")


def test_tsresourcesource_queued_duration_no_durations():
    """Test the queued_duration method"""
    src = TSResourceSource(
        start_time=0,
        duration=Offset.fromsamples(100, sample_rate=1),
        source_pad_names=["O1"],
    )
    src.setup()
    assert src.queued_duration == 0.0


def test_tsresourcesource_queued_duration_some_durations():
    """Test the queued_duration method"""
    src = TSResourceSource(
        start_time=0,
        duration=Offset.fromsamples(100, sample_rate=1),
        source_pad_names=["O1"],
    )
    src.setup()

    # Make two frames, one with a duration of 10 and the other with a duration of 20
    frame1 = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=1,
                shape=(10,),
                data=numpy.array(range(10)),
            )
        ]
    )
    frame2 = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=1,
                shape=(20,),
                data=numpy.array(range(20)),
            )
        ]
    )

    # Put the frames in the buffer_queue (updated for new API)
    src.buffer_queue[src.srcs["O1"]].append(frame1)
    src.buffer_queue[src.srcs["O1"]].append(frame2)

    # Check that the queued_duration is 20_000_000_000 nanoseconds
    assert src.queued_duration == 20_000_000_000


def test_tsresourcesource_base_class_worker_process_err():
    """Test the base class worker_process method raises NotImplementedError"""
    src = TSResourceSource(
        start_time=0,
        duration=Offset.fromsamples(100, sample_rate=1),
        source_pad_names=["O1"],
    )
    # Create a mock context
    from sgn.subprocess import WorkerContext
    import queue

    context = WorkerContext(
        input_queue=queue.Queue(),
        output_queue=queue.Queue(),
        worker_stop=None,
        worker_shutdown=None,
        shm_list=[],
    )

    with pytest.raises(NotImplementedError):
        src.worker_process(context)


def test_tsresourcesource_set_data_empty_buf():
    """Test the set_data method with offset == end_offset"""
    src = TSResourceSource(
        start_time=0,
        duration=Offset.fromsamples(100, sample_rate=1),
        source_pad_names=["O1"],
    )
    inframe = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=1,
                shape=(0,),
                data=None,
            )
        ]
    )
    outframe = src.set_data(
        out_frame=inframe,
        pad=src.srcs["O1"],
    )
    # Strong check here instead of equivalence, but this is since
    # the method returns the exact object passed in
    assert outframe is inframe


def test_tsresourcesource_set_data_no_intersection():
    """Test the set_data method with no intersection"""
    src = TSResourceSource(
        start_time=0,
        duration=Offset.fromsamples(100, sample_rate=1),
        source_pad_names=["O1"],
    )
    inframe = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=0,
                sample_rate=1,
                shape=(10,),
                data=numpy.array(range(10)),
            )
        ]
    )
    prev_frame = TSFrame(
        buffers=[
            SeriesBuffer(
                offset=Offset.fromsamples(10, sample_rate=1),
                sample_rate=1,
                shape=(10,),
                data=numpy.array(range(10)),
            )
        ]
    )

    # Prepare the buffer_queue state of src (updated for new API)
    src.setup()
    src.buffer_queue[src.srcs["O1"]].append(prev_frame)

    # Get the output frame
    outframe = src.set_data(
        out_frame=inframe,
        pad=src.srcs["O1"],
    )

    # Strong check here instead of equivalence, but this is since
    # the method returns the exact object passed in
    assert outframe is inframe


if __name__ == "__main__":
    test_resource_source()
