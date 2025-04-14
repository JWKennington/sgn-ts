"""Unittests for the sgnts.base.base module

Note:
    As of 20250315 this module only covers the missing coverage exposed by the
    build suite.
"""

import numpy
import pytest

from sgnts.base import Offset, TSFrame, TSResourceSource, Time
from sgnts.base.base import (
    AdapterConfig,
    SeriesBuffer,
    TSSlice,
    TSSource,
    TSTransform,
)
from sgnts.base.numpy_backend import NumpyBackend


class TestAdapterConfig:
    """Test group for the AdapterConfig class"""

    def test_init(self):
        """Test creating an instance of the AdapterConfig class"""
        ac = AdapterConfig()
        assert isinstance(ac, AdapterConfig)
        assert ac.overlap == (0, 0)
        assert ac.stride == 0
        assert not ac.pad_zeros_startup
        assert not ac.skip_gaps
        assert ac.backend == NumpyBackend

    def test_valid_buffer_no_shape(self):
        """Test the valid_buffer method with no shape"""
        ac = AdapterConfig()
        inbuf = SeriesBuffer(
            offset=0,
            sample_rate=1,
            shape=(0,),
        )
        outbuf = ac.valid_buffer(inbuf)
        assert isinstance(outbuf, SeriesBuffer)
        assert outbuf.slice == TSSlice(0, 0)


class Test_TSTransSink:
    """Test group for the TSTransSink class
    Note, since the _TSTransSink class is not actually instantiable,
    we use the TSTransform class to test the _TSTransSink class,
    but limit the tests to the _TSTransSink class methods
    """

    @pytest.fixture(autouse=True)
    def ts(self):
        """Test creating an instance of the TSTransSink class"""
        ts = TSTransform(
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            max_age=100 * Time.SECONDS,
        )
        return ts

    def test_pull_err_timeout(self, ts):
        """Test the pull method with a timeout"""
        # Timeout occurs when difference in time between the oldest and newest
        # offsets in the .inbufs attr is greater than the max_age attr

        # First we define the frame that will trigger the error
        buf_old = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=1,
                    shape=(101,),
                    data=numpy.array(range(101)),
                )
            ]
        )

        # The error should trigger when we try to pull the new buffer
        # that contains data exceeding the max_age attr
        with pytest.raises(ValueError):
            ts.pull(pad=ts.snks["I1"], frame=buf_old)

    def test__align_slice_from_pad_no_inbufs(self, ts):
        """Test _align method in case of no inbufs"""
        # If there are no inbufs, the method should return None
        assert not ts._is_aligned
        ts._align()
        assert ts._is_aligned

    def test_latest(self, ts):
        """Test the latest property"""
        assert ts.latest == -1


class TestTSTransform:
    """Test group for the TSTransform class"""

    def test_init(self):
        """Test creating an instance of the TSTransform class"""
        ts = TSTransform(
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            max_age=100 * Time.SECONDS,
        )
        assert isinstance(ts, TSTransform)

    def test_base_class_new_err(self):
        """Test the base class new method"""
        ts = TSTransform(
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            max_age=100 * Time.SECONDS,
        )
        with pytest.raises(NotImplementedError):
            ts.new(pad=ts.srcs["O1"])


class Test_TSSource:
    """Test group for the _TSSource class. Similar to the _TSTransSink class,
    we use the TSSource class to test the _TSSource class, since it
    is not actually instantiable.
    """

    @pytest.fixture(autouse=True)
    def src(self):
        """Test creating an instance of the TSSource class"""
        src = TSSource(
            t0=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )
        return src

    def test_base_class_end_offset_err(self, src):
        """Test the base class end_offset method"""
        with pytest.raises(NotImplementedError):
            super(TSSource, src).end_offset()

    def test_base_class_start_offset_err(self, src):
        """Test the base class end_offset method"""
        with pytest.raises(NotImplementedError):
            super(TSSource, src).start_offset()

    def test_prepare_frame_latest_lt_end_offset(self, src):
        """Test case latest_offset < frame.end_offset"""
        # Create a frame that will walk the intended code path

        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=1,
                    shape=(101,),
                    data=numpy.array(range(101)),
                )
            ]
        )

        # Prepare the src object state
        src._next_frame_dict[src.srcs["O1"]] = frame

        # Get the output frame
        outframe = src.prepare_frame(
            pad=src.srcs["O1"],
            latest_offset=Offset.fromsec(100),
        )

        assert isinstance(outframe, TSFrame)

    def test_prepare_frame_end_offset_gt_src_offset(self, src):
        """Test case latest_offset < frame.end_offset"""
        # Create a frame that will walk the intended code path
        # The frame will start 5 seconds before the src ends and
        # extend 5 seconds after the src ends
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=Offset.fromsec(Offset.tosec(src.end_offset) - 5),
                    sample_rate=1,
                    shape=(101,),
                    data=numpy.array(range(101)),
                )
            ]
        )

        # Prepare the src object state
        src._next_frame_dict[src.srcs["O1"]] = frame

        # Get the output frame
        outframe = src.prepare_frame(
            pad=src.srcs["O1"],
        )

        assert isinstance(outframe, TSFrame)


class TestTSSource:
    """Test group for the TSSource class"""

    def test_init(self):
        """Test creating an instance of the TSSource class"""
        src = TSSource(
            t0=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )
        assert isinstance(src, TSSource)

    def test_init_err_t0_none(self):
        """Test creating an instance of the TSSource class with t0=None"""
        with pytest.raises(ValueError):
            TSSource(
                t0=None,
                duration=Offset.fromsamples(100, sample_rate=1),
                source_pad_names=["O1"],
            )

    def test_init_err_end_and_duation(self):
        """Test creating an instance of the TSSource class with t0=None"""
        with pytest.raises(ValueError):
            TSSource(
                t0=0,
                end=1,
                duration=1,
                source_pad_names=["O1"],
            )

    def test_end_offset_inf(self):
        """Test the end_offset method with end=None"""
        # This seems unlikely / unintended since the end attribute is always not None
        # by the end of the __post_init__ method, but we're aiming for coverage
        src = TSSource(
            t0=0,
            end=float("inf"),
            source_pad_names=["O1"],
        )

        # Manually reset the end attribute to None
        src.end = None
        assert src.end_offset == float("inf")


class TestTSResourceSource:
    """Test group for the TSResourceSource class"""

    def test_init(self):
        """Test creating an instance of the TSResourceSource class"""
        src = TSResourceSource(
            start_time=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )
        assert isinstance(src, TSResourceSource)
        # TODO see note in test_init_no_duration about name mangling
        assert src._TSResourceSource__end == 1638400

    def test_init_no_duration(self):
        """Test creating an instance of the TSResourceSource class with no duration"""
        # TODO should these concepts of "max time" be unified across TSResourceSource
        #  and TSSlice?
        # TODO do we need to be using name mangling in the __end attribute?
        src = TSResourceSource(
            start_time=0,
            source_pad_names=["O1"],
        )
        assert isinstance(src, TSResourceSource)
        assert src.duration == numpy.iinfo(numpy.int64).max
        # Note: the __end attribute is not directly accessible since it is name-mangled
        assert src._TSResourceSource__end == numpy.iinfo(numpy.int64).max

    def test_end_offset_inf(self):
        """Test the end_offset method with end=None"""
        # This seems unlikely / unintended since the __end attribute is always not None
        # by the end of the __post_init__ method, but we're aiming for coverage
        src = TSResourceSource(
            start_time=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )

        # Manually reset the end attribute to None
        src._TSResourceSource__end = None
        assert src.end_offset == float("inf")

    def test_queued_duration_no_durations(self):
        """Test the queued_duration method"""
        src = TSResourceSource(
            start_time=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )
        src.setup()
        assert src.queued_duration == 0.0

    def test_queued_duration_some_durations(self):
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

        # Put the frames in the out_queue
        src.out_queue[src.srcs["O1"]].append(frame1)
        src.out_queue[src.srcs["O1"]].append(frame2)

        # Check that the queued_duration is 20mm?
        assert src.queued_duration == 20_000_000_000

    def test_base_class_get_data_err(self):
        """Test the base class get_data method"""
        src = TSResourceSource(
            start_time=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )
        with pytest.raises(NotImplementedError):
            src.get_data()

    def test_exit_context(self):
        """Test the exit_context method"""
        src = TSResourceSource(
            start_time=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )
        assert src.thread is None
        src.setup()
        assert src.thread is not None
        src.__exit__()  # Calling exit here to make sure it calls "stop"
        assert src.thread is not None
        # TODO figure out how to check for thread
        #  staying alive

    def test_set_data_empty_buf(self):
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

    def test_set_data_no_intersection(self):
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

        # Prepare the out_queue state of src
        src.setup()
        src.out_queue[src.srcs["O1"]].append(prev_frame)

        # Get the output frame
        outframe = src.set_data(
            out_frame=inframe,
            pad=src.srcs["O1"],
        )

        # Strong check here instead of equivalence, but this is since
        # the method returns the exact object passed in
        assert outframe is inframe
