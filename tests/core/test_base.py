"""Unittests for the sgnts.base.base module

Note:
    As of 20250315 this module only covers the missing coverage exposed by the
    build suite.
"""

import numpy
import pytest

from sgnts.base import Offset, SeriesBuffer, TSFrame, Time
from sgnts.base.base import (
    AdapterConfig,
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
        assert ac.align_to is None

    def test_init_with_alignment(self):
        """Test creating an AdapterConfig with alignment parameters"""
        align_boundary = Offset.fromsec(1)
        ac = AdapterConfig(align_to=align_boundary)
        assert isinstance(ac, AdapterConfig)
        assert ac.align_to == align_boundary

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

    def test_valid_buffer_with_shape(self):
        """Test the valid_buffer method with non-empty buffer (shape != 0)"""
        # Create an AdapterConfig with overlap
        ac = AdapterConfig(
            overlap=(Offset.fromsec(1), Offset.fromsec(2)), stride=Offset.fromsec(1)
        )

        # Create a buffer with the expected shape based on overlap and stride
        # expected_shape = overlap[0] samples + overlap[1] samples + stride samples
        sample_rate = 16  # 16 Hz (allowed rate - power of 2)
        overlap0_samples = Offset.tosamples(ac.overlap[0], sample_rate)  # 16 samples
        overlap1_samples = Offset.tosamples(ac.overlap[1], sample_rate)  # 32 samples
        stride_samples = Offset.sample_stride(sample_rate)  # 1024 samples at 16Hz
        expected_shape = (
            overlap0_samples + overlap1_samples + stride_samples
        )  # 1072 samples

        # Create input buffer with the expected shape
        inbuf = SeriesBuffer(
            offset=Offset.fromsec(0),
            sample_rate=sample_rate,
            shape=(expected_shape,),
            data=numpy.zeros(expected_shape),
        )

        # Test valid_buffer
        # The valid_buffer will create a new buffer with the non-overlapping portion
        # So the new shape will be smaller than the input shape
        outbuf = ac.valid_buffer(inbuf, data=0)  # Use 0 to create zeros array

        # Verify the output buffer
        assert isinstance(outbuf, SeriesBuffer)
        assert outbuf.slice == TSSlice(
            inbuf.slice[0] + ac.overlap[0], inbuf.slice[1] - ac.overlap[1]
        )
        # The new buffer should have removed the overlap samples
        # New shape = original shape - overlap[0] samples - overlap[1] samples
        expected_output_shape = expected_shape - overlap0_samples - overlap1_samples
        assert outbuf.shape == (expected_output_shape,)
        assert outbuf.data is not None

    def test_valid_buffer_with_wrong_shape(self):
        """Test the valid_buffer method with wrong shape - should raise assertion"""
        # Create an AdapterConfig with overlap
        ac = AdapterConfig(
            overlap=(Offset.fromsec(1), Offset.fromsec(2)), stride=Offset.fromsec(1)
        )

        # Create a buffer with wrong shape
        sample_rate = 16  # Use allowed rate (power of 2)
        wrong_shape = 50  # This is not the expected shape

        inbuf = SeriesBuffer(
            offset=Offset.fromsec(0),
            sample_rate=sample_rate,
            shape=(wrong_shape,),
            data=numpy.zeros(wrong_shape),
        )

        # This should raise AssertionError because the shape doesn't match expected
        with pytest.raises(AssertionError):
            ac.valid_buffer(inbuf)


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
        assert not ts.is_aligned
        ts._align()
        assert ts.is_aligned

    def test_pull_unaligned_pad(self):
        """Test pull method with unaligned pad"""
        ts = TSTransform(
            sink_pad_names=["aligned", "unaligned"],
            source_pad_names=["out"],
            unaligned=["unaligned"],
        )

        # Create a frame for the unaligned pad
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=1,
                    shape=(10,),
                    data=numpy.arange(10),
                )
            ]
        )

        # Pull to the unaligned pad - should store in unaligned_data
        unaligned_pad = ts.snks["unaligned"]
        ts.pull(pad=unaligned_pad, frame=frame)

        # Verify the frame was stored
        assert ts.unaligned_data[unaligned_pad] is not None
        assert ts.unaligned_data[unaligned_pad] == frame

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

    def test_init_with_adapter_config_alignment(self):
        """Test TSTransform initialization with adapter_config that has alignment"""
        one_second = Offset.fromsec(1)
        config = AdapterConfig(
            stride=one_second,
            overlap=(0, 0),
            align_to=one_second,
        )

        ts = TSTransform(
            sink_pad_names=["test"],
            source_pad_names=["test"],
            adapter_config=config,
        )

        assert ts.adapter_config is not None
        assert ts.adapter_config.align_to == one_second
        assert ts.stride == one_second
        assert ts.audioadapters is not None
        assert len(ts.aligned_sink_pads) == 1
        assert "test" in ts.aligned_sink_pads[0].name

    def test_init_with_unaligned_pads(self):
        """Test TSTransform initialization with unaligned pads"""
        ts = TSTransform(
            sink_pad_names=["aligned", "unaligned"],
            source_pad_names=["out"],
            unaligned=["unaligned"],
        )

        assert len(ts.unaligned_sink_pads) == 1
        assert "unaligned" in ts.unaligned_sink_pads[0].name
        assert len(ts.aligned_sink_pads) == 1
        assert "aligned" in ts.aligned_sink_pads[0].name
        assert ts.unaligned_sink_pads[0] in ts.unaligned_data


class DummyTSSource(TSSource):
    """Concrete test implementation of TSSource for testing purposes"""

    def new(self, pad):
        """Simple implementation that returns an empty frame for testing"""
        frame = self.prepare_frame(pad)
        return frame


class Test_TSSource:
    """Test group for the _TSSource class. Similar to the _TSTransSink class,
    we use the TSSource class to test the _TSSource class, since it
    is not actually instantiable.
    """

    @pytest.fixture(autouse=True)
    def src(self):
        """Test creating an instance of the TSSource class"""
        src = DummyTSSource(
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
        src = DummyTSSource(
            t0=0,
            duration=Offset.fromsamples(100, sample_rate=1),
            source_pad_names=["O1"],
        )
        assert isinstance(src, TSSource)

    def test_init_err_t0_none(self):
        """Test creating an instance of the TSSource class with t0=None"""
        with pytest.raises(ValueError):
            DummyTSSource(
                t0=None,
                duration=Offset.fromsamples(100, sample_rate=1),
                source_pad_names=["O1"],
            )

    def test_init_err_end_and_duation(self):
        """Test creating an instance of the TSSource class with t0=None"""
        with pytest.raises(ValueError):
            DummyTSSource(
                t0=0,
                end=1,
                duration=1,
                source_pad_names=["O1"],
            )

    def test_end_offset_inf(self):
        """Test the end_offset method with end=None"""
        # This seems unlikely / unintended since the end attribute is always not None
        # by the end of the __post_init__ method, but we're aiming for coverage
        src = DummyTSSource(
            t0=0,
            end=float("inf"),
            source_pad_names=["O1"],
        )

        # Manually reset the end attribute to None
        src.end = None
        assert src.end_offset == float("inf")
