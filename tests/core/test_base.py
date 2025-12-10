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
from sgnts.base.buffer import EventFrame
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

    def test_alignment_with_parameters(self):
        """Test the alignment builder method with parameters"""
        ac = AdapterConfig()

        # Test setting stride
        result = ac.alignment(stride=Offset.fromsec(2))
        assert result is ac  # Check method chaining
        assert ac.stride == Offset.fromsec(2)

        # Test setting align_to
        align_boundary = Offset.fromsec(1)
        result = ac.alignment(align_to=align_boundary)
        assert result is ac
        assert ac.align_to == align_boundary

        # Test setting overlap
        overlap = (Offset.fromsec(0.5), Offset.fromsec(1.5))
        result = ac.alignment(overlap=overlap)
        assert result is ac
        assert ac.overlap == overlap

        # Test setting shift
        result = ac.alignment(shift=Offset.fromsec(0.1))
        assert result is ac
        assert ac.offset_shift == Offset.fromsec(0.1)

    def test_on_gap_with_parameters(self):
        """Test the on_gap builder method with parameters"""
        ac = AdapterConfig()

        # Test setting skip_gaps to True
        result = ac.on_gap(skip=True)
        assert result is ac  # Check method chaining
        assert ac.skip_gaps is True

        # Test setting skip_gaps to False
        result = ac.on_gap(skip=False)
        assert result is ac
        assert ac.skip_gaps is False


class DummyTSTransform(TSTransform):
    pass


class Test_TSTransSink:
    """Test group for the TSTransSink class
    Note, since the _TSTransSink class is not actually instantiable,
    we use the TSTransform class to test the _TSTransSink class,
    but limit the tests to the _TSTransSink class methods
    """

    @pytest.fixture(autouse=True)
    def ts(self):
        """Test creating an instance of the TSTransSink class"""
        ts = DummyTSTransform(
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
        ts = DummyTSTransform(
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
        ts = DummyTSTransform(
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            max_age=100 * Time.SECONDS,
        )
        assert isinstance(ts, TSTransform)

    def test_base_class_internal(self):
        """Test that the base class internal method can be called"""
        # Create transform with adapter disabled to test basic internal() behavior
        ts = DummyTSTransform(
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            max_age=100 * Time.SECONDS,
            adapter_config=AdapterConfig(),
        )

        # Create dummy buffer for internal() to process
        dummy_buffer = SeriesBuffer(
            offset=Offset.fromsamples(0, 256),
            sample_rate=256,
            data=numpy.zeros((1, 256)),
            shape=(1, 256),
        )

        # Push buffer to the audioadapter
        ts.inbufs[ts.sink_pads[0]].push(dummy_buffer)

        # Set up the output offset state
        ts.preparedoutoffsets = {
            "offset": Offset.fromsamples(0, 256),
            "noffset": Offset.fromsamples(256, 256),
        }

        # internal() should work without raising - subclasses override to
        # add behavior
        ts.internal()

        # Verify that outframes is not populated by the base class
        assert ts.outframes[ts.source_pads[0]] is None

    def test_next_outputs_helper(self):
        """Test the next_outputs() convenience method"""
        ts = DummyTSTransform(
            sink_pad_names=["I1"],
            source_pad_names=["O1"],
            max_age=100 * Time.SECONDS,
            adapter_config=AdapterConfig(),
        )

        # Set up the output offset state
        ts.preparedoutoffsets = {
            "offset": Offset.fromsamples(0, 256),
            "noffset": Offset.fromsamples(256, 256),
        }

        # Call next_outputs() to get output frames
        output_frames = ts.next_outputs()

        # Verify we get a dict with the source pad
        assert isinstance(output_frames, dict)
        assert ts.source_pads[0] in output_frames
        assert output_frames[ts.source_pads[0]] is not None

    def test_init_with_adapter_config_alignment(self):
        """Test TSTransform initialization with adapter_config that has alignment"""
        one_second = Offset.fromsec(1)
        config = AdapterConfig(
            stride=one_second,
            overlap=(0, 0),
            align_to=one_second,
        )

        ts = DummyTSTransform(
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
        ts = DummyTSTransform(
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


class TestUnalignedTSFrameInputs:
    """Test group for unaligned TSFrame inputs coverage"""

    def test_next_input_unaligned_tsframe(self):
        """Test next_input_inputs with unaligned TSFrame pad (lines 249-254)"""
        # Create transform with unaligned pad configured
        ts = DummyTSTransform(
            sink_pad_names=["aligned", "unaligned"],
            source_pad_names=["out"],
            unaligned=["unaligned"],
        )

        # Set the input frame type to TSFrame for the unaligned pad
        ts.input_frame_types["unaligned"] = TSFrame

        # Create a TSFrame for the unaligned pad
        unaligned_frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=1,
                    shape=(10,),
                    data=numpy.arange(10),
                )
            ]
        )

        # Store it in unaligned_data as would happen during pull()
        unaligned_pad = ts.snks["unaligned"]
        ts.unaligned_data[unaligned_pad] = unaligned_frame

        # Create dummy frame for aligned pad
        aligned_frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=1,
                    shape=(10,),
                    data=numpy.zeros(10),
                )
            ]
        )
        aligned_pad = ts.snks["aligned"]
        ts.preparedframes[aligned_pad] = aligned_frame

        # Call next_inputs() which should retrieve the unaligned TSFrame
        result = ts.next_inputs()

        # Verify the unaligned frame was returned
        assert unaligned_pad in result
        assert result[unaligned_pad] is unaligned_frame


class TestEventFrameInputs:
    """Test group for EventFrame inputs coverage"""

    def test_next_event_inputs_aligned_pad(self):
        """Test next_event_inputs with aligned EventFrame pad (line 286)"""
        # Create transform with aligned pad
        ts = DummyTSTransform(
            sink_pad_names=["event_input"],
            source_pad_names=["out"],
        )

        # Configure the pad to expect EventFrame input
        ts.input_frame_types["event_input"] = EventFrame

        # Create an EventFrame
        event_frame = EventFrame(
            offset=Offset.fromsamples(0, 256), noffset=Offset.fromsamples(256, 256)
        )

        # Store it in preparedframes (as would happen for aligned pads)
        event_pad = ts.snks["event_input"]
        ts.preparedframes[event_pad] = event_frame

        # Call next_event_inputs() which should retrieve from preparedframes
        result = ts.next_event_inputs()

        # Verify the event frame was returned
        assert event_pad in result
        assert result[event_pad] is event_frame


class TestEventFrameOutputs:
    """Test group for EventFrame outputs coverage"""

    def test_next_event_output_single(self):
        """Test next_event_output with single event output pad (lines 343-348)"""
        # Create transform with one source pad configured for events
        ts = DummyTSTransform(
            sink_pad_names=["in"],
            source_pad_names=["event_out"],
        )

        # Configure the source pad to produce EventFrame
        ts.output_frame_types["event_out"] = EventFrame

        # Set up preparedoutoffsets as would be done by internal()
        ts.preparedoutoffsets = {
            "offset": Offset.fromsamples(0, 256),
            "noffset": Offset.fromsamples(256, 256),
        }

        # Set up preparedframes (empty is fine for this test)
        ts.preparedframes = {}

        # Call next_event_output() which should create an EventFrame
        event_pad, event_frame = ts.next_event_output()

        # Verify the event frame was created
        assert event_pad == ts.srcs["event_out"]
        assert isinstance(event_frame, EventFrame)
        assert event_frame.offset == Offset.fromsamples(0, 256)
        assert event_frame.noffset == Offset.fromsamples(256, 256)

    def test_next_event_outputs_multiple(self):
        """Test next_event_outputs with multiple event output pads (lines 360-373)"""
        # Create transform with multiple source pads configured for events
        ts = DummyTSTransform(
            sink_pad_names=["in"],
            source_pad_names=["event_out1", "event_out2"],
        )

        # Configure both source pads to produce EventFrame
        ts.output_frame_types["event_out1"] = EventFrame
        ts.output_frame_types["event_out2"] = EventFrame

        # Set up preparedoutoffsets
        ts.preparedoutoffsets = {
            "offset": Offset.fromsamples(0, 256),
            "noffset": Offset.fromsamples(256, 256),
        }

        # Set up preparedframes with EOS flag
        input_frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0, sample_rate=256, shape=(256,), data=numpy.zeros(256)
                )
            ]
        )
        input_frame.EOS = True
        ts.preparedframes = {ts.snks["in"]: input_frame}

        # Call next_event_outputs() which should create EventFrames for both pads
        result = ts.next_event_outputs()

        # Verify both event frames were created
        assert len(result) == 2
        assert ts.srcs["event_out1"] in result
        assert ts.srcs["event_out2"] in result

        # Verify properties
        for pad, frame in result.items():
            assert isinstance(frame, EventFrame)
            assert frame.offset == Offset.fromsamples(0, 256)
            assert frame.noffset == Offset.fromsamples(256, 256)
            assert frame.EOS is True
            # Verify it was registered in outframes
            assert ts.outframes[pad] is frame
