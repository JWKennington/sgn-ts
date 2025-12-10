"""Unit tests for the buffer module"""

import numpy

import pytest

from sgnts.base import NumpyBackend, Offset
from sgnts.base.buffer import (
    SeriesBuffer,
    TSFrame,
    Event,
    EventBuffer,
    EventFrame,
)
from sgnts.base.slice_tools import TSSlice, TSSlices


class TestSeriesBuffer:
    """Test group for series buffer"""

    def test_init(self):
        """Test that the buffer is initialized correctly"""
        buffer = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=None,
            shape=(10, 2),
        )
        assert isinstance(buffer, SeriesBuffer)

    def test_set_noffset(self):
        """Test read-only attributes"""
        buffer = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=None,
            shape=(10, 2),
        )
        with pytest.raises(AttributeError):
            buffer.noffset = 10

    def test_validation_ones(self):
        """Test case for validation: ones, e.g. data==1 and shape!=(-1,)"""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=1,
            shape=(10, 2),
        )
        assert isinstance(buf, SeriesBuffer)
        assert buf.data.shape == (10, 2)

    def test_filleddata_backend(self):
        """Test using the backend for filleddata"""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=1,
            shape=(10, 2),
        )
        data = buf.filleddata(zeros_func=None)
        assert data.shape == (10, 2)

    def test_contains_seriesbuffer(self):
        """Test contains for case item is a SeriesBuffer"""
        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=1,
            shape=(10, 2),
        )
        buf2 = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=1,
            shape=(10, 2),
        )
        assert buf1 in buf2

    def test_and_operator(self):
        """Test __and__ operator for SeriesBuffer intersection"""
        # Create two overlapping buffers - buf2 starts at sample 1 (offset 16)
        # buf1 goes from offset 0 to offset 32 (2 samples at rate 1024)
        # buf2 goes from offset 16 to offset 48 (2 samples at rate 1024,
        # starting at sample 1)
        # They should overlap from offset 16 to 32
        buf1 = SeriesBuffer(
            offset=Offset.fromsamples(0, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(2, 2),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(1, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(2, 2),
        )
        result = buf1 & buf2
        assert result is not None
        assert isinstance(result, SeriesBuffer)

    def test_and_operator_no_intersection(self):
        """Test __and__ operator when buffers don't intersect"""
        buf1 = SeriesBuffer(
            offset=Offset.fromsamples(0, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10, 2),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(20, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10, 2),
        )
        result = buf1 & buf2
        assert result is None

    def test_isfinite(self):
        """Test isfinite method for SeriesBuffer"""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=1,
            shape=(10, 2),
        )
        assert buf.isfinite()


class TestTSFrame:
    """Test group for TSFrame class"""

    def test_init(self):
        """Test that the frame is initialized correctly"""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=None,
            shape=(10, 2),
        )
        frame = TSFrame(
            buffers=[buf],
        )
        assert isinstance(frame, TSFrame)

    def test_set_offsets_with_data(self):
        """Test that offset/noffset cannot be set when providing buffers"""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=None,
            shape=(10, 2),
        )
        # Offset/noffset are mutable fields but cannot be specified when buffers exist
        with pytest.raises(
            ValueError, match="Cannot specify offset/noffset when providing buffers"
        ):
            TSFrame(buffers=[buf], offset=10, noffset=20)

    def test_backend_prop(self):
        """Test backend property"""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=None,
            shape=(10, 2),
        )
        frame = TSFrame(
            buffers=[buf],
        )
        assert frame.backend == NumpyBackend

    def test_filleddata_tarr(self):
        """Test filleddata tarr methods"""
        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(10, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        frame = TSFrame(
            buffers=[buf1, buf2],
        )
        data = frame.filleddata()
        assert data.shape == (20,)
        assert all(data == numpy.ones(20))
        assert frame.samples == 20
        tarr = frame.tarr
        assert len(tarr) == 20
        expected = numpy.arange(0, 20) / 1024
        assert all(tarr == expected)

    def test_search(self):
        """Test search method for TSFrame"""
        buf1 = SeriesBuffer(
            offset=Offset.fromsamples(0, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(10, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        frame = TSFrame(buffers=[buf1, buf2])

        search_buf = SeriesBuffer(
            offset=Offset.fromsamples(0, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(15,),
        )
        result = frame.search(search_buf)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_align(self):
        """Test align method for TSFrame with data buffers"""
        buf1 = SeriesBuffer(
            offset=Offset.fromsamples(0, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(10, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        frame = TSFrame(buffers=[buf1, buf2])

        tsslices = TSSlices(
            [
                TSSlice(
                    Offset.fromsamples(0, sample_rate=1024),
                    Offset.fromsamples(5, sample_rate=1024),
                ),
                TSSlice(
                    Offset.fromsamples(5, sample_rate=1024),
                    Offset.fromsamples(20, sample_rate=1024),
                ),
            ]
        )
        result = frame.align(tsslices)
        assert isinstance(result, TSFrame)
        assert len(result.buffers) == 2

    def test_align_with_gaps(self):
        """Test align method for TSFrame with gap buffers"""
        # Create a frame with a gap buffer
        buf1 = SeriesBuffer(
            offset=Offset.fromsamples(0, sample_rate=1024),
            sample_rate=1024,
            data=None,  # This makes it a gap
            shape=(10,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(10, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        frame = TSFrame(buffers=[buf1, buf2])

        tsslices = TSSlices(
            [
                TSSlice(
                    Offset.fromsamples(0, sample_rate=1024),
                    Offset.fromsamples(5, sample_rate=1024),
                ),
                TSSlice(
                    Offset.fromsamples(5, sample_rate=1024),
                    Offset.fromsamples(20, sample_rate=1024),
                ),
            ]
        )
        result = frame.align(tsslices)
        assert isinstance(result, TSFrame)

    def test_empty_frame_creation(self):
        """Test creating empty TSFrame with explicit offset/noffset"""
        offset = Offset.fromsamples(0, sample_rate=1024)
        noffset = Offset.fromsamples(20, sample_rate=1024)
        frame = TSFrame(offset=offset, noffset=noffset)
        assert frame.offset == offset
        assert frame.noffset == noffset
        assert frame.end_offset == offset + noffset
        assert len(frame.buffers) == 0

    def test_empty_frame_zero_offsets(self):
        """Test that offset=0, noffset=0 is valid for empty frame"""
        frame = TSFrame(offset=0, noffset=0)
        assert frame.offset == 0
        assert frame.noffset == 0
        assert len(frame.buffers) == 0

    def test_validate_span_incomplete_start(self):
        """Test validate_span fails when first buffer doesn't start at frame offset"""
        offset = Offset.fromsamples(0, sample_rate=1024)
        noffset = Offset.fromsamples(20, sample_rate=1024)
        frame = TSFrame(offset=offset, noffset=noffset)

        # Add buffer that doesn't start at frame offset (bypass append validation)
        buf = SeriesBuffer(
            offset=Offset.fromsamples(5, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(15,),
        )
        frame.buffers.append(buf)

        with pytest.raises(
            AssertionError, match="First buffer offset.*!= frame offset"
        ):
            frame.validate_span()

    def test_validate_span_gap(self):
        """Test validate_span fails when there's a gap between buffers"""
        offset = Offset.fromsamples(0, sample_rate=1024)
        noffset = Offset.fromsamples(20, sample_rate=1024)
        frame = TSFrame(offset=offset, noffset=noffset)

        # Add two buffers with a gap (bypass append validation)
        buf1 = SeriesBuffer(offset=offset, sample_rate=1024, data=1, shape=(5,))
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(10, sample_rate=1024),
            sample_rate=1024,
            data=1,
            shape=(10,),
        )
        frame.buffers.append(buf1)
        frame.buffers.append(buf2)

        with pytest.raises(AssertionError, match="Buffer offset.*must match"):
            frame.validate_span()


class TestEventBuffer:
    """Test group for EventBuffer"""

    def test_bad_init(self):
        """Test init with wrong arguments"""
        # start > end
        with pytest.raises(ValueError):
            EventBuffer.from_span(3_000_000_000, 0)

    def test_init_span(self):
        """Test calculated attributes from span constructor"""
        buf = EventBuffer.from_span(0, 10_000_000_000)
        assert buf.start == 0
        assert buf.end == 10_000_000_000

    def test_compare(self):
        """Test equality and comparison ops"""
        event = Event(0)
        buf = EventBuffer(0, 10)
        assert buf != event
        other_buf = EventBuffer(0, 3)
        assert buf != other_buf
        assert other_buf in buf

    def test_event_access(self):
        """Test event access from buffer"""
        events = [Event(0), Event(10)]
        buf = EventBuffer(0, 10, data=events)
        assert buf.events == events
        assert buf[0] == events[0]
        for event, expected in zip(buf, events):
            assert event == expected


class TestEventFrame:
    """Test group for EventFrame"""

    def test_bad_init(self):
        """Test init with wrong arguments"""
        # start > end
        buf1 = EventBuffer(0, 10)
        buf2 = EventBuffer(20, 30)
        with pytest.raises(ValueError):
            EventFrame(data=[buf2, buf1])

    def test_set_offsets_with_data(self):
        """Test that offset/noffset cannot be set when providing data"""
        buf = EventBuffer(0, 10)
        # Offset/noffset are mutable fields but cannot be specified when data exists
        with pytest.raises(
            ValueError, match="Cannot specify offset/noffset when providing data"
        ):
            EventFrame(data=[buf], offset=5, noffset=10)

    def test_compare(self):
        """Test equality and comparison ops"""
        event = Event(0)
        buf = EventBuffer(0, 10, data=[event])
        frame = EventFrame(data=[buf])
        other_frame = EventFrame(data=[EventBuffer(0, 3, data=[event])])
        assert frame != other_frame
        assert other_frame in frame

    def test_buffer_access(self):
        """Test event access from frame"""
        events = [Event(0), Event(10)]
        buf = EventBuffer(0, 10, data=events)
        frame = EventFrame(data=[buf])
        assert frame.events == events
        assert frame[0] == buf
        for thisbuf in frame:
            assert thisbuf == buf

    def test_empty_frame_creation(self):
        """Test creating empty EventFrame with explicit offset/noffset"""
        frame = EventFrame(offset=100, noffset=200)
        assert frame.offset == 100
        assert frame.noffset == 200
        assert frame.end_offset == 300
        assert len(frame.data) == 0

    def test_empty_frame_zero_offsets(self):
        """Test that offset=0, noffset=0 is valid for empty frame"""
        frame = EventFrame(offset=0, noffset=0)
        assert frame.offset == 0
        assert frame.noffset == 0
        assert len(frame.data) == 0

    def test_append_to_empty_frame(self):
        """Test appending buffers to empty EventFrame"""
        frame = EventFrame(offset=100, noffset=200)

        # Append buffer that spans part of the frame
        buf1 = EventBuffer(offset=100, noffset=50)
        frame.append(buf1)
        assert len(frame.data) == 1
        assert frame.data[0] == buf1

        # Append contiguous buffer
        buf2 = EventBuffer(offset=150, noffset=150)
        frame.append(buf2)
        assert len(frame.data) == 2

    def test_append_validation_bounds(self):
        """Test that append validates buffer falls within frame bounds"""
        frame = EventFrame(offset=100, noffset=200)

        # Buffer starts before frame
        buf_before = EventBuffer(offset=50, noffset=10)
        with pytest.raises(AssertionError, match="starts before frame offset"):
            frame.append(buf_before)

        # Buffer extends beyond frame
        buf_after = EventBuffer(offset=100, noffset=250)
        with pytest.raises(AssertionError, match="extends beyond frame"):
            frame.append(buf_after)

    def test_append_validation_contiguity(self):
        """Test that append validates buffer is contiguous with previous"""
        frame = EventFrame(offset=100, noffset=200)

        # Add first buffer
        buf1 = EventBuffer(offset=100, noffset=50)
        frame.append(buf1)

        # Try to add non-contiguous buffer (gap)
        buf2_gap = EventBuffer(offset=160, noffset=50)
        with pytest.raises(AssertionError, match="not contiguous"):
            frame.append(buf2_gap)

        # Try to add overlapping buffer
        buf2_overlap = EventBuffer(offset=140, noffset=50)
        with pytest.raises(AssertionError, match="not contiguous"):
            frame.append(buf2_overlap)

    def test_validate_span_complete(self):
        """Test validate_span with complete data"""
        frame = EventFrame(offset=100, noffset=200)

        # Add buffers that fully span the frame
        buf1 = EventBuffer(offset=100, noffset=100)
        buf2 = EventBuffer(offset=200, noffset=100)
        frame.append(buf1)
        frame.append(buf2)

        # Should pass validation
        frame.validate_span()

    def test_validate_span_incomplete_start(self):
        """Test validate_span fails when first buffer doesn't start at
        frame offset"""
        frame = EventFrame(offset=100, noffset=200)

        # Add buffer that doesn't start at frame offset
        # (impossible through append, but test anyway)
        buf = EventBuffer(offset=150, noffset=150)
        frame.data.append(buf)  # Bypass append validation

        with pytest.raises(
            AssertionError, match="First buffer offset.*!= frame offset"
        ):
            frame.validate_span()

    def test_validate_span_incomplete_end(self):
        """Test validate_span fails when last buffer doesn't reach frame end"""
        frame = EventFrame(offset=100, noffset=200)

        # Add buffer that doesn't reach the end
        buf = EventBuffer(offset=100, noffset=100)
        frame.append(buf)

        with pytest.raises(
            AssertionError, match="Last buffer end_offset.*!= frame end_offset"
        ):
            frame.validate_span()

    def test_validate_span_gap(self):
        """Test validate_span fails when there's a gap between buffers"""
        frame = EventFrame(offset=100, noffset=200)

        # Add two buffers with a gap (impossible through append, but test anyway)
        buf1 = EventBuffer(offset=100, noffset=50)
        buf2 = EventBuffer(offset=160, noffset=140)
        frame.data.append(buf1)  # Bypass append validation
        frame.data.append(buf2)

        with pytest.raises(AssertionError, match="Gap between buffer"):
            frame.validate_span()

    def test_series_buffer_copy_is_gap_true(self):
        """Test SeriesBuffer.copy() with is_gap=True parameter"""
        # Create a non-gap buffer
        buffer = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=numpy.arange(256),
            shape=(256,),
        )
        assert not buffer.is_gap

        # Replace with is_gap=True should create a gap buffer (data=None)
        new_buffer = buffer.copy(is_gap=True)
        assert new_buffer.is_gap
        assert new_buffer.data is None

    def test_series_buffer_copy_is_gap_false(self):
        """Test SeriesBuffer.copy() with is_gap=False parameter"""
        # Create a gap buffer
        gap_buffer = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=None,
            shape=(256,),
        )
        assert gap_buffer.is_gap

        # Replace with is_gap=False and new data
        new_data = numpy.zeros(256)
        new_buffer = gap_buffer.copy(is_gap=False, data=new_data)
        assert not new_buffer.is_gap
        assert new_buffer.data is new_data
