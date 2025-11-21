"""Unit tests for the buffer module"""

import pytest

from sgnts.base import NumpyBackend, Offset
from sgnts.base.buffer import (
    SeriesBuffer,
    TSFrame,
    Event,
    EventBuffer,
    EventFrame,
    TSEmptyFrame,
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

    def test_set_offsets(self):
        """Test read-only attributes"""
        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=None,
            shape=(10, 2),
        )
        frame = TSFrame(
            buffers=[buf],
        )
        with pytest.raises(AttributeError):
            frame.offset = 10
        with pytest.raises(AttributeError):
            frame.noffset = 10

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

    def test_filleddata(self):
        """Test filleddata method"""
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
        frame2 = frame.filleddata()
        assert isinstance(frame2, TSFrame)
        assert len(frame2.buffers) == 1
        assert frame2.buffers[0].shape == (20,)

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


class TestTSEmptyFrame:
    """Test group for TSEmptyFrame class"""

    def test_init_and_call(self):
        """Test TSEmptyFrame initialization and promotion to TSFrame"""
        empty_frame = TSEmptyFrame(
            offset=0, noffset=Offset.fromsamples(20, sample_rate=1024)
        )

        buf = SeriesBuffer(
            offset=0,
            sample_rate=1024,
            data=1,
            shape=(20,),
        )
        result = empty_frame([buf])
        assert isinstance(result, TSFrame)


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

    def test_set_offsets(self):
        """Test set read-only attributes"""
        buf = EventBuffer(0, 10)
        frame = EventFrame(data=[buf])
        with pytest.raises(AttributeError):
            frame.offset = 5
        with pytest.raises(AttributeError):
            frame.noffset = 5

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
