"""Tests for TSCollectFrame context manager."""

import pytest
import numpy as np

from sgnts.base import Offset, SeriesBuffer, TSCollectFrame, TSFrame


class TestTSCollectFrameBasic:
    """Basic tests for TSCollectFrame functionality"""

    def test_fill_returns_collector(self):
        """Test that TSFrame.fill() returns a TSCollectFrame"""
        frame = TSFrame(offset=0, noffset=1000)
        collector = frame.fill()
        assert isinstance(collector, TSCollectFrame)
        assert collector.parent_frame is frame

    def test_context_manager_basic(self):
        """Test basic context manager usage"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))

        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        with frame.fill() as collector:
            collector.append(buf)

        # After context exit, frame should have buffers
        assert len(frame.buffers) == 1
        assert frame.buffers[0] == buf

    def test_manual_close(self):
        """Test manual close without context manager"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))

        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        collector = frame.fill()
        collector.append(buf)
        collector.close()

        # After close, frame should have buffers
        assert len(frame.buffers) == 1
        assert frame.buffers[0] == buf

    def test_multiple_buffers(self):
        """Test appending multiple buffers"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(256, 256),
            sample_rate=256,
            data=np.ones((256,)) * 2,
            shape=(256,),
        )

        with frame.fill() as collector:
            collector.append(buf1)
            collector.append(buf2)

        assert len(frame.buffers) == 2
        assert frame.buffers[0] == buf1
        assert frame.buffers[1] == buf2

    def test_extend_method(self):
        """Test extend with multiple buffers at once"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(256, 256),
            sample_rate=256,
            data=np.ones((256,)) * 2,
            shape=(256,),
        )

        with frame.fill() as collector:
            collector.extend([buf1, buf2])

        assert len(frame.buffers) == 2

    def test_iadd_operator(self):
        """Test += operator for appending"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))

        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        with frame.fill() as collector:
            collector += buf

        assert len(frame.buffers) == 1


class TestTSCollectFrameValidation:
    """Tests for TSCollectFrame validation"""

    def test_validates_span(self):
        """Test that close() validates buffers span the frame"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        # Only provide half the buffers
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        with pytest.raises(AssertionError, match="Last buffer end_offset"):
            with frame.fill() as collector:
                collector.append(buf)

    def test_validates_contiguity(self):
        """Test that append validates buffer contiguity"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )
        # Gap - buf2 doesn't start where buf1 ends
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(300, 256),  # Not contiguous!
            sample_rate=256,
            data=np.ones((212,)),
            shape=(212,),
        )

        with pytest.raises(AssertionError, match="not contiguous"):
            with frame.fill() as collector:
                collector.append(buf1)
                collector.append(buf2)

    def test_validates_first_buffer_offset(self):
        """Test that first buffer must start at frame offset"""
        frame = TSFrame(offset=100, noffset=Offset.fromsamples(256, 256))

        # Buffer starts at 0, not 100
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        with pytest.raises(
            AssertionError, match="Buffer offset.*starts before frame offset"
        ):
            with frame.fill() as collector:
                collector.append(buf)

    def test_validates_buffer_within_bounds(self):
        """Test that buffers must fall within frame bounds"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))

        # Buffer extends beyond frame
        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((512,)),
            shape=(512,),
        )

        with pytest.raises(AssertionError, match="extends beyond frame"):
            with frame.fill() as collector:
                collector.append(buf)

    def test_cannot_append_after_close(self):
        """Test that appending after close raises error"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))

        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        collector = frame.fill()
        collector.append(buf)
        collector.close()

        with pytest.raises(ValueError, match="Cannot append to closed"):
            collector.append(buf)

    def test_cannot_close_twice(self):
        """Test that closing twice raises error"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))

        buf = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        collector = frame.fill()
        collector.append(buf)
        collector.close()

        with pytest.raises(ValueError, match="already closed"):
            collector.close()

    def test_cannot_fill_nonempty_frame(self):
        """Test that fill() cannot be used on frame with buffers"""
        frame = TSFrame(
            buffers=[
                SeriesBuffer(
                    offset=0,
                    sample_rate=256,
                    data=np.ones((256,)),
                    shape=(256,),
                )
            ]
        )

        with pytest.raises(ValueError, match="already has buffers"):
            with frame.fill() as _:
                pass


class TestTSCollectFrameAtomic:
    """Tests for atomic all-or-nothing behavior"""

    def test_exception_prevents_commit(self):
        """Test that exception in context prevents buffer commit"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        try:
            with frame.fill() as collector:
                collector.append(buf1)
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        # Frame should still be empty
        assert len(frame.buffers) == 0

    def test_validation_failure_prevents_commit(self):
        """Test that validation failure prevents any buffers from being added"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )
        # Invalid buf2 - doesn't complete the span

        try:
            with frame.fill() as collector:
                collector.append(buf1)
                # Missing second buffer - will fail validation
        except (AssertionError, ValueError):
            pass

        # Frame should still be empty - atomic failure
        assert len(frame.buffers) == 0


class TestTSCollectFrameProperties:
    """Tests for TSCollectFrame properties and iteration"""

    def test_len_returns_buffer_count(self):
        """Test that len() returns number of collected buffers"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(256, 256),
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        collector = frame.fill()
        assert len(collector) == 0

        collector.append(buf1)
        assert len(collector) == 1

        collector.append(buf2)
        assert len(collector) == 2

    def test_iteration(self):
        """Test that collector is iterable"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(512, 256))

        buf1 = SeriesBuffer(
            offset=0,
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )
        buf2 = SeriesBuffer(
            offset=Offset.fromsamples(256, 256),
            sample_rate=256,
            data=np.ones((256,)),
            shape=(256,),
        )

        collector = frame.fill()
        collector.append(buf1)
        collector.append(buf2)

        buffers = list(collector)
        assert len(buffers) == 2
        assert buffers[0] == buf1
        assert buffers[1] == buf2

    def test_inherits_frame_properties(self):
        """Test that collector inherits offset/noffset from parent"""
        frame = TSFrame(offset=1000, noffset=2000, EOS=True, metadata={"test": "value"})

        collector = frame.fill()
        assert collector.offset == 1000
        assert collector.noffset == 2000
        assert collector.EOS is True
        assert collector.metadata == {"test": "value"}


class TestTSFrameEnsureNonempty:
    """Tests for @ensure_nonempty decorator on TSFrame properties"""

    def test_empty_frame_raises_on_shape(self):
        """Test that accessing shape on empty TSFrame raises ValueError"""
        frame = TSFrame(offset=0, noffset=1000)
        # Frame is empty, no buffers
        with pytest.raises(
            ValueError,
            match="TSFrame.shape cannot be used when there are no buffers",
        ):
            _ = frame.shape

    def test_empty_frame_raises_on_sample_rate(self):
        """Test that accessing sample_rate on empty TSFrame raises ValueError"""
        frame = TSFrame(offset=0, noffset=1000)
        with pytest.raises(
            ValueError,
            match="TSFrame.sample_rate cannot be used when there are no buffers",
        ):
            _ = frame.sample_rate

    def test_empty_frame_raises_on_heartbeat(self):
        """Test that calling heartbeat() on empty TSFrame raises ValueError"""
        frame = TSFrame(offset=0, noffset=1000)
        with pytest.raises(
            ValueError,
            match="TSFrame.heartbeat cannot be used when there are no buffers",
        ):
            frame.heartbeat()


class TestTSCollectFrameEmptyValidation:
    """Tests for empty collector validation"""

    def test_close_empty_collector_raises(self):
        """Test that closing collector with no buffers raises ValueError"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))
        collector = frame.fill()

        # Try to close without adding any buffers
        with pytest.raises(ValueError, match="Cannot validate empty collector"):
            collector.close()

    def test_context_manager_empty_collector_raises(self):
        """Test that exiting context with no buffers raises ValueError"""
        frame = TSFrame(offset=0, noffset=Offset.fromsamples(256, 256))

        with pytest.raises(ValueError, match="Cannot validate empty collector"):
            with frame.fill():
                # Don't append any buffers
                pass
