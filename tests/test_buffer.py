"""Unit tests for the buffer module"""

from sgnts.base import NumpyBackend, Offset, SeriesBuffer, TSFrame


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
