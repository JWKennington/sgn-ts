"""Tests for audioadapter"""

from collections import deque

import numpy
import pytest

from sgnts.base import Audioadapter, Offset, SeriesBuffer


class TestProperties:
    """Test group for audioadapter properties"""

    def test_offset(self):
        """Test offset error"""
        a = Audioadapter()
        with pytest.raises(ValueError):
            a.offset

    def test_offset2(self):
        """Test offset"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        a.push(b1)
        assert a.offset == 0

    def test_end_offset(self):
        """Test end offset error"""
        a = Audioadapter()
        with pytest.raises(ValueError):
            a.end_offset

    def test_end_offset2(self):
        """Test end offset"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        a.push(b1)
        assert a.end_offset == b1.end_offset

    def test_slice(self):
        """Test slice error"""
        a = Audioadapter()
        with pytest.raises(ValueError):
            a.slice

    def test_slice2(self):
        """Test slice"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        a.push(b1)
        assert a.slice == (b1.offset, b1.end_offset)

    def test_is_gap(self):
        """Test is_gap method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        a.push(b1)
        assert a.is_gap is False

    def test_is_gap2(self):
        """Test is_gap method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(2048,))
        a.push(b1)
        assert a.is_gap

    def test_is_gap3(self):
        """Test is_gap method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(2048,))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        a.push(b2)
        assert a.is_gap is False


class TestConcatenateData:
    """Test group for concatenate_date method"""

    def test_concatenate_data(self):
        """Test concatenate_data method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.arange(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.concatenate_data()
        assert a.pre_cat_data == SeriesBuffer(
            offset=0, sample_rate=2048, data=numpy.arange(4096)
        )

    def test_concatenate_data2(self):
        """Test concatenate_data method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.arange(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.concatenate_data((Offset.fromsec(0.5), Offset.fromsec(1.5)))
        assert a.pre_cat_data == SeriesBuffer(
            offset=Offset.fromsec(0.5), sample_rate=2048, data=numpy.arange(1024, 3072)
        )


class TestPush:
    """Test group for push method"""

    def test_push1(self):
        """Test push one buffer"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        a.push(b1)
        assert a.offset == b1.offset
        assert a.end_offset == b1.end_offset
        assert a.slice == (b1.offset, b1.end_offset)
        assert a.is_gap is False
        assert a.size == b1.samples
        assert a.gap_size == 0
        assert a.nongap_size == b1.samples
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_push2(self):
        """Test push two buffers"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        a.push(b2)
        assert a.offset == b1.offset
        assert a.end_offset == b2.end_offset
        assert a.slice == (b1.offset, b2.end_offset)
        assert a.size == b1.samples + b2.samples
        assert a.is_gap is False
        assert a.gap_size == 0
        assert a.nongap_size == b1.samples + b2.samples
        assert a.sample_rate == b2.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 2

    def test_push_zero_length1(self):
        """Test push one zero length buffer"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(0,))
        a.push(b1)
        assert a.offset == b1.offset
        assert a.end_offset == b1.end_offset
        assert a.slice == (b1.offset, b1.end_offset)
        assert a.size == 0
        assert a.is_gap is True
        assert a.gap_size == 0
        assert a.nongap_size == 0
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_push_zero_length2(self):
        """Test push two zero length buffer"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(0,))
        b2 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(0,))
        a.push(b1)
        a.push(b2)
        assert a.offset == 0
        assert a.end_offset == 0
        assert a.slice == (0, 0)
        assert a.size == 0
        assert a.is_gap is True
        assert a.gap_size == 0
        assert a.nongap_size == 0
        assert a.sample_rate == b2.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_push_zero_length3(self):
        """Test push zero length buffer"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(0,))
        b2 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        a.push(b1)
        a.push(b2)
        assert a.offset == 0
        assert a.end_offset == Offset.fromsec(1)
        assert a.slice == (0, Offset.fromsec(1))
        assert a.size == 2048
        assert a.is_gap is False
        assert a.gap_size == 0
        assert a.nongap_size == 2048
        assert a.sample_rate == b2.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_push_zero_length4(self):
        """Test push zero length buffer"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(0,))
        b2 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        b3 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(0,)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples_by_end_offset(Offset.fromsec(1))
        a.push(b3)
        assert a.offset == Offset.fromsec(1)
        assert a.end_offset == Offset.fromsec(1)
        assert a.slice == (Offset.fromsec(1), Offset.fromsec(1))
        assert a.size == 0
        assert a.is_gap is True
        assert a.gap_size == 0
        assert a.nongap_size == 0
        assert a.sample_rate == b2.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_offset_overlap(self):
        """Test push offset overlap"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        a.push(b1)
        with pytest.raises(ValueError):
            a.push(b1)

    def test_offset_discont(self):
        """Test push offset discontinuities"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset * 2, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        with pytest.raises(ValueError):
            a.push(b2)

    def test_sample_rate_mismatch(self):
        """Test push sample rate mismatch"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=1024, data=numpy.random.rand(2048)
        )
        a.push(b1)
        with pytest.raises(ValueError):
            a.push(b2)


class TestGetSlicedBuffers:
    """Test group for get_sliced_buffers method"""

    def test_get_sliced_buffers1(self):
        """Test get_sliced_buffers method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        a.push(b2)
        offset = Offset.fromsamples(1000, 2048)
        end_offset = Offset.fromsamples(4000, 2048)
        bufs = a.get_sliced_buffers((offset, end_offset))
        assert isinstance(bufs, deque)
        assert len(bufs) == 2
        assert bufs[0].offset == offset
        assert bufs[-1].end_offset == end_offset

    def test_get_sliced_buffers2(self):
        """Test get_sliced_buffers method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.random.rand(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        a.push(b2)
        offset = Offset.fromsamples(1000, 2048)
        end_offset = Offset.fromsamples(4000, 2048)
        with pytest.raises(ValueError):
            a.get_sliced_buffers((offset, end_offset))

    def test_get_sliced_buffers3(self):
        """Test get_sliced_buffers method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        a.push(b2)
        offset = Offset.fromsamples(0, 2048)
        end_offset = Offset.fromsamples(5000, 2048)
        with pytest.raises(ValueError):
            a.get_sliced_buffers((offset, end_offset))

    def test_get_sliced_buffers4(self):
        """Test get_sliced_buffers method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.random.rand(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        a.push(b2)
        offset = Offset.fromsamples(0, 2048)
        end_offset = Offset.fromsamples(5000, 2048)
        bufs = a.get_sliced_buffers((offset, end_offset), pad_start=True)
        assert len(bufs) == 3
        assert bufs[0].is_gap
        assert bufs[0].offset == 0
        assert bufs[1].offset == b1.offset
        assert bufs[2].offset == b2.offset
        assert bufs[2].end_offset == end_offset

    def test_get_sliced_buffers5(self):
        """Test get_sliced_buffers method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.random.rand(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.random.rand(2048)
        )
        a.push(b1)
        a.push(b2)
        offset = Offset.fromsamples(0, 2048)
        end_offset = Offset.fromsamples(4096, 2048)
        bufs = a.get_sliced_buffers((offset, end_offset))
        assert bufs[0].offset == 0
        assert bufs[0].end_offset == b1.end_offset
        assert bufs[1].offset == b2.offset
        assert bufs[1].end_offset == b2.end_offset


class TestCopySamples:
    """Test group for copy_samples method"""

    def test_copy_samples1(self):
        """Test copy_samples method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.arange(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        out = a.copy_samples(10)
        assert numpy.all(out == numpy.arange(10))

    def test_copy_samples2(self):
        """Test copy_samples method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.arange(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        out = a.copy_samples(10, 2)
        assert numpy.all(out == numpy.arange(2, 12))


class TestCopySamplesByOffsetSegment:
    """Test group for copy_samples_by_offset_segment method"""

    def test_copy_samples_by_offset_segment(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.arange(2048))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        out = a.copy_samples_by_offset_segment(
            (Offset.fromsamples(10, 2048), Offset.fromsamples(110, 2048))
        )
        assert numpy.all(out == numpy.arange(10, 110))

    def test_copy_samples_by_offset_segment2(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        out = a.copy_samples_by_offset_segment(
            (Offset.fromsamples(2030, 2048), Offset.fromsamples(4096, 2048)),
            pad_start=True,
        )
        assert numpy.all(
            out == numpy.concatenate((numpy.zeros(18), numpy.arange(2048)))
        )

    def test_copy_samples_by_offset_segment3(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        with pytest.raises(AssertionError):
            a.copy_samples_by_offset_segment(
                (Offset.fromsamples(2048, 2048), Offset.fromsamples(2048 * 4, 2048))
            )

    def test_copy_samples_by_offset_segment4(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        with pytest.raises(AssertionError):
            a.copy_samples_by_offset_segment(
                (Offset.fromsamples(2030, 2048), Offset.fromsamples(4096, 2048))
            )

    def test_copy_samples_by_offset_segment5(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=None, shape=(2048,))
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        out = a.copy_samples_by_offset_segment(
            (Offset.fromsamples(10, 2048), Offset.fromsamples(2058, 2048))
        )
        assert out is None

    def test_copy_samples_by_offset_segment6(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(offset=0, sample_rate=2048, data=numpy.arange(2048))
        a.push(b1)
        out = a.copy_samples_by_offset_segment(
            (Offset.fromsamples(10, 2048), Offset.fromsamples(100, 2048))
        )
        assert numpy.all(out == numpy.arange(10, 100))

    def test_copy_samples_by_offset_segment7(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.concatenate_data()
        out = a.copy_samples_by_offset_segment(
            (Offset.fromsamples(2048, 2048), Offset.fromsamples(4000, 2048))
        )
        assert numpy.all(out == numpy.arange(2048, 4000) - 2048)

    def test_copy_samples_by_offset_segment8(self):
        """Test copy_samples_by_offset_segment method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.concatenate_data(
            (Offset.fromsamples(3000, 2048), Offset.fromsamples(4000, 2048))
        )
        out = a.copy_samples_by_offset_segment(
            (Offset.fromsamples(3048, 2048), Offset.fromsamples(3996, 2048))
        )
        assert numpy.all(out == numpy.arange(3048, 3996) - 2048)


class TestFlushSamples:
    """Test group for flush_samples method"""

    def test_flush_samples(self):
        """Test flush_samples method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples(1024)
        assert a.offset == Offset.fromsec(1.5)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(1.5), b2.end_offset)
        assert a.size == 3072
        assert a.is_gap is False
        assert a.gap_size == 0
        assert a.nongap_size == 3072
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 2

    def test_flush_samples2(self):
        """Test flush_samples method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples(1024)
        assert a.offset == Offset.fromsec(1.5)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(1.5), b2.end_offset)
        assert a.size == 3072
        assert a.is_gap is False
        assert a.gap_size == 1024
        assert a.nongap_size == 2048
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 2
        assert a.buffers[0].is_gap
        assert not a.buffers[1].is_gap


class TestFlushSamplesByEndOffset:
    """Test group for flush_samples_end_offset method"""

    def test_flush_samples_by_end_offset(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples_by_end_offset(Offset.fromsec(2.75))
        assert a.offset == Offset.fromsec(2.75)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(2.75), b2.end_offset)
        assert a.size == 512
        assert a.is_gap is False
        assert a.gap_size == 0
        assert a.nongap_size == 512
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_flush_samples_by_end_offset2(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        with pytest.raises(ValueError):
            a.flush_samples_by_end_offset(Offset.fromsec(0.75))

    def test_flush_samples_by_end_offset3(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        with pytest.raises(ValueError):
            a.flush_samples_by_end_offset(Offset.fromsec(3.75))

    def test_flush_samples_by_end_offset4(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=numpy.arange(2048)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples_by_end_offset(Offset.fromsec(2.75))
        assert a.offset == Offset.fromsec(2.75)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(2.75), b2.end_offset)
        assert a.size == 512
        assert a.is_gap is True
        assert a.gap_size == 512
        assert a.nongap_size == 0
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_flush_samples_by_end_offset5(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples_by_end_offset(Offset.fromsec(2.75))
        assert a.offset == Offset.fromsec(2.75)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(2.75), b2.end_offset)
        assert a.size == 512
        assert a.is_gap is False
        assert a.gap_size == 0
        assert a.nongap_size == 512
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_flush_samples_by_end_offset6(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048, 4096)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples_by_end_offset(Offset.fromsec(3))
        assert a.offset == Offset.fromsec(3)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(3), b2.end_offset)
        assert a.size == 0
        assert a.is_gap is True
        assert a.gap_size == 0
        assert a.nongap_size == 0
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_flush_samples_by_end_offset7(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples_by_end_offset(Offset.fromsec(3))
        assert a.offset == Offset.fromsec(3)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(3), b2.end_offset)
        assert a.size == 0
        assert a.is_gap is True
        assert a.gap_size == 0
        assert a.nongap_size == 0
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1

    def test_flush_samples_by_end_offset8(self):
        """Test flush_samples_end_offset method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.flush_samples_by_end_offset(Offset.fromsec(2))
        assert a.offset == Offset.fromsec(2)
        assert a.end_offset == b2.end_offset
        assert a.slice == (Offset.fromsec(2), b2.end_offset)
        assert a.size == 2048
        assert a.is_gap is True
        assert a.gap_size == 2048
        assert a.nongap_size == 0
        assert a.sample_rate == b1.sample_rate
        assert a.pre_cat_data is None
        assert len(a) == 1


class TestBuffersGapsInfo:
    """Test group for buffers_gaps_info method"""

    def test_buffers_gaps_info(self):
        """Test buffers_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        gaps = a.buffers_gaps_info((Offset.fromsec(1.75), Offset.fromsec(3.75)))
        assert gaps == [True, False, True]

    def test_buffers_gaps_info2(self):
        """Test buffers_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        gaps = a.buffers_gaps_info(
            (Offset.fromsec(0.75), Offset.fromsec(3.75)), pad_start=True
        )
        assert gaps == [True, True, False, True]

    def test_buffers_gaps_info3(self):
        """Test buffers_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        with pytest.raises(ValueError):
            a.buffers_gaps_info((Offset.fromsec(0.75), Offset.fromsec(3.75)))


class TestSamplesGapsInfo:
    """Test group for samples_gaps_info method"""

    def test_samples_gaps_info(self):
        """Test samples_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        gaps = a.samples_gaps_info((Offset.fromsec(1.75), Offset.fromsec(3.75)))
        assert numpy.all(
            gaps
            == numpy.concatenate(
                (numpy.ones(512), numpy.zeros(2048), numpy.ones(1536))
            ).astype(bool)
        )


class TestSegmentGapsInfo:
    """Test group for segment_gaps_info method"""

    def test_segment_gaps_info(self):
        """Test segment_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        gaps = a.segment_gaps_info((Offset.fromsec(1.75), Offset.fromsec(3.75)))
        assert gaps == (True, True)

    def test_segment_gaps_info2(self):
        """Test segment_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1),
            sample_rate=2048,
            data=numpy.arange(2048),
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        gaps = a.segment_gaps_info(
            (Offset.fromsec(0.75), Offset.fromsec(3.75)), pad_start=True
        )
        assert gaps == (True, True)

    def test_segment_gaps_info3(self):
        """Test segment_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1),
            sample_rate=2048,
            data=numpy.arange(2048),
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=numpy.arange(2048)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        with pytest.raises(ValueError):
            a.segment_gaps_info((Offset.fromsec(0.75), Offset.fromsec(3.75)))

    def test_segment_gaps_info4(self):
        """Test segment_gaps_info method"""
        a = Audioadapter()
        b1 = SeriesBuffer(
            offset=Offset.fromsec(1), sample_rate=2048, data=None, shape=(2048,)
        )
        b2 = SeriesBuffer(
            offset=b1.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        b3 = SeriesBuffer(
            offset=b2.end_offset, sample_rate=2048, data=None, shape=(2048,)
        )
        a.push(b1)
        a.push(b2)
        a.push(b3)
        gaps = a.segment_gaps_info((Offset.fromsec(0.75), Offset.fromsec(3.75)))
        assert gaps == (True, False)
