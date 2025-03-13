"""Tests for the fake_series module."""

import numpy

from sgnts.base import TSFrame
from sgnts.sources import FakeSeriesSource


class TestFakeSeriesSource:
    """Test group for the FakeSeriesSource class."""

    def test_init(self):
        """Test the __init__ method."""
        src = FakeSeriesSource(
            name="test",
            rate=2048,
            sample_shape=(2, 4),
            signal_type="white",
            t0=0,
            duration=1,
            source_pad_names=["S1"],
        )
        assert isinstance(src, FakeSeriesSource)

    def test_sin_wave_1d(self):
        """Test the sine wave generation for 1D data."""
        src = FakeSeriesSource(
            name="test",
            rate=16,
            signal_type="sin",
            fsin=1,
            t0=0,
            duration=1,
            source_pad_names=["S1"],
        )
        res = src.new(pad=src.srcs["S1"])
        assert isinstance(res, TSFrame)
        expected = numpy.array(
            [
                0.00000000e00,
                3.82683432e-01,
                7.07106781e-01,
                9.23879533e-01,
                1.00000000e00,
                9.23879533e-01,
                7.07106781e-01,
                3.82683432e-01,
                1.22464680e-16,
                -3.82683432e-01,
                -7.07106781e-01,
                -9.23879533e-01,
                -1.00000000e00,
                -9.23879533e-01,
                -7.07106781e-01,
                -3.82683432e-01,
            ]
        )
        numpy.testing.assert_almost_equal(res.buffers[0].data, expected, decimal=5)

    def test_sin_wave_2d(self):
        """Test the sine wave generation for 2D data."""
        src = FakeSeriesSource(
            name="test",
            rate=16,
            sample_shape=(2,),
            signal_type="sin",
            fsin=1,
            t0=0,
            duration=1,
            source_pad_names=["S1"],
        )
        res = src.new(pad=src.srcs["S1"])
        assert isinstance(res, TSFrame)
        expected = numpy.array(
            [
                [
                    0.00000000e00,
                    3.82683432e-01,
                    7.07106781e-01,
                    9.23879533e-01,
                    1.00000000e00,
                    9.23879533e-01,
                    7.07106781e-01,
                    3.82683432e-01,
                    1.22464680e-16,
                    -3.82683432e-01,
                    -7.07106781e-01,
                    -9.23879533e-01,
                    -1.00000000e00,
                    -9.23879533e-01,
                    -7.07106781e-01,
                    -3.82683432e-01,
                ],
                [
                    0.00000000e00,
                    3.82683432e-01,
                    7.07106781e-01,
                    9.23879533e-01,
                    1.00000000e00,
                    9.23879533e-01,
                    7.07106781e-01,
                    3.82683432e-01,
                    1.22464680e-16,
                    -3.82683432e-01,
                    -7.07106781e-01,
                    -9.23879533e-01,
                    -1.00000000e00,
                    -9.23879533e-01,
                    -7.07106781e-01,
                    -3.82683432e-01,
                ],
            ]
        )
        numpy.testing.assert_almost_equal(res.buffers[0].data, expected, decimal=5)

    def test_const_int(self):
        """Test the constant int generation"""
        src = FakeSeriesSource(
            name="test",
            rate=16,
            signal_type="const",
            const=2,
            t0=0,
            duration=1,
            source_pad_names=["S1"],
        )
        res = src.new(pad=src.srcs["S1"])
        assert isinstance(res, TSFrame)
        expected = numpy.array(
            [
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )
        numpy.testing.assert_equal(res.buffers[0].data, expected)

    def test_const_float(self):
        """Test the constant float generation"""
        src = FakeSeriesSource(
            name="test",
            rate=16,
            signal_type="const",
            const=3.4,
            t0=0,
            duration=1,
            source_pad_names=["S1"],
        )
        res = src.new(pad=src.srcs["S1"])
        assert isinstance(res, TSFrame)
        expected = numpy.array(
            [
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
                3.4,
            ]
        )
        numpy.testing.assert_equal(res.buffers[0].data, expected)
