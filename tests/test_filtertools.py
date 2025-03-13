"""Tests for the filtertools module."""

import numpy.testing
import pytest
import scipy

from sgnts import filtertools


class TestLowPassFilter:
    """Test group for low_pass_filter function."""

    def test_size_from_n_zeros(self):
        """Test that size is computed correctly from n_zeros."""
        assert filtertools.sinc_sample_size(64, 1024, 5) == 81
        assert filtertools.sinc_sample_size(30, 1024, 4) == 139

    def test_raw_filter(self):
        """Test that the raw filter is computed correctly."""
        f = filtertools.low_pass_filter(
            64,
            1024,
            n_zeros=5,
            normalize=False,
            win_func=None,
        )
        assert len(f) == 81
        exp = numpy.array(
            [
                3.89817183e-17,
                2.49870605e-02,
                4.73850693e-02,
                6.35848625e-02,
                7.07355303e-02,
                6.72182832e-02,
                5.29597833e-02,
                2.95301624e-02,
                -3.89817183e-17,
                -3.14353341e-02,
            ]
        )
        numpy.testing.assert_almost_equal(f[:10], exp, decimal=5)

    def test_raw_filter_norm(self):
        """Test that the raw filter is computed correctly."""
        f = filtertools.low_pass_filter(
            64,
            1024,
            n_zeros=5,
            normalize=True,
            win_func=None,
        )
        assert len(f) == 81
        exp = numpy.array(
            [
                4.68668983e-18,
                3.00414162e-03,
                5.69700701e-03,
                7.64467400e-03,
                8.50438372e-03,
                8.08151251e-03,
                6.36724312e-03,
                3.55034918e-03,
                -4.68668983e-18,
                -3.77940397e-03,
            ]
        )
        numpy.testing.assert_almost_equal(f[:10], exp, decimal=5)

    def test_windowed(self):
        """Test that the windowed filter is computed correctly."""
        f = filtertools.low_pass_filter(
            64,
            1024,
            n_zeros=5,
            normalize=True,
            win_func=scipy.signal.windows.blackman,
        )
        assert len(f) == 81
        exp = numpy.array(
            [
                -6.76375019e-35,
                1.73823674e-06,
                1.32728420e-05,
                4.05090577e-05,
                8.13028934e-05,
                1.22942418e-04,
                1.42490492e-04,
                1.10759563e-04,
                -1.95989274e-19,
                -2.05600360e-04,
            ]
        )
        numpy.testing.assert_almost_equal(f[:10], exp, decimal=5)

    def test_low_pass_filter_size_gt_fixsize(self):
        """Test err low pass filter args"""
        res = filtertools.low_pass_filter(
            64,
            1024,
            size=100,
            fix_size=50,
        )
        assert len(res) == 50

    def test_low_pass_filter_size_lt_fixsize(self):
        """Test err low pass filter args"""
        res = filtertools.low_pass_filter(
            64,
            1024,
            size=50,
            fix_size=100,
        )
        assert len(res) == 100

    def test_err_low_pass_filter_args(self):
        """Test err low pass filter args"""
        with pytest.raises(ValueError):
            filtertools.low_pass_filter(
                64,
                1024,
                size=None,
                n_zeros=None,
            )
