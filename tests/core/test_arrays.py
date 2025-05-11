"""Tests for array backends"""

import numpy
import pytest

from sgnts.base.array_ops import ArrayBackend, NumpyBackend, TorchBackend

TorchBackend.DEVICE


class TestArrayBackend:
    """Test group for ArrayBackend class"""

    def test_arange(self):
        """Test the arange method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.arange(0, 0, 0)

    def test_cat(self):
        """Test the cat method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.cat([], axis=0)

    def test_full(self):
        """Test the full method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.full(shape=(1, 2), fill_value=0)

    def test_matmul(self):
        """Test the matmul method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.matmul(None, None)

    def test_ones(self):
        """Test the ones method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.ones(shape=(1, 2))

    def test_pad(self):
        """Test the pad method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.pad(None, pad_samples=(0, 0))

    def test_stack(self):
        """Test the stack method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.stack([], axis=0)

    def test_sum(self):
        """Test the sum method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.sum(None, 0)

    def test_zeros(self):
        """Test the zeros method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.zeros(shape=(1, 2))


class TestNumpyBackend:
    """Test group for NumpyBackend class"""

    def test_constants(self):
        """Test the constants of the NumpyBackend class"""
        assert NumpyBackend.DEVICE == "cpu"
        assert NumpyBackend.DTYPE == numpy.float64

    def test_arange(self):
        """Test the arange method of the NumpyBackend class"""
        res = NumpyBackend.arange(10, start=1, step=2)
        assert numpy.all(res == numpy.arange(start=1, stop=10, step=2))

    def test_cat(self):
        """Test the cat method of the NumpyBackend class"""
        res = NumpyBackend.cat(
            [
                numpy.array([1, 2, 3]),
                numpy.array([4, 5, 6]),
            ],
            axis=0,
        )
        assert numpy.all(res == numpy.array([1, 2, 3, 4, 5, 6]))

    def test_full(self):
        """Test the full method of the NumpyBackend class"""
        res = NumpyBackend.full(shape=(2, 3), fill_value=1)
        assert numpy.all(res == (numpy.ones((2, 3))))

    def test_matmul(self):
        """Test the matmul method of the NumpyBackend class"""
        res = NumpyBackend.matmul(numpy.ones((3, 3)), numpy.ones((3, 3)))
        assert numpy.all(res == numpy.ones((3, 3)) * 3)

    def test_ones(self):
        """Test the ones method of the NumpyBackend class"""
        res = NumpyBackend.ones(shape=(2, 3))
        assert numpy.all(res == numpy.ones((2, 3)))

    def test_pad(self):
        """Test the pad method of the NumpyBackend class"""
        res = NumpyBackend.pad(numpy.array([1, 2, 3]), pad_samples=(1, 2))
        assert numpy.all(res == numpy.array([0, 1, 2, 3, 0, 0]))

    def test_stack(self):
        """Test the stack method of the NumpyBackend class"""
        res = NumpyBackend.stack(
            [
                numpy.array([1, 2, 3]),
                numpy.array([4, 5, 6]),
            ],
            axis=0,
        )
        assert numpy.all(res == numpy.array([[1, 2, 3], [4, 5, 6]]))

    def test_sum(self):
        """Test the sum method of the NumpyBackend class"""
        res = NumpyBackend.sum(numpy.array([[1, 2, 3], [4, 5, 6]]), axis=0)
        assert numpy.all(res == numpy.array([5, 7, 9]))

    def test_zeros(self):
        """Test the zeros method of the NumpyBackend class"""
        res = NumpyBackend.zeros(shape=(2, 3))
        assert numpy.all(res == numpy.zeros((2, 3)))
