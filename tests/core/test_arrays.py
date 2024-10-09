"""Tests for array backends"""

import sys
from unittest import mock

import numpy
import pytest
import torch

from sgnts.base.array_ops import ArrayBackend, CPUTorchBackend, GPUTorchBackend, \
    NumpyBackend


class TestArrayBackend:
    """Test group for ArrayBackend class"""

    def test_constants(self):
        """Test the constants of the ArrayBackend class"""
        assert ArrayBackend.DEVICE is None
        assert ArrayBackend.DTYPE is None

    def test_cat(self):
        """Test the cat method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.cat([], axis=0)

    def test_pad(self):
        """Test the pad method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.pad(None, pad_samples=(0, 0))

    def test_full(self):
        """Test the full method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.full(shape=(1, 2), fill_value=0)

    def test_zeros(self):
        """Test the zeros method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.zeros(shape=(1, 2))

    def test_stack(self):
        """Test the stack method of the ArrayBackend class"""
        with pytest.raises(NotImplementedError):
            ArrayBackend.stack([], axis=0)


class TestNumpyBackend:

    def test_constants(self):
        """Test the constants of the NumpyBackend class"""
        assert NumpyBackend.DEVICE == "cpu"
        assert NumpyBackend.DTYPE == numpy.float64

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

    def test_pad(self):
        """Test the pad method of the NumpyBackend class"""
        res = NumpyBackend.pad(numpy.array([1, 2, 3]), pad_samples=(0, 2))
        assert numpy.all(res == numpy.array([1, 2, 3, 0, 0]))

    def test_full(self):
        """Test the full method of the NumpyBackend class"""
        res = NumpyBackend.full(shape=(2, 3), fill_value=1)
        assert numpy.all(res == (numpy.zeros((2, 3)) + 1))

    def test_zeros(self):
        """Test the zeros method of the NumpyBackend class"""
        res = NumpyBackend.zeros(shape=(2, 3))
        assert numpy.all(res == numpy.zeros((2, 3)))

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


class TestCPUTorchBackend:
    """Test group for CPUTorchBackend class"""

    def test_constants(self):
        """Test the constants of the CPUTorchBackend class"""
        assert CPUTorchBackend.DEVICE == "cpu"
        assert CPUTorchBackend.DTYPE == torch.float32

    def test_cat(self):
        """Test the cat method of the CPUTorchBackend class"""
        res = CPUTorchBackend.cat(
            [
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ],
            axis=0,
        )
        assert torch.all(res == torch.tensor([1, 2, 3, 4, 5, 6]))

    def test_pad(self):
        """Test the pad method of the CPUTorchBackend class"""
        res = CPUTorchBackend.pad(torch.tensor([1, 2, 3]), pad_samples=(0, 2))
        assert torch.all(res == torch.tensor([1, 2, 3, 0, 0]))

    def test_full(self):
        """Test the full method of the CPUTorchBackend class"""
        res = CPUTorchBackend.full(shape=(2, 3), fill_value=1)
        assert torch.all(res == (torch.zeros((2, 3)) + 1))

    def test_zeros(self):
        """Test the zeros method of the CPUTorchBackend class"""
        res = CPUTorchBackend.zeros(shape=(2, 3))
        assert torch.all(res == torch.zeros((2, 3)))

    def test_stack(self):
        """Test the stack method of the CPUTorchBackend class"""
        res = CPUTorchBackend.stack(
            [
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ],
            axis=0,
        )
        assert torch.all(res == torch.tensor([[1, 2, 3], [4, 5, 6]]))


class TestGPUTorchBackend:
    """Test group for GPUTorchBackend class"""

    def test_constants(self):
        """Test the constants of the GPUTorchBackend class"""
        assert GPUTorchBackend.DEVICE == "cuda"
        assert GPUTorchBackend.DTYPE == torch.float32

    def test_cat(self):
        """Test the cat method of the GPUTorchBackend class"""
        res = GPUTorchBackend.cat(
            [
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ],
            axis=0,
        )
        assert torch.all(res == torch.tensor([1, 2, 3, 4, 5, 6]))

    def test_pad(self):
        """Test the pad method of the GPUTorchBackend class"""
        res = GPUTorchBackend.pad(torch.tensor([1, 2, 3]), pad_samples=(0, 2))
        assert torch.all(res == torch.tensor([1, 2, 3, 0, 0]))

    def test_full(self):
        """Test the full method of the GPUTorchBackend class"""
        res = GPUTorchBackend.full(shape=(2, 3), fill_value=1)
        assert torch.all(res == (torch.zeros((2, 3)) + 1))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_zeros(self):
        """Test the zeros method of the GPUTorchBackend class"""
        res = GPUTorchBackend.zeros(shape=(2, 3))
        assert torch.all(res == torch.zeros((2, 3)))

    def test_stack(self):
        """Test the stack method of the GPUTorchBackend class"""
        res = GPUTorchBackend.stack(
            [
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
            ],
            axis=0,
        )
        assert torch.all(res == torch.tensor([[1, 2, 3], [4, 5, 6]]))

    def test_check_torch(self):
        """Test the check_torch method of the GPUTorchBackend class"""
        GPUTorchBackend._check_torch()

    def test_check_torch_err(self):
        """Test the check_torch method of the GPUTorchBackend class"""
        # Patch the torch import to raise an ImportError
        original = sys.modules
        keys = ['torch', 'sgnts']
        clean = {k: v for k, v in original.items() if all(key not in k for key in keys)}
        clean.update({'torch': None})
        with mock.patch.dict("sys.modules", clear=True, values=clean):
            from sgnts.base.array_ops import GPUTorchBackend

            with pytest.raises(ImportError):
                GPUTorchBackend._check_torch()
