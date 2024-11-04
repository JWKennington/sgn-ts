#!/usr/bin/env python3

from typing import Any
from sgn.apps import Pipeline

import numpy
import torch

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSrc
from sgnts.base.buffer import SeriesBuffer

import pytest
import unittest

def test_tsgraph(capsys):

    pipeline = Pipeline()

    #
    #       ----------
    #      | src1     |
    #       ----------
    #              \
    #           H1  \
    #           ------------
    #          | snk1      |
    #           ------------

    pipeline.insert(
        FakeSeriesSrc(
            name="src1",
            source_pad_names=("H1",),
            num_buffers=2,
            rate=2048,
            signal_type="white",
            random_seed=1234,
        ),
        FakeSeriesSink(
            name="snk1",
            sink_pad_names=("H1",),
            verbose=True,
        ),
        link_map={"snk1:sink:H1": "src1:src:H1"},
    )

    pipeline.run()

"""Tests of buffer addition"""

def make_ones_buffer(ones_function, offset=0, sample_rate=1, shape=(0,)):
    return SeriesBuffer(
        data = ones_function(shape),
        offset=offset,
        sample_rate=sample_rate,
        shape=shape
    )

@pytest.fixture
def a_params():
    return {"offset":0, "sample_rate":1024, "shape":(1024,)}

@pytest.fixture
def b_params():
    return {"offset":1024, "sample_rate":1024, "shape":(1024,)}

@pytest.fixture
def c_params():
    return {"offset":4096, "sample_rate":1024, "shape":(2, 1024,)}

@pytest.fixture
def d_params():
    return {"offset":2048, "sample_rate":1024, "shape":(2, 1024,)}

@pytest.fixture
def e_params():
    return {"offset":8192, "sample_rate":2048, "shape":(1024,)}

@pytest.fixture
def numpy_a(a_params):
    return make_ones_buffer(numpy.ones, **a_params)

@pytest.fixture
def numpy_b(b_params):
    return make_ones_buffer(numpy.ones, **b_params)

@pytest.fixture
def numpy_c(c_params):
    return make_ones_buffer(numpy.ones, **c_params)

@pytest.fixture
def numpy_d(d_params):
    return make_ones_buffer(numpy.ones, **d_params)

@pytest.fixture
def numpy_e(e_params):
    return make_ones_buffer(numpy.ones, **e_params)

@pytest.fixture
def torch_a(a_params):
    return make_ones_buffer(torch.ones, **a_params)

@pytest.fixture
def torch_b(b_params):
    return make_ones_buffer(torch.ones, **b_params)

@pytest.fixture
def torch_c(c_params):
    return make_ones_buffer(torch.ones, **c_params)

@pytest.fixture
def torch_d(d_params):
    return make_ones_buffer(torch.ones, **d_params)

@pytest.fixture
def torch_e(e_params):
    return make_ones_buffer(torch.ones, **e_params)

def test_fail_incompatible_data_types(numpy_a, torch_a):
    with pytest.raises(TypeError):
        numpy_a + torch_a, TypeError

def test_fail_incompatible_sample_rates(numpy_a, numpy_e):
    with pytest.raises(ValueError):
        numpy_a + numpy_e

def test_fail_incompatible_dimensions(numpy_a, numpy_c):
    with pytest.raises(ValueError):
        numpy_a + numpy_c

def test_fail_non_series_buffer_addition(numpy_a, a_params):
    with pytest.raises(TypeError):
        numpy_a + numpy.ones(a_params['shape'])

def test_add_self_numpy(numpy_a, a_params):
    one_plus_one = SeriesBuffer(
        data=numpy.ones(a_params['shape']) * 2,
        **a_params
    )
    assert numpy_a + numpy_a == one_plus_one
    numpy_a += numpy_a
    assert numpy_a == one_plus_one

if __name__ == "__main__":
    test_tsgraph(None)
