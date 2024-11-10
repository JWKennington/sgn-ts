#!/usr/bin/env python3

import numpy
import pytest
import torch
from sgn.apps import Pipeline

from sgnts.base import SeriesBuffer
from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSrc


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
            rate=2048,
            signal_type="white",
            random_seed=1234,
            end=2,
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
        data=ones_function(shape), offset=offset, sample_rate=sample_rate, shape=shape
    )


@pytest.fixture
def a_params():
    return {"offset": 0, "sample_rate": 1024, "shape": (1024,)}


@pytest.fixture
def b_params():
    return {"offset": 1024, "sample_rate": 1024, "shape": (1024,)}


@pytest.fixture
def c_params():
    return {
        "offset": 4096,
        "sample_rate": 1024,
        "shape": (
            2,
            1024,
        ),
    }


@pytest.fixture
def d_params():
    return {
        "offset": 2048,
        "sample_rate": 1024,
        "shape": (
            2,
            1024,
        ),
    }


@pytest.fixture
def e_params():
    return {"offset": 8192, "sample_rate": 2048, "shape": (1024,)}


@pytest.fixture
def f_params():
    return {"offset": 65536, "sample_rate": 1024, "shape": (1024,)}


@pytest.fixture
def g_params():
    return {"offset": 8192, "sample_rate": 1024, "shape": (2048,)}


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
def numpy_f(f_params):
    return make_ones_buffer(numpy.ones, **f_params)


@pytest.fixture
def numpy_g(g_params):
    return make_ones_buffer(numpy.ones, **g_params)


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


@pytest.fixture
def torch_f(f_params):
    return make_ones_buffer(torch.ones, **f_params)


@pytest.fixture
def torch_g(g_params):
    return make_ones_buffer(torch.ones, **g_params)


# @pytest.fixture
# def empty_a(a_params):
#     return SeriesBuffer(data=None, **a_params)


# @pytest.fixture
# def empty_b(b_params):
#     return SeriesBuffer(data=None, **b_params)


# @pytest.fixture
# def empty_c(c_params):
#     return SeriesBuffer(data=None, **c_params)


# @pytest.fixture
# def empty_d(d_params):
#     return SeriesBuffer(data=None, **d_params)


# @pytest.fixture
# def empty_e(e_params):
#     return SeriesBuffer(data=None, **e_params)


# @pytest.fixture
# def empty_f(f_params):
#     return SeriesBuffer(data=None, **f_params)


# @pytest.fixture
# def empty_g(g_params):
#     return SeriesBuffer(data=None, **g_params)


def test_fail_incompatible_data_types(numpy_a, torch_a):
    with pytest.raises(TypeError):
        numpy_a + torch_a


def test_fail_incompatible_sample_rates(numpy_a, numpy_e):
    with pytest.raises(ValueError):
        numpy_a + numpy_e


def test_fail_incompatible_dimensions(numpy_a, numpy_c):
    with pytest.raises(ValueError):
        numpy_a + numpy_c


def test_fail_non_series_buffer_addition(numpy_a, a_params):
    with pytest.raises(TypeError):
        numpy_a + numpy.ones(a_params["shape"])


def test_add_self_numpy(numpy_a, a_params):
    one_plus_one = SeriesBuffer(data=numpy.ones(a_params["shape"]) * 2, **a_params)
    assert numpy_a + numpy_a == one_plus_one
    numpy_a += numpy_a
    assert numpy_a == one_plus_one


def test_add_overlapping_numpy(numpy_a, numpy_b):
    # At srate of 1024 b's offset of 1024
    # is 64 samples behind that of a
    data = numpy.concat(
        [
            numpy.ones(64),
            2 * numpy.ones(960),
            numpy.ones(
                64,
            ),
        ]
    )
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=data.shape, data=data)
    assert numpy_a + numpy_b == correct
    numpy_a += numpy_b
    assert numpy_a == correct


def test_add_different_shape_numpy(numpy_a, numpy_g):
    # g starts 512 samples after a
    # and is 2048 samples long
    data = numpy.concat([numpy.ones(512), 2 * numpy.ones(512), numpy.ones(1536)])
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=data.shape, data=data)
    assert numpy_a + numpy_g == correct
    numpy_a += numpy_g
    assert numpy_a == correct


def test_add_disjoint_numpy(numpy_a, numpy_f):
    # At sample rate of 1024 offset of 65536 comes
    # 4096 samples after offset of 0
    # since a has shape 1024 that leaves 3072 zeros
    # between a and f
    data = numpy.concat([numpy.ones(1024), numpy.zeros(3072), numpy.ones(1024)])
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=data.shape, data=data)
    assert numpy_a + numpy_f == correct
    numpy_a += numpy_f
    assert numpy_a == correct


def test_add_nonflat_numpy(numpy_c, numpy_d):
    # At sample rate of 1024 offset of 4096 comes
    # 128 samples after offset of 1028
    # since c and d have time shape 1024
    # There are 128 samples on either side
    data = numpy.concat([numpy.ones(128), 2 * numpy.ones(896), numpy.ones(128)])
    data = numpy.expand_dims(data, axis=1)
    data = data.transpose()
    data = data.repeat(2, axis=0)
    correct = SeriesBuffer(offset=2048, sample_rate=1024, shape=data.shape, data=data)
    assert numpy_c + numpy_d == correct
    numpy_c += numpy_d
    assert numpy_c == correct


def test_add_self_torch(torch_a, a_params):
    one_plus_one = SeriesBuffer(data=torch.ones(a_params["shape"]) * 2, **a_params)
    assert torch_a + torch_a == one_plus_one
    torch_a += torch_a
    assert torch_a == one_plus_one


def test_add_overlapping_torch(torch_a, torch_b):
    # At srate of 1024 b's offset of 1024
    # is 64 samples behind that of a
    data = torch.concat(
        [
            torch.ones(64),
            2 * torch.ones(960),
            torch.ones(
                64,
            ),
        ]
    )
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=data.shape, data=data)
    assert torch_a + torch_b == correct
    torch_a += torch_b
    assert torch_a == correct


def test_add_different_shape_torch(torch_a, torch_g):
    # g starts 512 samples after a
    # and is 2048 samples long
    data = torch.concat([torch.ones(512), 2 * torch.ones(512), torch.ones(1536)])
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=(2560,), data=data)
    assert torch_a + torch_g == correct
    torch_a += torch_g
    assert torch_a == correct


def test_add_disjoint_torch(torch_a, torch_f):
    # At sample rate of 1024 offset of 65536 comes
    # 4096 samples after offset of 0
    # since a has shape 1024 that leaves 3072 zeros
    # between a and f
    data = torch.concat([torch.ones(1024), torch.zeros(3072), torch.ones(1024)])
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=data.shape, data=data)
    assert torch_a + torch_f == correct
    torch_a += torch_f
    assert torch_a == correct


def test_add_nonflat_torch(torch_c, torch_d):
    # At sample rate of 1024 offset of 4096 comes
    # 128 samples after offset of 1028
    # since c and d have time shape 1024
    # There are 128 samples on either side
    data = torch.concat([torch.ones(128), 2 * torch.ones(896), torch.ones(128)])
    data = data[None, :]
    data = data.repeat((2, 1))
    correct = SeriesBuffer(offset=2048, sample_rate=1024, shape=data.shape, data=data)
    assert torch_c + torch_d == correct
    torch_c += torch_d
    assert torch_c == correct
