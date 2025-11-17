import numpy
import pytest

from sgnts.base import SeriesBuffer, Event, EventBuffer, TSSlice, EventFrame, TSFrame
from sgnts.base.array_ops import TorchBackend
from sgnts.base.time import Time

torch = pytest.importorskip("torch")


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


def test_fail_incompatible_data_types(numpy_a, torch_a):
    with pytest.raises(TypeError):
        numpy_a + torch_a


def test_add_self_torch(torch_a, a_params):
    one_plus_one = SeriesBuffer(data=torch.ones(a_params["shape"]) * 2, **a_params)
    assert torch_a + torch_a == one_plus_one
    torch_a += torch_a
    assert torch_a == one_plus_one


def test_add_overlapping_torch(torch_a, torch_b):
    # At srate of 1024 b's offset of 1024
    # is 64 samples behind that of a
    data = torch.concatenate(
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
    data = torch.concatenate([torch.ones(512), 2 * torch.ones(512), torch.ones(1536)])
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=(2560,), data=data)
    assert torch_a + torch_g == correct
    torch_a += torch_g
    assert torch_a == correct


def test_add_disjoint_torch(torch_a, torch_f):
    # At sample rate of 1024 offset of 65536 comes
    # 4096 samples after offset of 0
    # since a has shape 1024 that leaves 3072 zeros
    # between a and f
    data = torch.concatenate([torch.ones(1024), torch.zeros(3072), torch.ones(1024)])
    correct = SeriesBuffer(offset=0, sample_rate=1024, shape=data.shape, data=data)
    assert torch_a + torch_f == correct
    torch_a += torch_f
    assert torch_a == correct


def test_add_nonflat_torch(torch_c, torch_d):
    # At sample rate of 1024 offset of 4096 comes
    # 128 samples after offset of 1028
    # since c and d have time shape 1024
    # There are 128 samples on either side
    data = torch.concatenate([torch.ones(128), 2 * torch.ones(896), torch.ones(128)])
    data = data[None, :]
    data = data.repeat((2, 1))
    correct = SeriesBuffer(offset=2048, sample_rate=1024, shape=data.shape, data=data)
    assert torch_c + torch_d == correct
    torch_c += torch_d
    assert torch_c == correct


def test_badly_initialized_event_buffer():
    with pytest.raises(ValueError):
        EventBuffer("a", "b")


def test_print_event_buffer():
    ebuf = EventBuffer(1, 2)
    repr(ebuf)


def test_bool_event_buffer():
    ebuf = EventBuffer(1, 2)
    assert not ebuf


def test_misc_event_buffer():
    ebuf = EventBuffer(offset=1, noffset=2)
    assert 1 in ebuf
    assert "a" not in ebuf
    assert EventBuffer(0, 1) < ebuf
    assert EventBuffer(0, 2) <= ebuf
    assert EventBuffer(2, 3) > ebuf
    assert EventBuffer(1, 3) >= ebuf
    assert ebuf.slice == TSSlice(1, 3)
    assert ebuf.end_offset == 3
    assert ebuf.is_gap
    event = Event(1, data={"a": "b"})
    ebuf = EventBuffer(1, 2, data=[event])
    assert not ebuf.is_gap


def test_event_frame():
    event = Event(1, data={"k": "v"})
    ebuf = EventBuffer(1, 2, data=[event])
    eframe = EventFrame(data=[ebuf])
    assert eframe.events[0].data["k"] == "v"
    for e in eframe.events:
        assert e.data["k"] == "v"
    repr(eframe)


def test_valid_series_buffer():
    with pytest.raises(ValueError):
        SeriesBuffer(offset=0, sample_rate=1234)
    with pytest.raises(ValueError):
        SeriesBuffer(offset=0, sample_rate=128, data=0)
    with pytest.raises(ValueError):
        SeriesBuffer(offset=0, sample_rate=128, data=1)
    with pytest.raises(ValueError):
        SeriesBuffer(offset=0, sample_rate=128)
    buf = SeriesBuffer(offset=0, sample_rate=128, shape=(128,), data=0)
    buf2 = SeriesBuffer(offset=16384, sample_rate=128, shape=(128,), data=0)
    buf3 = SeriesBuffer(offset=0, sample_rate=128, shape=(512,), data=0)
    frame = TSFrame(buffers=[buf])
    frame2 = TSFrame(buffers=[buf2])
    frame3 = TSFrame(buffers=[buf3])
    b, i, a = frame2.intersect(frame)
    assert i is None and a is None and b is not None
    b, i, a = frame.intersect(frame2)
    assert b is None and i is None and a is not None
    b, i, a = frame3.intersect(frame2)
    assert b is None and i is not None and a is None
    b, i, a = frame2.intersect(frame3)
    assert b is not None and i is not None and a is not None

    assert frame.slice == TSSlice(0, 16384)
    assert frame.sample_shape == ()
    assert frame.heartbeat().end_offset == 0

    assert len(buf) == 128
    with pytest.raises(ValueError):
        SeriesBuffer(
            offset=0, sample_rate=128, shape=(128,), data=numpy.array([1, 2, 3])
        )
    buf2 = buf.new(data=buf.data)
    assert numpy.array_equal(
        buf.tarr,
        buf2.backend.arange(buf.samples) / buf.sample_rate + buf.t0 / Time.SECONDS,
    )
    assert buf2 == buf
    with pytest.raises(ValueError):
        buf2.set_data(data=numpy.array([1, 2, 3]))
    assert not buf2 == "blah"
    assert not buf2 == SeriesBuffer(offset=0, sample_rate=128, shape=(127,), data=0)

    assert buf2.end == 1_000_000_000
    assert buf2 <= SeriesBuffer(offset=0, sample_rate=128, shape=(128,), data=0)
    assert buf2 >= SeriesBuffer(offset=0, sample_rate=128, shape=(128,), data=0)
    assert buf2 > SeriesBuffer(offset=-16384, sample_rate=128, shape=(128,), data=0)
    assert 1000 in buf2
    assert "blah" not in buf2

    with pytest.raises(NotImplementedError):
        buf2.split("blah")

    bufset_test = SeriesBuffer(offset=0, sample_rate=128, shape=(128,), data=0)
    bufset_test.set_data(1)
    bufset_test.set_data(0)

    tbuf = SeriesBuffer(
        offset=-16384, sample_rate=128, shape=(128,), data=0, backend=TorchBackend
    )
    assert tbuf.sample_shape == ()
    original_dtype = TorchBackend.DTYPE
    TorchBackend.DTYPE = torch.float64
    with pytest.raises(ValueError):
        assert tbuf._backend_from_data.DTYPE == original_dtype
    TorchBackend.DTYPE = original_dtype
    tbuf.data = None
    assert tbuf._backend_from_data is None

    tbuf = SeriesBuffer(
        offset=-16384,
        sample_rate=128,
        shape=(128,),
        data=0,
        backend=TorchBackend,
    )
    tbuf2 = SeriesBuffer(
        offset=-16384, sample_rate=128, shape=(128,), data=None, backend=TorchBackend
    )

    assert (tbuf + tbuf2) == tbuf
    assert (tbuf2 + tbuf) == tbuf
    assert (tbuf2 + tbuf2) == tbuf2

    # NOTE: These seem like impossible situations, but I am testing it here
    # anyway since it is required for coverage
    # FIXME: check conditionals of SeriesBuffer equality.  asserting types are
    # the same seems useless, maybe it should be dtypes? Or is this meant to
    # caputure torch arrays?
    buf3 = SeriesBuffer(offset=0, sample_rate=128, shape=(128,), data=numpy.arange(128))
    buf3.data = "blah"
    assert not buf2 == buf3
    buf4 = SeriesBuffer(offset=0, sample_rate=128, shape=(128,), data=numpy.arange(128))
    buf4.data = "blah"
    with pytest.raises(ValueError):
        assert buf3 == buf4
