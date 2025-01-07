#!/usr/bin/env python3
import pytest
from sgn.apps import Pipeline

from sgnts.sinks import FakeSeriesSink
from sgnts.sources import FakeSeriesSource


@pytest.mark.skip(reason="Multiplier has bugs and will need to be reworked")
def test_tsgraph():
    from sgnts.transforms import Multiplier

    """Test the tsgraph function

     ----------   ----------     ----------
    |src1      | |src2      |...|srcN      |
     ----------   ----------     ----------
          \           |          /
           \          |         /
            \         |        /
             \        |       /
              \       |      /
               \      |     /
                \     |    /
                 ----------
                |multiply  |
                 ----------
                     |
                     |
                 ----------
                |sink1     |
                 ----------
    """
    num_pads = 2  # sets the number of src pads
    pipeline = Pipeline()
    pipeline.insert(
        FakeSeriesSource(
            name="src1",
            source_pad_names={",".join(["pad" + str(n)]) for n in range(num_pads)},
            signal_type="white",
            rate=2048,
            end=16,
            ngap=2,
        ),
        Multiplier(
            name="mult",
            source_pad_names=("H1",),
            sink_pad_names={",".join(["pad" + str(n)]) for n in range(num_pads)},
            num_samples=2048,
        ),
        FakeSeriesSink(
            name="snk1",
            sink_pad_names=("L1",),
            verbose=True,
        ),
        link_map={  # joining together two dicts to allow for arbitrary num_pads
            "mult:snk:pad" + str(n): "src1:src:pad" + str(n) for n in range(num_pads)
        }
        | {"snk1:snk:L1": "mult:src:H1"},
    )

    pipeline.run()
