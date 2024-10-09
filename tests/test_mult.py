#!/usr/bin/env python3
from sgn.apps import Pipeline
from sgnts.sinks import FakeSeriesSink
from sgnts.transforms import Multiplier
from sgnts.sources import FakeSeriesSrc

def test_tsgraph(capsys): 
    #   ----------   ----------     ----------
    #  |src1      | |src2      |...|srcN      |
    #   ----------   ----------     ----------
    #        \           |          /
    #         \          |         /
    #          \         |        /
    #           \        |       /
    #            \       |      / 
    #             \      |     /
    #              \     |    / 
    #               ----------
    #              |multiply  |
    #               ----------
    #                   |
    #                   |
    #               ----------
    #              |sink1     |
    #               ----------
    
    global num_pads
    num_pads = 2 #sets the number of src pads
    pipeline = Pipeline()
    pipeline.insert(
        FakeSeriesSrc(
               name = "src1",
               source_pad_names = {",".join(["pad" + str(n)]) for n in range(num_pads)},
               num_buffers = 1,
               signal_type = 'white',
               rate=2048,
               num_samples=2048,
         ),Multiplier(
               name = 'mult',
               source_pad_names = ("H1",),
               sink_pad_names = {",".join(["pad" + str(n)]) for n in range(num_pads)},
               num_samples=2048,
         ),FakeSeriesSink(
               name = "snk1",
               sink_pad_names = ("L1",),
         ),link_map= #joining together two dicts to allow for arbitrary num_pads
                {"mult:sink:pad"+str(n):"src1:src:pad"+str(n) for n in range(num_pads)}|{"snk1:sink:L1":"mult:src:H1"}
         )

    pipeline.run()

if __name__ == "__main__":
    test_tsgraph(None)

