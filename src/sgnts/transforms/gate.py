from dataclasses import dataclass

from sgn import validator
from sgn.base import SinkPad

from sgnts.base import TSCollectFrame, TSFrame, TSSlices, TSTransform
from sgnts.decorators import transform


@dataclass(kw_only=True)
class Gate(TSTransform):
    """Uses one sink pad's buffers to control the state of anothers. The control buffer
    state is defined by either being gap or not. The actual content of the data is
    ignored otherwise.

    Args:
        control:
            str, the name of the pad to use as a control signal
    """

    control: str

    def configure(self) -> None:
        self.controlpad = self.snks[self.control]
        data_pad_name = list(set(self.sink_pad_names) - set([self.control]))[0]
        self.sinkpad = self.snks[data_pad_name]
        self.source_pad = self.source_pads[0]

    @validator.num_pads(sink_pads=2, source_pads=1)
    def validate(self) -> None:
        assert self.control and self.control in self.sink_pad_names, (
            f"Control pad '{self.control}' must be specified and exist "
            f"in sink_pad_names: {self.sink_pad_names}"
        )

    @transform.many_to_one
    def process(
        self, input_frames: dict[SinkPad, TSFrame], output_frame: TSCollectFrame
    ) -> None:
        """Gate input based on control pad."""
        nongap_slices = TSSlices([b.slice for b in input_frames[self.controlpad] if b])
        bufs = sorted(
            [
                b
                for bs in [
                    buf.split(nongap_slices.search(buf.slice), contiguous=True)
                    for buf in input_frames[self.sinkpad]
                ]
                for b in bs
            ]
        )
        output_frame.extend(bufs)
