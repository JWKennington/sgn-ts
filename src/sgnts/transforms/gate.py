from dataclasses import dataclass

from sgn import validator
from sgn.base import SourcePad

from sgnts.base import TSFrame, TSSlices, TSTransform


@dataclass
class Gate(TSTransform):
    """Uses one sink pad's buffers to control the state of anothers. The control buffer
    state is defined by either being gap or not. The actual content of the data is
    ignored otherwise.

    Args:
        control:
            str, the name of the pad to use as a control signal
    """

    control: str = ""

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

    def new(self, pad: SourcePad) -> TSFrame:
        nongap_slices = TSSlices(
            [b.slice for b in self.preparedframes[self.controlpad] if b]
        )
        out = sorted(
            [
                b
                for bs in [
                    buf.split(nongap_slices.search(buf.slice), contiguous=True)
                    for buf in self.preparedframes[self.sinkpad]
                ]
                for b in bs
            ]
        )
        return TSFrame(buffers=out, EOS=self.at_EOS)
