from dataclasses import dataclass
from ..base import TSTransform, TSSlices, TSFrame


@dataclass
class Gate(TSTransform):
    """
    Uses one sink pad's buffers to control the state of anothers. The control
    buffer state is defined by either being gap or not. The actual content of the
    data is ignored otherwise.

    Parameters:
    -----------
    control: str
        The name of the pad to use as a control signal

    """

    control: str = None

    def __post_init__(self):
        assert self.control is not None and self.control in self.sink_pad_names
        super().__post_init__()
        assert len(self.sink_pads) == 2
        assert len(self.source_pads) == 1
        self.controlpad = self.sink_pad_dict["%s:sink:%s" % (self.name, self.control)]
        self.sinkpad = self.sink_pad_dict[
            "%s:sink:%s"
            % (self.name, list(set(self.sink_pad_names) - set([self.control]))[0])
        ]

    def transform(self, pad):
        nongap_slices = TSSlices(
            [b.slice for b in self.preparedframes[self.controlpad] if b]
        )
        out = sorted(
            [
                b
                for bs in [
                    buf.split(nongap_slices, contiguous=True)
                    for buf in self.preparedframes[self.sinkpad]
                ]
                for b in bs
            ]
        )
        return TSFrame(buffers=out, EOS=self.at_EOS)
