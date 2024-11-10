from dataclasses import dataclass

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

    def __post_init__(self):
        assert self.control and self.control in self.sink_pad_names
        super().__post_init__()
        assert len(self.sink_pads) == 2
        assert len(self.source_pads) == 1
        self.controlpad = self.sink_pad_dict["%s:sink:%s" % (self.name, self.control)]
        self.sinkpad = self.sink_pad_dict[
            "%s:sink:%s"
            % (self.name, list(set(self.sink_pad_names) - set([self.control]))[0])
        ]

    def transform(self, pad: SourcePad) -> TSFrame:
        """Gate out sub-buffers when buffers from the control pad is a gap.

        Args:
            pad:
                SourcePad, the source pad that outputs the gated data

        Returns:
            TSFrame, the output TSFrame
        """
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
