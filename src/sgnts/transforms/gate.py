from dataclasses import dataclass
from functools import wraps

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
        self.controlpad = self.sink_pad_dict["%s:snk:%s" % (self.name, self.control)]
        self.sinkpad = self.sink_pad_dict[
            "%s:snk:%s"
            % (self.name, list(set(self.sink_pad_names) - set([self.control]))[0])
        ]

    # FIXME: wraps are not playing well with mypy.  For now ignore and hope
    # that a future version of mypy will be able to handle this
    @wraps(TSTransform.new)
    def new(self, pad: SourcePad) -> TSFrame:  # type: ignore
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
