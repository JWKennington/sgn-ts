from dataclasses import dataclass
from functools import wraps

from sgn.base import SourcePad

from sgnts.base import TSFrame, TSTransform


@dataclass
class Align(TSTransform):
    """Align frames from multiple sink pads."""

    def __post_init__(self):
        assert set(self.source_pad_names) == set(self.sink_pad_names)
        super().__post_init__()
        self.pad_map = {
            p: self.sink_pad_dict["%s:snk:%s" % (self.name, p.name.split(":")[-1])]
            for p in self.source_pads
        }

    # FIXME: wraps are not playing well with mypy.  For now ignore and hope
    # that a future version of mypy will be able to handle this
    @wraps(TSTransform.new)
    def new(self, pad: SourcePad) -> TSFrame:  # type: ignore
        out = self.preparedframes[self.pad_map[pad]]
        self.preparedframes[self.pad_map[pad]] = None
        return out
