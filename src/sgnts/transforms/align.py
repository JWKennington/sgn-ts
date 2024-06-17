from dataclasses import dataclass
from ..base import TSTransform


@dataclass
class Align(TSTransform):
    def __post_init__(self):
        assert set(self.source_pad_names) == set(self.sink_pad_names)
        super().__post_init__()
        self.pad_map = {
            p: self.sink_pad_dict["%s:sink:%s" % (self.name, p.name.split(":")[-1])]
            for p in self.source_pads
        }

    def transform(self, pad):
        out = self.preparedframes[self.pad_map[pad]]
        self.preparedframes[self.pad_map[pad]] = None
        return out
