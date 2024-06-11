from dataclasses import dataclass


@dataclass
class TSSlice():
    start: int
    stop: int

    @property
    def slice(self):
        if self:
            return slice(self.start, self.stop, 1)
        else:
            return slice(-1,-1,1)

    def __and__(self, o):
        _start,_stop = max(self.start, o.start),min(self.stop, o.stop)
        return TSSlice(_start, _stop)

    def __or__(self, o):
        return TSSlice(min(self.start, o.start), max(self.stop, o.stop))

    def __bool__(self):
        return self.start < self.stop

    def __add__(self, o):
        if (self & o):
            return [self | o]
        else:
            return [self, o]

    def __sub__(self, o):
        b = self | o
        i = self & o
        return TSSlice(b.start, i.start), TSSlice(i.stop, b.stop)

    @staticmethod
    def intersection(slices):
        s = slices[0]
        for s2 in slices[1:]:
            s = s & s2
        return s
