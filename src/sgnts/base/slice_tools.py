"""Utilities for working with intervals of time"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy

# Define beginning / end of time as min / max int 64
TIME_MIN = int(numpy.int64(numpy.iinfo(numpy.int64).min))
TIME_MAX = int(numpy.int64(numpy.iinfo(numpy.int64).max))


@dataclass
class TSSlice:
    """A class to support operations on an ordered tuple of integers start, stop.

    Args:
        start:
            int, The start of the TSSlice
        stop:
            int, The stop of the TSSlice
    """

    start: int = TIME_MIN
    stop: int = TIME_MAX

    def __post_init__(self):
        if (self.start is None and self.stop is not None) or (
            self.stop is None and self.start is not None
        ):
            raise ValueError("if one of start or stop is None, both must be")
        if self.start is not None:
            if not isinstance(self.start, (int, numpy.int64)) or not isinstance(
                self.stop, (int, numpy.int64)
            ):
                raise ValueError("if not None, start and stop must be integers")
            if not (self.stop >= self.start):
                raise ValueError("stop must be greater than or equal to start")

            if self.start < TIME_MIN:
                raise ValueError(f"start must be greater than {TIME_MIN}")
            if self.stop > TIME_MAX:
                raise ValueError(f"stop must be less than {TIME_MAX}")

    @property
    def slice(self):
        """Convert to a python slice object with a stride of 1."""
        if self:
            return slice(self.start, self.stop, 1)
        else:
            return slice(-1, -1, 1)

    def __getitem__(self, item):
        assert item in (0, 1)
        if item == 0:
            return self.start
        else:
            return self.stop

    def __and__(self, o):
        """Find the intersection of two TSSlices

        Examples:
            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=2, stop=5)
            >>> A&B
            TSSlice(start=2, stop=3)
            >>> B&A
            TSSlice(start=2, stop=3)

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=4, stop=6)
            >>> A&B
            TSSlice(start=None, stop=None)
            >>> B&A
            TSSlice(start=None, stop=None)

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=None, stop=None)
            >>> A&B
            TSSlice(start=None, stop=None)
            >>> B&A
            TSSlice(start=None, stop=None)
        """
        if self.start is None or self.stop is None or o.start is None or o.stop is None:
            return TSSlice(None, None)
        _start, _stop = max(self.start, o.start), min(self.stop, o.stop)
        if _start > _stop:
            return TSSlice(None, None)
        return TSSlice(_start, _stop)

    def __or__(self, o):
        """Find the TSSlice that spans both self and o.

        Examples:
            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=2, stop=5)
            >>> A|B
            TSSlice(start=0, stop=5)
            >>> B|A
            TSSlice(start=0, stop=5)

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=4, stop=6)
            >>> A|B
            TSSlice(start=0, stop=6)
            >>> B|A
            TSSlice(start=0, stop=6)

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=None, stop=None)
            >>> A|B
            TSSlice(start=None, stop=None)
            >>> B|A
            TSSlice(start=None, stop=None)
        """
        if self.start is None or self.stop is None or o.start is None or o.stop is None:
            return TSSlice(None, None)
        return TSSlice(min(self.start, o.start), max(self.stop, o.stop))

    def __bool__(self):
        """Check the truth value of this TSSlice.

        Examples:
            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=2, stop=5)
            >>> True if A else False
            True
            >>> True if B else False
            True

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=4, stop=6)
            >>> True if A else False
            True
            >>> True if B else False
            True

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=None, stop=None)
            >>> True if A else False
            True
            >>> True if B else False
            False
        """

        if self.start is None:
            assert self.stop is None
        if self.stop is None:
            assert self.start is None
        if self.start is None:
            return False
        else:
            return True

    def __add__(self, o):
        """Add two TSSlices together producing a single TSSlice if they intersect
        otherwise returning each in a list.

        Examples:
            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=2, stop=5)
            >>> A+B
            [TSSlice(start=0, stop=5)]
            >>> B+A
            [TSSlice(start=0, stop=5)]

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=4, stop=6)
            >>> A+B
            [TSSlice(start=0, stop=3), TSSlice(start=4, stop=6)]
            >>> B+A
            [TSSlice(start=0, stop=3), TSSlice(start=4, stop=6)]

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=None, stop=None)
            >>> A+B
            [TSSlice(start=0, stop=3), TSSlice(start=None, stop=None)]
            >>> B+A
            [TSSlice(start=None, stop=None), TSSlice(start=0, stop=3)]
        """
        if self & o:
            return [self | o]
        else:
            return sorted([self, o])

    def __gt__(self, o):
        """Check if a slice is greater than another slice.

        Examples:
            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=2, stop=5)
            >>> A>B
            False
            >>> B>A
            True

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=4, stop=6)
            >>> A>B
            False
            >>> B>A
            True

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=None, stop=None)
            >>> A>B
            False
            >>> B>A
            False
        """
        if self.start is None or self.stop is None or o.start is None or o.stop is None:
            return False
        return self.start > o.start and self.stop > o.stop

    def __lt__(self, o):
        if self.start is None or self.stop is None or o.start is None or o.stop is None:
            return False
        return self.start < o.start and self.stop < o.stop

    def __ge__(self, o):
        return self.start >= o.start and self.stop >= o.stop

    def __le__(self, o):
        return self.start <= o.start and self.stop <= o.stop

    def __sub__(self, o):
        """Find the difference of two overlapping slices, it not overlapping return an
        empty list.

        Examples:
            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=2, stop=5)
            >>> A-B
            [TSSlice(start=0, stop=2), TSSlice(start=3, stop=5)]
            >>> B-A
            [TSSlice(start=0, stop=2), TSSlice(start=3, stop=5)]

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=4, stop=6)
            >>> A-B
            []
            >>> B-A
            []

            >>> A = TSSlice(start=0, stop=3)
            >>> B = TSSlice(start=None, stop=None)
            >>> A-B
            []
            >>> B-A
            []
        """
        b = self | o
        i = self & o
        if not b or not i:
            return []
        out = [TSSlice(b.start, i.start), TSSlice(i.stop, b.stop)]
        return sorted(o for o in out if o.isfinite())

    def __contains__(self, o):
        return o.start >= self.start and o.stop <= self.stop

    def split(self, o: int):
        """Split the slice with the given boundary value.

        Args:
            o:
                int, the boundary to split the tsslice
        """
        assert self.start <= o < self.stop
        return [TSSlice(self.start, o), TSSlice(o, self.stop)]

    def isfinite(self):
        if not self:
            return False
        else:
            return self.stop > self.start


@dataclass
class TSSlices:
    """A class that holds a list of TSSlice objects and defines some operations on them.

    Args:
        slices:
            list, A list of TSSlice objects. These will be stored in a sorted order and
            are assumed to be immutable
    """

    slices: list

    def __post_init__(self):
        self.slices = sorted(self.slices)

    def __iadd__(self, other):
        """Inplace add (a new instance is made though)"""
        return TSSlices(self.slices + other.slices)

    def simplify(self):
        """Merge overlapping slices and return a new instance of TSSlices.

        Examples:
            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6)])
            >>> slices.simplify()
            TSSlices(slices=[TSSlice(start=0, stop=6)])

            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6),
            ...     TSSlice(start=8, stop=10)])
            >>> slices.simplify()
            TSSlices(slices=[TSSlice(start=0, stop=6), TSSlice(start=8, stop=10)])
        """

        out = self.slices[0:1].copy()
        for s in self.slices[1:]:
            this = s + out[-1]
            if len(this) == 2:
                out.append(this[-1])
            else:
                out[-1] = this[0]
        return TSSlices(out)

    def intersection(self):
        """Find the intersection of all slices. Might be empty.

        Examples:
            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6)])
            >>> slices.intersection()
            TSSlice(start=2, stop=3)

            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6),
            ...     TSSlice(start=8, stop=10)])
            >>> slices.intersection()
            TSSlice(start=None, stop=None)
        """
        s = TSSlice(self.slices[0].start, self.slices[0].stop)
        for s2 in self.slices[1:]:
            s = s & s2
        return s

    def search(self, tsslice: TSSlice, align: bool = True):
        """Search for the set of TSSlices that overlap wtih tsslice. If align=True the
        returned slices will be truncated to exactly fall within tsslice.

        Args:
            tsslice:
                TSSlice, the tsslice to search for overlap with
            align:
                bool, whether to align the tsslices

        Examples:
            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6)])
            >>> slices.search(TSSlice(2,4), align=True)
            TSSlices(slices=[TSSlice(start=2, stop=4), TSSlice(start=2, stop=3),
                TSSlice(start=2, stop=4)])
            >>> slices.search(TSSlice(2,4), align=False)
            TSSlices(slices=[TSSlice(start=0, stop=4), TSSlice(start=1, stop=3),
                TSSlice(start=2, stop=6)])

            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6),
            ...     TSSlice(start=8, stop=10)])
            >>> slices.search(TSSlice(2,4), align=True)
            TSSlices(slices=[TSSlice(start=2, stop=4), TSSlice(start=2, stop=3),
                TSSlice(start=2, stop=4)])
            >>> slices.search(TSSlice(2,4), align=False)
            TSSlices(slices=[TSSlice(start=0, stop=4), TSSlice(start=1, stop=3),
                TSSlice(start=2, stop=6)])
        """

        startix = bisect.bisect_left(self.slices, TSSlice(tsslice.start, tsslice.start))
        stopix = bisect.bisect_right(self.slices, TSSlice(tsslice.stop, tsslice.stop))
        if not align:
            return TSSlices(self.slices[startix:stopix])
        else:
            out = []
            for s in self.slices[startix:stopix]:
                o = s & tsslice
                if o.isfinite():
                    out.append(o)
            return TSSlices(out)

    def invert(self, boundary_slice: TSSlice):
        """Within boundary_slice, return an inverted set of TSSlice's.

        Args:
            boundary_slice:
                TSSlice, the boundary to invert the TSSlices

        Examples:
            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6)])
            >>> slices.invert(TSSlice(2,4))
            TSSlices(slices=[])

            >>> slices = TSSlices(slices=[TSSlice(start=0, stop=4),
            ...     TSSlice(start=1, stop=3), TSSlice(start=2, stop=6),
            ...     TSSlice(start=8, stop=10)])
            >>> slices.invert(TSSlice(2,4))
            TSSlices(slices=[TSSlice(start=6, stop=8)])
        """

        if len(self.slices) == 0:
            return TSSlices([TSSlice(boundary_slice.start, boundary_slice.stop)])
        _slices = self.simplify().slices
        out = []
        if boundary_slice.start < _slices[0].start:
            out.append(TSSlice(boundary_slice.start, _slices[0].start))
        out.extend(
            [TSSlice(s1.stop, s2.start) for (s1, s2) in zip(_slices[:-1], _slices[1:])]
        )
        if boundary_slice.stop > _slices[-1].stop:
            out.append(TSSlice(_slices[-1].stop, boundary_slice.stop))
        return TSSlices(out)
