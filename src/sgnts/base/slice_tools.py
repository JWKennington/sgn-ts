"""Utilities for working with intervals of time"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy

from sgnts.base.offset import Offset, TimeUnits

# Define beginning / end of time as min / max int 64
TIME_MIN = int(numpy.int64(numpy.iinfo(numpy.int64).min))
TIME_MAX = int(numpy.int64(numpy.iinfo(numpy.int64).max))


@dataclass
class TSSlice:
    """A class to support operations on an ordered tuple of integers start, stop.

    Args:
        start:
            Union[int, float], The start of the TSSlice. Must be int
            unless units=SECONDS.
        stop:
            Union[int, float], The stop of the TSSlice. Must be int
            unless units=SECONDS.
        units:
            TimeUnits, The unit of time for start and stop. Defaults to OFFSETS.
    """

    start: Union[int, float] = TIME_MIN
    stop: Union[int, float] = TIME_MAX
    units: TimeUnits = TimeUnits.OFFSETS

    def __post_init__(self):
        if (self.start is None and self.stop is not None) or (
            self.stop is None and self.start is not None
        ):
            raise ValueError("if one of start or stop is None, both must be")

        if self.start is not None:
            # 2. Validate Types based on Units
            if self.units == TimeUnits.SECONDS:
                # Allow floats, ints, or numpy numerics for SECONDS
                if not isinstance(
                    self.start, (int, float, numpy.number)
                ) or not isinstance(self.stop, (int, float, numpy.number)):
                    raise ValueError("start and stop must be numeric for SECONDS")
            else:
                # OFFSETS, NANOSECONDS, SAMPLES must be integers
                if not isinstance(self.start, (int, numpy.integer)) or not isinstance(
                    self.stop, (int, numpy.integer)
                ):
                    raise ValueError(
                        f"start and stop must be integers for unit {self.units}"
                    )

            # 3. Validate Order
            if not (self.stop >= self.start):
                raise ValueError("stop must be greater than or equal to start")

            # 4. Validate Bounds (Skip for seconds as float bounds/precision differ)
            if self.units != TimeUnits.SECONDS:
                if self.start < TIME_MIN:
                    raise ValueError(f"start must be greater than {TIME_MIN}")
                if self.stop > TIME_MAX:
                    raise ValueError(f"stop must be less than {TIME_MAX}")

    def convert(
        self,
        target_unit: TimeUnits,
        from_sample_rate: Optional[int] = None,
        to_sample_rate: Optional[int] = None,
    ) -> "TSSlice":
        """Convert this TSSlice to a new TSSlice with different units.

        Args:
            target_unit: The unit to convert to.
            from_sample_rate: Required if converting FROM samples.
            to_sample_rate: Required if converting TO samples.

        Returns:
            A new TSSlice instance in the target units.
        """
        # Handle infinite/empty slices simply
        if self.start is None:
            return TSSlice(None, None, units=target_unit)

        # Delegate the calculation strictly to Offset.convert
        # Offset.convert handles validation of whether rates are required/forbidden
        new_start = Offset.convert(
            self.start,
            from_unit=self.units,
            to_unit=target_unit,
            from_sample_rate=from_sample_rate,
            to_sample_rate=to_sample_rate,
        )

        new_stop = Offset.convert(
            self.stop,
            from_unit=self.units,
            to_unit=target_unit,
            from_sample_rate=from_sample_rate,
            to_sample_rate=to_sample_rate,
        )

        return TSSlice(new_start, new_stop, units=target_unit)

    @property
    def slice(self):
        """Convert to a python slice object with a stride of 1."""
        if self.units == TimeUnits.SECONDS:
            raise TypeError("Cannot create python slice from SECONDS units (float).")

        if self:
            return slice(self.start, self.stop, 1)
        else:
            return slice(-1, -1, 1)

    def _ensure_compatible(self, other: "TSSlice"):
        """Ensure two slices have compatible units before boolean operations."""
        if self.units != other.units:
            raise ValueError(
                f"Cannot operate on mixed units: {self.units} vs {other.units}. "
                "Convert one slice first."
            )

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
        self._ensure_compatible(o)
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
        self._ensure_compatible(o)
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
        self._ensure_compatible(o)
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
        self._ensure_compatible(o)
        if self.start is None or self.stop is None or o.start is None or o.stop is None:
            return False
        return self.start > o.start and self.stop > o.stop

    def __lt__(self, o):
        self._ensure_compatible(o)
        if self.start is None or self.stop is None or o.start is None or o.stop is None:
            return False
        return self.start < o.start and self.stop < o.stop

    def __ge__(self, o):
        self._ensure_compatible(o)
        return self.start >= o.start and self.stop >= o.stop

    def __le__(self, o):
        self._ensure_compatible(o)
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
        self._ensure_compatible(o)
        b = self | o
        i = self & o
        if not b or not i:
            return []
        out = [TSSlice(b.start, i.start), TSSlice(i.stop, b.stop)]
        return sorted(o for o in out if o.isfinite())

    def __contains__(self, o):
        self._ensure_compatible(o)
        return o.start >= self.start and o.stop <= self.stop

    def split(self, o: int):
        """Split the slice with the given boundary value.

        Args:
            o:
                int, the boundary to split the tsslice
        """
        assert self.start <= o < self.stop
        return [
            TSSlice(self.start, o, units=self.units),
            TSSlice(o, self.stop, units=self.units),
        ]

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
        # Validate unit consistency
        if len(self.slices) > 0:
            base_unit = self.slices[0].units
            for s in self.slices[1:]:
                if s.units != base_unit:
                    raise ValueError("All slices in TSSlices must have the same units.")

        self.slices = sorted(self.slices)

    def __iadd__(self, other):
        """Inplace add (a new instance is made though)"""
        return TSSlices(self.slices + other.slices)

    def __iter__(self):
        return iter(self.slices)

    def convert(
        self,
        target_unit: TimeUnits,
        from_sample_rate: Optional[int] = None,
        to_sample_rate: Optional[int] = None,
    ) -> "TSSlices":
        """Convert all contained slices to a new unit."""
        new_slices = [
            s.convert(target_unit, from_sample_rate, to_sample_rate)
            for s in self.slices
        ]
        return TSSlices(new_slices)

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

    @property
    def slice(self):
        "Provide a slice that corresponds to the start and end offset"
        return TSSlice(
            self.slices[0].start,
            self.slices[-1].stop,
            units=self.slices[0].units,
        )

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
        # Safety check: Search requires matching units
        if self.slices and tsslice.units != self.slices[0].units:
            raise ValueError(
                f"Search slice units ({tsslice.units}) do not match TSSlices "
                f"units ({self.slices[0].units}). "
                "Convert search slice manually."
            )

        startix = bisect.bisect_left(
            self.slices, TSSlice(tsslice.start, tsslice.start, units=tsslice.units)
        )
        stopix = bisect.bisect_right(
            self.slices, TSSlice(tsslice.stop, tsslice.stop, units=tsslice.units)
        )
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
        # Check units
        if self.slices and boundary_slice.units != self.slices[0].units:
            raise ValueError(
                f"Boundary slice units ({boundary_slice.units}) do not match "
                f"TSSlices units ({self.slices[0].units}). "
                "Convert boundary slice manually."
            )

        if len(self.slices) == 0:
            return TSSlices(
                [
                    TSSlice(
                        boundary_slice.start,
                        boundary_slice.stop,
                        units=boundary_slice.units,
                    )
                ]
            )
        _slices = self.simplify().slices
        out = []
        if boundary_slice.start < _slices[0].start:
            out.append(
                TSSlice(
                    boundary_slice.start, _slices[0].start, units=boundary_slice.units
                )
            )
        out.extend(
            [
                TSSlice(s1.stop, s2.start, units=boundary_slice.units)
                for (s1, s2) in zip(_slices[:-1], _slices[1:])
            ]
        )
        if boundary_slice.stop > _slices[-1].stop:
            out.append(
                TSSlice(
                    _slices[-1].stop, boundary_slice.stop, units=boundary_slice.units
                )
            )
        return TSSlices(out)

    @classmethod
    def intersection_of_multiple(cls, tsslices_list: list["TSSlices"]) -> "TSSlices":
        """Find the intersection of multiple TSSlices objects.

        This method computes regions that are present in ALL input TSSlices.
        It's useful for finding common valid data regions across multiple streams.

        Args:
            tsslices_list: List of TSSlices objects to intersect

        Returns:
            TSSlices containing only regions present in all inputs

        Examples:
            >>> slices1 = TSSlices([TSSlice(0, 10), TSSlice(20, 30)])
            >>> slices2 = TSSlices([TSSlice(5, 15), TSSlice(25, 35)])
            >>> slices3 = TSSlices([TSSlice(7, 12), TSSlice(22, 28)])
            >>> TSSlices.intersection_of_multiple([slices1, slices2, slices3])
            TSSlices(slices=[TSSlice(start=7, stop=10), TSSlice(start=25, stop=28)])

            >>> # Empty case
            >>> TSSlices.intersection_of_multiple([])
            TSSlices(slices=[])

            >>> # No overlap case
            >>> slices1 = TSSlices([TSSlice(0, 10)])
            >>> slices2 = TSSlices([TSSlice(20, 30)])
            >>> TSSlices.intersection_of_multiple([slices1, slices2])
            TSSlices(slices=[])
        """
        if not tsslices_list:
            return cls([])

        if len(tsslices_list) == 1:
            return tsslices_list[0]

        # Start with first set of slices
        intersection = tsslices_list[0]

        # Intersect with each subsequent set
        for slices in tsslices_list[1:]:
            new_intersection = []
            for int_slice in intersection.slices:
                for curr_slice in slices.slices:
                    overlap = int_slice & curr_slice
                    if overlap and overlap.isfinite():
                        new_intersection.append(overlap)
            intersection = cls(new_intersection)

            # Early exit if no intersection
            if not intersection.slices:
                return cls([])

        # Optionally simplify to merge overlapping slices
        return intersection.simplify() if intersection.slices else intersection

    def align_to_rate(self, target_rate: int) -> TSSlices:
        """Align TSSlices to integer sample boundaries at a target sample rate.
        Forces conversion to OFFSETS to perform alignment logic.
        """
        assert (
            target_rate in Offset.ALLOWED_RATES
        ), f"Target rate {target_rate} not in ALLOWED_RATES: {Offset.ALLOWED_RATES}"

        # 1. Determine the source slices in OFFSETS units
        if self.slices and self.slices[0].units != TimeUnits.OFFSETS:
            # We can't auto-convert SAMPLES because we don't know the source rate here
            if self.slices[0].units == TimeUnits.SAMPLES:
                raise ValueError(
                    "Cannot auto-align from SAMPLES units. Convert to OFFSETS first."
                )
            slices_to_process = self.convert(TimeUnits.OFFSETS).slices
        else:
            slices_to_process = self.slices

        # 2. Perform alignment logic
        offset_per_sample = Offset.MAX_RATE // target_rate
        aligned_slices = []

        for slc in slices_to_process:
            if not slc or not slc.isfinite():
                continue

            start_samples = slc.start / offset_per_sample
            stop_samples = slc.stop / offset_per_sample

            aligned_start_samples = math.ceil(start_samples)
            aligned_stop_samples = math.floor(stop_samples)

            if aligned_start_samples < aligned_stop_samples:
                aligned_start = aligned_start_samples * offset_per_sample
                aligned_stop = aligned_stop_samples * offset_per_sample
                aligned_slices.append(
                    TSSlice(aligned_start, aligned_stop, units=TimeUnits.OFFSETS)
                )

        return TSSlices(aligned_slices)
