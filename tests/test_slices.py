import numpy
import pytest
from sgnts.base.slice_tools import TIME_MAX, TIME_MIN, TSSlice, TSSlices


class TestSlice:
    """Test group for TSSlice class"""

    def test_init(self):
        """Test creating a TSSlice object"""
        slc = TSSlice(1, 2)
        assert slc[0] == 1
        assert slc[1] == 2

    def test_err_valid_min_time(self):
        """Test error from start below TIME_MIN"""
        with pytest.raises(ValueError):
            TSSlice(TIME_MIN - 1, 0)

    def test_err_valid_max_time(self):
        """Test error from stop above TIME_MAX"""
        with pytest.raises(ValueError):
            TSSlice(0, TIME_MAX + 1)

    def test_err_valid_null(self):
        """Test validity of null case, e.g. both or neither can be None"""
        with pytest.raises(ValueError):
            TSSlice(0, None)
        with pytest.raises(ValueError):
            TSSlice(None, 0)

    def test_err_valid_dtype(self):
        """Test validity"""
        with pytest.raises(ValueError):
            TSSlice(1.0, 1.0)

    def test_err_valid_ordering(self):
        """Test validity"""
        with pytest.raises(ValueError):
            TSSlice(1, 0)

    def test_slice(self):
        """Test coercion to builtin slice"""
        assert TSSlice(1, 2).slice == slice(1, 2, 1)
        assert TSSlice(None, None).slice == slice(-1, -1, 1)

    def test_comparison(self):
        """Test comparing two slices"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(2, 3)
        assert slc2 >= slc
        assert slc2 > slc
        assert slc <= slc2
        assert slc < slc2

    def test_comparison_null(self):
        """Test that null slices are handled in comparisons"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(None, None)
        assert not slc2 > slc
        assert not slc2 < slc
        assert not slc > slc2
        assert not slc < slc2

    def test_subtraction(self):
        """Test subtraction method"""
        slc = TSSlice(1, 3)
        slc2 = TSSlice(2, 3)
        s = slc2 - slc
        assert s == [TSSlice(1, 2)]

    def test_subtraction_disjoint(self):
        """Test subtraction method"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(3, 4)
        s = slc2 - slc
        assert s == []

    def test_union(self):
        """Test union method"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(2, 3)
        u = slc | slc2
        assert u == TSSlice(1, 3)

    def test_union_null(self):
        """Test that a null slice is propagated in union"""
        slc = TSSlice(1, 2)
        slc2 = TSSlice(None, None)
        u = slc | slc2
        assert u == slc2

    def test_isfinite(self):
        """Test isfinite method"""
        slc = TSSlice(1, 2)
        assert slc.isfinite()
        slc = TSSlice(None, None)
        assert not slc.isfinite()

    def test_bool(self):
        """Test boolean coercion"""
        assert not TSSlice(None, None).isfinite()
        assert TSSlice(1, 2)

    def test_index_numpy(self):
        """Test compatibility indexing numpy arrays with a TSSlice object"""
        data = numpy.array([1, 2, 3, 4, 5])
        slc = TSSlice(1, 3)
        res = data[slc.slice]
        numpy.testing.assert_almost_equal(res, numpy.array([2, 3]))


class TestTSSlices:
    """Test group for TSSlices class"""

    def test_valid_slices(self):
        slcs = TSSlices([TSSlice(1, 2), TSSlice(3, 4)])
        assert slcs.invert(TSSlice(0, 5)) == TSSlices(
            slices=[
                TSSlice(start=0, stop=1),
                TSSlice(start=2, stop=3),
                TSSlice(start=4, stop=5),
            ]
        )

    def test_search(self):
        """Test search method for TSSlices"""
        slcs = TSSlices([TSSlice(1, 2), TSSlice(3, 4)])
        assert slcs.search(TSSlice(0, 5)) == TSSlices(
            slices=[TSSlice(start=1, stop=2), TSSlice(start=3, stop=4)]
        )
        assert slcs.search(TSSlice(-1, 0)) == TSSlices(slices=[])

    def test_search_null_slice(self):
        """Test search method for TSSlices"""
        slcs = TSSlices([TSSlice(TIME_MIN, TIME_MAX)])
        assert slcs.search(TSSlice(0, 5)) == TSSlices(slices=[TSSlice(start=0, stop=5)])

    @pytest.mark.skip(reason="Shouldn't use capsys")
    def test_slices(self, capsys):

        for A, B in [
            (TSSlice(0, 3), TSSlice(2, 5)),
            (TSSlice(0, 3), TSSlice(4, 6)),
            (TSSlice(0, 3), TSSlice(1, 2)),
            (TSSlice(0, 3), TSSlice(1, 3)),
            (TSSlice(0, 3), TSSlice(None, None)),
        ]:
            print("\nA: %s\nB: %s\n" % (A, B))
            print("1.\tTrue if A else False:", True if A else False)
            print("2.\tTrue if B else False:", True if B else False)
            print("3.\tA>B:", A > B)
            print("4.\tB>A:", B > A)
            print("5.\tA&B:", A & B)
            print("6.\tB&A:", B & A)
            print("7.\tA|B:", A | B)
            print("8.\tB|A:", B | A)
            print("9.\tA+B:", A + B)
            print("10.\tB+A:", B + A)
            print("11.\tA-B:", A - B)
            print("12.\tB-A:", B - A)

        for slices in [
            TSSlices(
                [
                    TSSlice(0, 4),
                    TSSlice(2, 6),
                    TSSlice(1, 3),
                ]
            ),
            TSSlices([TSSlice(0, 4), TSSlice(2, 6), TSSlice(1, 3), TSSlice(8, 10)]),
        ]:
            print("\nslices = %s\n" % (slices,))
            print("1.\tslices.simplify() = %s" % slices.simplify())
            print("2.\tslices.intersection() = %s" % slices.intersection())
            print(
                "3.\tslices.search(TSSlice(2,4), align=True) = %s"
                % slices.search(TSSlice(2, 4), align=True)
            )
            print(
                "4.\tslices.search(TSSlice(2,4), align=False) = %s"
                % slices.search(TSSlice(2, 4), align=False)
            )
            print("5.\tslices.invert(TSSlice(2,4)) = %s" % slices.invert(TSSlice(2, 4)))

    def test_search_unaligned(self):
        """Test searching for a slice in slices"""
        slc = TSSlice(1, 2)
        slcs = TSSlices([TSSlice(1, 3), TSSlice(3, 4)])
        res = slcs.search(slc, align=False)
        assert isinstance(res, TSSlices)
        assert res == TSSlices([TSSlice(1, 3)])
