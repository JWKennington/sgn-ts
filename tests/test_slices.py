#!/usr/bin/env python3

from sgnts.base.slice_tools import *


def test_slices(capsys):

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


    for slices in [TSSlices([TSSlice(0, 4), TSSlice(2, 6), TSSlice(1, 3),]), TSSlices([TSSlice(0, 4), TSSlice(2, 6), TSSlice(1, 3), TSSlice(8,10)])]:
        print ("\nslices = %s\n" % (slices,))
        print ("1.\tslices.simplify() = %s" % slices.simplify())
        print ("2.\tslices.intersection() = %s" % slices.intersection())
        print ("3.\tslices.search(TSSlice(2,4), align=True) = %s" % slices.search(TSSlice(2,4), align=True))
        print ("4.\tslices.search(TSSlice(2,4), align=False) = %s" % slices.search(TSSlice(2,4), align=False))
        print ("5.\tslices.invert(TSSlice(2,4)) = %s" % slices.invert(TSSlice(2,4)))

if __name__ == "__main__":
    test_slices(None)
