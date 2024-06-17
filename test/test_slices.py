#!/usr/bin/env python3

from sgnts.base.slice_tools import *


def test_slices(capsys):

    for A, B in [
        (TSSlice(0, 3), TSSlice(2, 5)),
        (TSSlice(0, 3), TSSlice(4, 6)),
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


if __name__ == "__main__":
    test_slices(None)
