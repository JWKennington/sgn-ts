from enum import Enum


class Time(int, Enum):
    SECONDS = 1_000_000_000
    MILLISECONDS = 1_000_000
    MICROSECONDS = 1_000
    NANOSECONDS = 1
