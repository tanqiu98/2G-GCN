from itertools import groupby
from typing import Iterable


def negative_range(n: int):
    """Given a positive integer n, return a range object that iterates through -1, -2, ..., -n.

    If input parameter n is non-positive, ValueError is raised.
    """
    if n < 1:
        raise ValueError(f'Input parameter n must be positive, but {n} was given as input.')
    return range(-1, -n - 1, -1)


def run_length_encoding(iterable: Iterable):
    """Make an iterator that returns a label and the number of appearances in its run-length encoding."""
    for k, v in groupby(iterable):
        yield k, len(list(v))
