from itertools import accumulate
import os
import shutil
from typing import Dict, Iterable

from pyrutils.itertools import run_length_encoding


def cleanup_directory(dirpath: str):
    """Remove files and sub-directories of input directory, but does not delete the directory itself.

    Credits to https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder.

    Arguments:
        dirpath - Path to directory to be cleaned up.
    """
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except Exception as e:
            print(f'Failed to delete {filepath}. Reason: {e}')


def read_dictionary(filepath: str) -> Dict[str, str]:
    """Read a dictionary from a file, where each file line is in the format 'key value'."""
    d = {}
    with open(filepath, mode='r') as f:
        for line in f:
            k, v = line.strip().split(sep=' ')
            d[k] = v
    return d


def run_length_encoding_intervals(iterable: Iterable):
    """Return a zip object over the initial (incl.) and final (excl.) indices in the rle of input iterable."""
    _, lengths = list(zip(*run_length_encoding(iterable)))
    initial_indices = [0] + list(accumulate(lengths))
    return zip(initial_indices[:-1], initial_indices[1:])
