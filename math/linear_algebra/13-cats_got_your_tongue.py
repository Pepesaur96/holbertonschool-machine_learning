#!/usr/bin/env python3
"""Module for the function np_cat(mat1, mat2, axis=0)."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis.
    Args:
        - mat1: a numpy.ndarray
        - mat2: a numpy.ndarray
        - axis: the axis along which the matrices will be concatenated
    Returns:
        A new numpy.ndarray containing the concatenated matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
