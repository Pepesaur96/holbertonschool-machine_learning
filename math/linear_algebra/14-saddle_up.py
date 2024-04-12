#!/usr/bin/env python3
"""Module for the function np_matmul(mat1, mat2)."""
import numpy as np


def np_matmul(mat1, mat2):
    """
    Function that performs matrix multiplication.
    Args:
        - mat1: a numpy.ndarray
        - mat2: a numpy.ndarray
    Returns:
        A numpy.ndarray containing the result of the multiplication
    """
    return np.matmul(mat1, mat2)
