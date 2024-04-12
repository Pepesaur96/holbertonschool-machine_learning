#!/usr/bin/env python3
"""Module for the function add_matrices2D(mat1, mat2)."""


def add_matrices2D(mat1, mat2):
    """
    Function that adds two matrices element-wise.
    Args:
        - mat1: a list of lists of integers/floats
        - mat2: a list of lists of integers/floats
    Returns:
        A new matrix representing the sum of mat1 and mat2.
        If mat1 and mat2 are not the same shape, return None.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
