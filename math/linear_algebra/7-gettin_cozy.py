#!/usr/bin/env python3
"""Module for the function cat_matrices2D."""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis.
    Args:
        - mat1: a list of lists of integers/floats
        - mat2: a list of lists of integers/floats
        - axis: the axis to concatenate along
    Returns:
        A new matrix representing the concatenation of mat1 and mat2.
        If the two matrices cannot be concatenated, return None.
    """
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None

    new_matrix = []
    if axis == 0:
        new_matrix = mat1 + mat2
    elif axis == 1:
        new_matrix = [mat1[i] + mat2[i] for i in range(len(mat1))]

    return new_matrix
