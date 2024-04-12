#!/usr/bin/env python3
"""Module for the function mat_mul(mat1, mat2)."""


def mat_mul(mat1, mat2):
    """
    Function that performs matrix multiplication.
    Args:
        - mat1: a list of lists of integers/floats
        - mat2: a list of lists of integers/floats
    Returns:
        A new matrix representing the multiplication of mat1 and mat2.
        If mat1 and mat2 cannot be multiplied, return None.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
