#!/usr/bin/env python3
"""Module for the function matrix_transpose(matrix)."""


def matrix_transpose(matrix):
    """
    Function that transposes a matrix.
    Args:
        - matrix: a list of lists whose transpose should be calculated.
    Returns:
        A list of lists, the transposed matrix.

    Example:
    matrix = [[1, 2], [3, 4], [5, 6]]
    matrix_transpose(matrix) -> [[1, 3, 5], [2, 4, 6]]
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
