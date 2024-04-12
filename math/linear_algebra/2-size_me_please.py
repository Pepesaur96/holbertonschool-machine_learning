#!/usr/bin/env python3
"""Module for the function matrix_shape(matrix)."""


def matrix_shape(matrix):
    """
    Function that calculates the shape of a matrix.
    Args:
        - matrix: a matrix to calculate the shape of.
    Returns:
        A list of integers, the shape of the matrix.

    Example:
    matrix = [[1, 2], [3, 4], [5, 6]]
    matrix_shape(matrix) -> [3, 2]
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]  # Dive into the next dimension
    return shape
