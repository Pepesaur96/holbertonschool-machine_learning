#!/usr/bin/env python3
"""Module for the function np_elementwise(mat1, mat2)."""


def np_elementwise(mat1, mat2):
    """
    Function that performs element-wise addition, subtraction, multiplication,
    and division.
    Args:
        - mat1: a numpy.ndarray
        - mat2: a numpy.ndarray
    Returns:
        A tuple containing the element-wise sum, difference, product, and
        quotient, respectively.
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
