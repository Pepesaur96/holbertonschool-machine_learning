#!/usr/bin/env python3
"""Module for the function cat_arrays(arr1, arr2)."""


def cat_arrays(arr1, arr2):
    """
    Function that concatenates two arrays.
    Args:
        - arr1: a list of numbers
        - arr2: a list of numbers
    Returns:
        A new list, the concatenation of arr1 and arr2

    Example:
    arr1 = [1, 2]
    arr2 = [3, 4]
    cat_arrays(arr1, arr2) -> [1, 2, 3, 4]
    """
    return arr1 + arr2
