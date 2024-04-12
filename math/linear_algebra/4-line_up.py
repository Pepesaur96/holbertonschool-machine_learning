#!/usr/bin/env python3
"""Module for the function add_arrays(arr1, arr2)."""


def add_arrays(arr1, arr2):
    """
    Function that adds two arrays element-wise.
    Args:
        - arr1: a list of numbers
        - arr2: a list of numbers
    Returns:
        A new list, the sum of arr1 and arr2
        If arr1 and arr2 have different lengths, return None

    Example:
    arr1 = [1, 2]
    arr2 = [3, 4]
    add_arrays(arr1, arr2) -> [4, 6]
    """
    # Check for equal length of the lists
    if len(arr1) != len(arr2):
        return None

    # Perform element-wise addition
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
