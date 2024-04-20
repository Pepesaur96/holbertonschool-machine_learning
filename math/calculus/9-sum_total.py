#!/usr/bin/env python3
"""Module that calculates the sum of the squares of n"""


def summation_i_squared(n):
    """
    Function that calculates the sum of the squares of n natural numbers
    Args:
        n (int): The number of natural numbers to sum
    Returns:
        int: The sum of the squares of the first n natural numbers
    """
    # Step 1: Validate the input
    if not isinstance(n, int) or n <= 0:
        return None  # Return None if n is not a valid number

    # Step 2: Apply the formula to calculate the sum of squares
    sum_squares = (n * (n + 1) * (2 * n + 1)) // 6

    # Step 3: Return the result
    return sum_squares
