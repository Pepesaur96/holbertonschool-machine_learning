#!/usr/bin/env python3
"""Module that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """
    Function that calculates the derivative of a polynomial
    Args:
        poly (list): A list of coefficients representing a polynomial
    Returns:
        list: A new list of coefficients representing the derivative
        of the polynomial
    """
    # Check if poly is valid
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    # Handle the case where the polynomial has no x term (it's a constant)
    if len(poly) == 1:
        return [0]

    # Initialize the derivative list
    derivative = []

    # Iterate over the coefficients, skipping the constant term (index 0)
    for power in range(1, len(poly)):
        # Calculate the derivative for each term: coeff * power
        derivative.append(poly[power] * power)

    # If the derivative is empty, it means the polynomial was a constant
    if not derivative:
        return [0]

    return derivative
