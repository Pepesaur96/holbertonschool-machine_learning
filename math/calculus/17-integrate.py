#!/usr/bin/env python3
"""Module that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """
    Function that calculates the integral of a polynomial
    Args:
        poly (list): A list of coefficients representing a polynomial
        C (int): An integer representing the integration constant
    Returns:
        list: A new list of coefficients representing the integral
        of the polynomial
    """
    # Check if poly is valid
    if not isinstance(poly, list) or not poly:
        return None

    # Check if poly contains only integers or floats
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    # Check if C is an integer or a float
    if not isinstance(C, (int, float)):
        return None

    # Handle the zero polynomial special case
    if poly == [0]:
        return [C]

    # Calculate the integral
    integral = [C]
    for power, coeff in enumerate(poly):
        new_coeff = coeff / (power + 1)
        integral.append(int(new_coeff) if new_coeff.is_integer()
                        else new_coeff)

    # Remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
