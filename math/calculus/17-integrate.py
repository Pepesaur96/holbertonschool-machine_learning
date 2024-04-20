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
    if not all(isinstance(coef, (int, float))
               for coef in poly) or not isinstance(C, (int, float)):
        return None  # Check for valid input

    if len(poly) == 0:
        return None

    integral = [C]  # Start the new list with the integration constant

    for power, coef in enumerate(poly):
        # Calculate the new coefficient for each term
        # Check if the division results in an integer
        if coef % (power + 1) == 0:
            new_coef = coef // (power + 1)
        else:
            new_coef = coef / (power + 1)

        integral.append(new_coef)

    # Remove any trailing zeros to minimize the size of the list
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
