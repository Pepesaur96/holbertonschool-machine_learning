#!/usr/bin/env python3
"""normalizes (standardizes) a matrix"""
import numpy as np


def normalize(X, m, s):
    """X = numpy.ndarray of shape (d, nx) to normalize
    d = number of data points
    nx = number of features
    m = mean of all features of X
    s = standard deviation of all features of X
    Returns: normalized X matrix"""
    return (X - m) / s
