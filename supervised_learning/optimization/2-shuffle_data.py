#!/usr/bin/env python3
"""shuffles the data points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """X = numpy.ndarray of shape (m, nx) to shuffle
    m = number of data points
    nx = number of features in X
    Y = second numpy.ndarray of shape (m, ny) to shuffle
    m = same number of data points as in X
    ny = number of features in Y
    Returns: the shuffled X and Y matrices"""
    m = X.shape[0]
    idx = np.random.permutation(m)
    return X[idx], Y[idx]
