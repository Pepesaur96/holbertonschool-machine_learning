#!/usr/bin/env python3
"""normalizes an unactivated output of a neural
network using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Z = numpy.ndarray of shape (m, n) that should
    be normalized
    n = number of features in Z
    gamma = numpy.ndarray of shape (1, n) containing the scales
    to be used for batch normalization
    beta = numpy.ndarray of shape (1, n) containing the offsets
    to be used for batch normalization
    epsilon = a small number used to avoid division by zero
    Returns: the normalized Z matrix"""
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_tilda = gamma * Z_norm + beta
    return Z_tilda
