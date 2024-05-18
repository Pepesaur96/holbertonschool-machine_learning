#!/usr/bin/env python3
"""sets up the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """alpha = learning rate
    beta1 = weight used for the first moment
    beta2 = weight use for the second moment
    epsilon = small number to avoid division by zero
    var = numpy.ndarray containing the variable to be updated
    grad = numpy.ndarray containing the gradient of var
    v = previous first moment of var
    s = previous second moment of var
    t = time step used to calculate the bias-corrected moments
    Returns: the updated variable, the new first moment,
    and the new second moment, respectively"""
    v = beta1 * v + (1 - beta1) * grad
    v_corrected = v / (1 - beta1 ** t)
    s = beta2 * s + (1 - beta2) * grad ** 2
    s_corrected = s / (1 - beta2 ** t)
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
