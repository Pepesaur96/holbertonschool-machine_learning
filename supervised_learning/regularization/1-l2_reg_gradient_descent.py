#!/usr/bin/env python3
"""This module updates the weights and biases of a neural network using
gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Parameters:
    Y: numpy.ndarray - a one-hot array with shape (classes, m) containing
    the correct labels for the data.
    weights: dict - a dictionary of the weights and biases of the
    neural network.
    cache: dict - a dictionary of the outputs of each layer of the
    neural network.
    alpha: float - the learning rate.
    lambtha: float - the L2 regularization parameter.
    L: int - the number of layers of the network.

    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation.
    The weights and biases of the network should be updated in place.
    """
    # Get the number of data points
    m = Y.shape[1]

    # Calculate the derivative of the cost with respect to
    # the output of the last layer
    dz = cache['A' + str(L)] - Y

    # Iterate from the last layer to the first
    for i in range(L, 0, -1):
        # Get the activations of the previous layer
        A_prev = cache['A' + str(i-1)]

        # Calculate the derivative of the cost with respect to the
        # weights, adding the L2 regularization term
        dw = \
            (1/m) * np.dot(dz, A_prev.T) + \
            ((lambtha/m) * weights['W' + str(i)])

        # Calculate the derivative of the cost with respect to the biases
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)

        # Get the weights of the current layer
        W = weights['W' + str(i)]

        # Calculate the derivative of the cost with respect to
        # the output of the previous layer
        dz = np.dot(W.T, dz) * (1 - np.power(A_prev, 2))

        # Update the weights and biases in place
        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db
