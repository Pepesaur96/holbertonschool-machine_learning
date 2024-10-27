#!/usr/bin/env python3
"""This module contains the function that performs
forward propagation on a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """This function performs forward propagation on a deep RNN
    Args:
        rnn_cells: list of RNNCell instances of length l that will be used
                   for the forward propagation
                   l: is the number of layers
        X: numpy.ndarray of shape (t, m, i) that contains the input data
           t: is the maximum number of time steps
           m: is the batch size
           i: is the dimensionality of the data
        h_0: numpy.ndarray of shape (l, m, h) that contains the initial hidden
             state
             h: is the dimensionality of the hidden state
    Returns: H, Y
             H: numpy.ndarray containing all of the hidden states
             Y: numpy.ndarray containing all of the outputs
    """
    # Extract the inputs
    t, m, i = X.shape
    h = h_0.shape[-1]
    length = len(rnn_cells)
    # print data for visualization
    # print("-"*50)
    # print(f"t(times steps): {t}, m(batch size): {m}")
    # print(f"i(dimentionality of the data): {i}")
    # print(f"h(dimentionality of the hidden states): {h}")
    # print(f"length(lenght of th ernn list): {length}")
    # print("-"*50)

    # initialize and array to store every h_next  for every time step
    # ((t + 1),length, m, h) => this indicates the number of time steps
    H = np.zeros((t + 1, length, m, h))
    # print("-"*50)
    # print(f"H shape: {H.shape}")
    # print(f"{H}")
    # print("-"*50)

    # Initialize the output array Y with zeros. The shape is (t, m, o), where:
    # t is the number of time steps,
    # m is the batch size,
    # o is the output dimensionality of the last RNN cell.
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))
    # print("-"*50)
    # print(f"Y shape: {Y.shape}")
    # print(f"{Y}")
    # print("-"*50)

    for i in range(t):
        # calculate the hidden state for each cell
        for j in range(length):
            if i == 0:
                H[i, j] = h_0[j]
                # print data for visualization
                # print("-"*50)
                # print(f"i: {i}, j: {j}")
                # print(f"H[i, j]: {H[i, j]}")
                # print(f"H[i, j].shape: {H[i, j].shape}")
                # print("-"*50)
            if j == 0:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j], X[i])
                # print
                # print("-"*50)
                # print(f"i: {i}, j: {j}")
                # print(f"H[i + 1, j]: {H[i + 1, j]}")
                # print(f"H[i + 1, j].shape: {H[i + 1, j].shape}")
                # print(f"Y[i]: {Y[i]}")
                # print(f"Y[i].shape: {Y[i].shape}")
                # print("-"*50)
            else:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(
                    H[i, j], H[i + 1, j - 1])
                # print
                # print("-"*50)
                # print(f"i: {i}, j: {j}")
                # print(f"H[i + 1, j]: {H[i + 1, j]}")
                # print(f"H[i + 1, j].shape: {H[i + 1, j].shape}")
                # print(f"Y[i]: {Y[i]}")
                # print(f"Y[i].shape: {Y[i].shape}")
                # print("-"*50)

    return H, Y
