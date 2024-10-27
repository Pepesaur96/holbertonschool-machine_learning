#!/usr/bin/env python3
"""This module contains the rnn function"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """This function preforms a forward propagation on a simple RNN
    Args:
        rnn_cell: is an instance of RNNCell that will be used for the forward
                  propagation
        X: is the data to be used, given as a numpy.ndarray of shape (t, m, i)
           t is the maximum number of time steps
           m is the batch size
           i is the dimensionality of the data
        h_0: is the initial hidden state, given as a numpy.ndarray of
        shape (m, h)
             h is the dimensionality of the hidden state
    Returns: H, Y
             H: is a numpy.ndarray containing all of the hidden states
             Y: is a numpy.ndarray containing all of the outputs
    """
    # Extract the shapes from the input data
    t = X.shape[0]
    m = X.shape[1]
    # print("-"*50)
    # print("t: ", t)
    # print("m: ", m)
    # print("i: ", i)
    # print("-"*50)
    # Extract the shape of the hidden state
    h = h_0.shape[1]
    # print("-"*50)
    # print("h: ", h)
    # print("-"*50)

    # initialez and empty list to store the hidden states
    # t+1: This represents the number of time steps plus one.
    # The extra one is to include
    # the initial hidden state h_0. So, if you have t time steps,
    # you need t+1 slots to
    # store all hidden states from h_0 to h_t.
    # m: This is the batch size, representing the number of
    # sequences being processed in parallel.
    # h: This is the dimensionality of the hidden state,
    # representing the number of
    # features in the hidden state.
    H = np.zeros((t + 1, m, h))
    # print("-"*50)
    # print("H shape: ", H.shape)
    # print(H)
    # print("-"*50)
    # initialez and empty list to store the outputs

    # Initialize Y to store the outputs for each time step
    # Shape: (t, m, o)
    # t: number of time steps
    # m: batch size
    # o: output dimensionality (determined by the second dimension
    # of rnn_cell.Wy)
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    # print("-"*50)
    # print("Y shape: ", Y.shape)
    # print(Y)
    # print("-"*50)

    for i in range(t):
        # Verify if its the first time step
        if i == 0:
            H[i] = h_0
            # print("-"*50)
            # print("H[0]: ", H[0])
            # print("-"*50)
        # Compute the next hidden state and output

        # Perform a forward pass using the RNN cell for the current time step
        # H[i+1]: the next hidden state
        # Y[i]: the output at the current time step
        # H[i]: the previous hidden state
        # X[i]: the input at the current time step
        H[i + 1], Y[i] = rnn_cell.forward(H[i], X[i])
        # print("-"*50)
        # print(f"H[{i+1}]: ", H[i+1])
        # print(f"Y[{i}]: ", Y[i])
        # print("-"*50)

    return H, Y
