#!/usr/bin/env python3
"""This modlue contains the RNNCell class"""

import numpy as np


class RNNCell:
    """This class reperesents a cell of a simple Rnn"""

    def __init__(self, i, h, o):
        """This is the class constructor
        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs
        """
        # print("-"*50)
        # print("Imput data to constructor")
        # print("i: ", i)
        # print("h: ", h)
        # print("o: ", o)
        # print("-"*50)
        # Weights
        # generate random values for the weights
        # for Wh (i + h, h) set the sahep to 25, 15

        # In an RNN cell, the weight matrix Wh is used to combine
        # both the input data and the previous hidden state to compute
        # the new hidden state. Therefore, the weight matrix needs
        # to accommodate both the input and the hidden state:

        self.Wh = np.random.normal(size=(i+h, h))

        # For Wy (h, o) set the shape to 15, 5
        # Wy: This weight matrix is used to transform the hidden state
        # into the output. In an RNN, after computing the new hidden state,
        # this hidden state is often transformed into an output vector.
        # The transformation is done using the weight matrix Wy.
        self.Wy = np.random.normal(size=(h, o))
        # Biases
        # for bh initialize to zeros with shape 1, 15
        self.bh = np.zeros((1, h))
        # for by initialize to zeros with shape 1, 5
        self.by = np.zeros((1, o))

        # Print the shapes of the weights and biases
        # print("-"*50)
        # print("Wh shape: ", self.Wh.shape)
        # print("Wy shape: ", self.Wy.shape)
        # print("bh shape: ", self.bh.shape)
        # print("by shape: ", self.by.shape)
        # print("-"*50)

    def forward(self, h_prev, x_t):
        """This methond preforms forward pass for one time step
        Args:
            h_prev: is a numpy.ndarray of shape (m, h) containing the
                    previous hidden state
            x_t: is a numpy.ndarray of shape (m, i) that contains the
                 data input for the cell
        Returns: h_next, y
            h_next: is the next hidden state
            y: is the output of the cell
        """
        # Concatenate the previous hidden state and the input data
        input_data = np.concatenate((h_prev, x_t), axis=1)
        # print("-"*50)
        # print("Imput data to forward pass")
        # print("h_prev: ", h_prev.shape)

        # Calculate the next hidden state
        h_next = np.tanh(np.matmul(input_data, self.Wh) + self.bh)
        # print("h_next: ", h_next.shape)

        # Apply softmax activation function to the output
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        # print("y: ", y.shape)
        # print("-"*50)

        return h_next, y

    def softmax(self, y):
        """This method calculates the softmax activation function
        Args:
            y: is a numpy.ndarray of shape (m, o) containing the
               input data
        Returns: softmax
            softmax: is a numpy.ndarray of shape (m, o) containing the
                     softmax activation function
        """
        return np.exp(y) / (np.sum(np.exp(y), axis=1, keepdims=True))
