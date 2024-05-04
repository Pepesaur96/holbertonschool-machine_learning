#!/usr/bin/env python3
"""defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """class Neuron"""

    def __init__(self, nx):
        """nx is the number of input features to the neuron
        Attributes:
        W: The weights vector for the neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        b: The bias for the neuron. Upon instantiation, it should be
        initialized to 0.
        A: The activated output of the neuron (prediction). Upon
        instantiation, it should be initialized to 0."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """returns private instance weight"""
        return self.__W

    @property
    def b(self):
        """returns private instance bias"""
        return self.__b

    @property
    def A(self):
        """returns private instance output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron
        X = a numpy.ndarray with shape (nx, m) that contains
        the input data
        nx = the number of input features to the neuron
        m = the number of examples
        It updates the private attribute __A
        neuron should use the sigmoid function"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
