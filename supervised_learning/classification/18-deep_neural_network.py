#!/usr/bin/env python3
""" Module that defines a deep neural network"""
import numpy as np


class DeepNeuralNetwork:
    """ Class that defines a deep neural network """

    def __init__(self, nx, layers):
        """
        Initializes the DeepNeuralNetwork instance
        Args:
            nx: is the number of input features
            layers: is a list representing the number of nodes in each
                    layer of the network
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if (not isinstance(layers, list) or
                not all(map(lambda x: isinstance(x, int) and x > 0, layers))):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if i == 0:
                self.weights["W" + str(i+1)] = (np.random.randn(layers[i], nx)
                                                * np.sqrt(2 / nx))
            else:
                self.weights["W" + str(i+1)] = (np.random.randn(layers[i],
                                                                layers[i - 1])
                                                * np.sqrt(2 / layers[i - 1]))
            self.weights["b" + str(i+1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Getter function for L """
        return self.__L

    @property
    def cache(self):
        """ Getter function for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter function for weights """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network
        Args:
            X: is a numpy.ndarray with shape (nx, m) that contains the input
               data
                nx: is the number of input features to the neuron
                m: is the number of examples
        Returns: the output of the neural network and the cache, respectively
        """

        if 'A0' not in self.__cache:
            self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            if i == 1:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                Z = np.matmul(W, X) + b
            else:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                X = self.__cache['A{}'.format(i-1)]
                Z = np.matmul(W, X) + b

            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i)], self.__cache
