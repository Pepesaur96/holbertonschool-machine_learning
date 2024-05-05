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
