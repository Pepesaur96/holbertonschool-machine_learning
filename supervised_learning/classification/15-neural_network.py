#!/usr/bin/env python3
"""defines a neural network with one hidden
layer performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """class NeuralNetwork"""

    def __init__(self, nx, nodes):
        """nx = the number of input features to the neuron
        nodes = the number of nodes found in the hidden layer
        W1: The weights vector for the hidden layer. Upon instantiation,
        it should be initialized using a random normal distribution.
        b1: The bias for the hidden layer. Upon instantiation,
        it should be initialized with 0’s.
        A1: The activated output for the hidden layer. Upon instantiation,
        it should be initialized to 0.
        W2: The weights vector for the output neuron. Upon instantiation,
        it should be initialized using a random normal distribution.
        b2: The bias for the output neuron. Upon instantiation,
        it should be initialized to 0.
        A2: The activated output for the output neuron (prediction).
        Upon instantiation, it should be initialized to 0."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """returns private instance weight"""
        return self.__W1

    @property
    def b1(self):
        """returns private instance bias"""
        return self.__b1

    @property
    def A1(self):
        """returns private instance output"""
        return self.__A1

    @property
    def W2(self):
        """returns private instance weight"""
        return self.__W2

    @property
    def b2(self):
        """returns private instance bias"""
        return self.__b2

    @property
    def A2(self):
        """returns private instance output"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        X = a numpy.ndarray with shape (nx, m) that contains the input data
        nx = the number of input features to the neuron
        m = the number of examples
        It updates the private attributes __A1 and __A2
        neurons should use a sigmoid activation function"""
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        Y = a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A = a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        To avoid division by zero errors, it will be used 1.0000001 - A
        instead of 1 - A"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions
        X = numpy.ndarray with shape (nx, m) that contains the input data
        Y = numpy.ndarray with shape (1, m) that contains the correct labels
        nx = the number of input features to the neuron
        m = the number of examples
        It returns the neuron’s prediction and the cost of the network"""
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates one pass of gradient descent on the neural network
        X = numpy.ndarray with shape (nx, m) that contains the input data
        Y = numpy.ndarray with shape (1, m) that contains the correct labels
        A1 = the output of the hidden layer
        A2 = the predicted output
        alpha = the learning rate
        Updates the private attributes __W1, __b1, __W2, and __b2"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dW2 = 1 / m * np.dot(dz2, A1.T)
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = 1 / m * np.dot(dz1, X.T)
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neural network
        X = numpy.ndarray with shape (nx, m) that contains the input data
        Y = numpy.ndarray with shape (1, m) that contains the correct labels
        iterations = positive integer containing the number of iterations
        alpha = positive integer containing the learning rate
        It also updates the private attributes __W, __b, and __A
        verbose = boolean that defines whether or not to print information
        graph = boolean that defines whether or not to graph information
        step = the number of iterations between printing information"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        x_points = []
        y_points = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if i % step == 0:
                cost = self.cost(Y, self.__A2)
                x_points.append(i)
                y_points.append(cost)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
