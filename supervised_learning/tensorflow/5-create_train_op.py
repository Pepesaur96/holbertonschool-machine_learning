#!/usr/bin/env python3
""" Module that contains a function to train a model """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Parameters:
    loss -- the loss of the network’s prediction, a TensorFlow tensor.
    alpha -- the learning rate, a float.

    Returns:
    An operation that trains the network using gradient descent.
    """
    # Define the optimizer with the given learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    # Create the training operation to minimize the loss
    train_op = optimizer.minimize(loss)

    return train_op
