#!/usr/bin/env python3
""" Module that contains a function of placeholders """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders, x and y, for a neural network
    Args:
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
    Returns:
        x: the placeholder for the input data to the neural network
        y: the placeholder for the one-hot labels for the input data
    """

    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

    return x, y
