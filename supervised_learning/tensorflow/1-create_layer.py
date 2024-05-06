#!/usr/bin/env python3
""" Module that contains a function of layers"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """
    Function that creates a layer for a neural network
    Args:
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function that the layer should use
    Returns:
        tensor output of the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name="layer")

    return layer(prev)
