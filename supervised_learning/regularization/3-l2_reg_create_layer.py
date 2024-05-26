#!/usr/bin/env python3
"""This module contains a function that creates a neural network layer
in TensorFlow that includes L2 regularization.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow that
    includes L2 regularization.

    Parameters:
    prev: tensor - a tensor containing the output of the
    previous layer.
    n: int - the number of nodes the new layer should contain.
    activation: function - the activation function that
    should be used on the layer.
    lambtha: float - the L2 regularization parameter.

    Returns:
    The output of the new layer.
    """
    # Initialize the weights using a kernel regularizer
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    regularizer = tf.keras.regularizers.l2(lambtha)

    # Create the layer
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)

    # Connect the new layer to the previous layer
    output = layer(prev)

    return output
