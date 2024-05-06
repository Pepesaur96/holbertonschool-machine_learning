#!/usr/bin/env python3
""" Module that contains a function of forward propagation """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the neural network
    Args:
        x: the placeholder for the input data
        layer_sizes: a list containing the number of nodes in each layer of the network
        activations: a list containing the activation functions for each layer of the network
    Returns:
        the prediction of the network in tensor form
    """
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction
