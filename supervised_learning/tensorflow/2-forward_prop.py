#!/usr/bin/env python3
""" Module that contains a function of forward propagation """
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Parameters:
    x -- TensorFlow placeholder for the input data.
    layer_sizes -- List containing the number of nodes in each layer of the network.
    activations -- List containing the activation functions for each layer of the network.

    Returns:
    TensorFlow tensor that represents the output of the network.
    """
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i is 0:
            prediction = create_layer(x, layer_sizes[i], activations[i])
        else:
            prediction = create_layer(prediction, layer_sizes[i],
                                      activations[i])
    return prediction
