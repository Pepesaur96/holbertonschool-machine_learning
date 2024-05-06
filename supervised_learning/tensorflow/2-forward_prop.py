#!/usr/bin/env python3
""" Module that contains a function of forward propagation """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Assuming the 'create_layer' function is defined in the '1-create_layer.py' file
create_layer = __import__('1-create_layer').create_layer


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
    # Initialize the input layer
    input_data = x

    # Sequentially add each layer
    for i, size in enumerate(layer_sizes):
        activation_func = activations[i] if i < len(activations) else None
        input_data = create_layer(
            input_data, size, activation_func, name=f"layer_{i}")

    # The final output tensor
    return input_data
