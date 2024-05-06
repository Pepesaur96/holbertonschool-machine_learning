#!/usr/bin/env python3
""" Module that contains a function of forward propagation """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def forward_prop(x, layer_sizes=[], activations=[]):
    create_layer = __import__('1-create_layer').create_layer
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction
