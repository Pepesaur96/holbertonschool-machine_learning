#!/usr/bin/env python3
""" Module that contains a function of loss """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Parameters:
    y -- TensorFlow placeholder for the labels of the input data,
            one-hot encoded.
    y_pred -- TensorFlow tensor containing the network's predictions
            (logits before softmax).

    Returns:
    A tensor containing the loss of the prediction.
    """
    # Apply softmax to logits and compute the cross-entropy loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y))
    return loss
