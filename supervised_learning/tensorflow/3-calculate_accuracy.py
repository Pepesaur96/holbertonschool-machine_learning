#!/usr/bin/env python3
""" Module that contains a function of accuracy """
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Parameters:
    y -- TensorFlow placeholder for the labels of the input data.
    y_pred -- TensorFlow tensor containing the network's predictions.

    Returns:
    A tensor containing the decimal accuracy of the prediction.
    """
    # Find the predicted class from y_pred (assumed to be logits or probabilities)
    prediction = tf.argmax(y_pred, 1)
    # Find the actual class from y
    correct = tf.argmax(y, 1)
    # Compare predictions to the actual labels
    correct_prediction = tf.equal(prediction, correct)
    # Cast booleans to floats and calculate the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
