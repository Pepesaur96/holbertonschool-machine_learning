#!/usr/bin/env python3
"""sets up the gradient descent with momentum
optimization algorithm"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """alpha = learning rate
    beta1 = momentum weight
    Returns: the momentum optimization operation"""
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1)

    return optimizer
