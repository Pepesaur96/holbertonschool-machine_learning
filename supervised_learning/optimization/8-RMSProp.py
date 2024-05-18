#!/usr/bin/env python3
"""sets up RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """alpha = learning rate
    beta2 = RMSProp weight(Discounting factor)
    epsilon = small number to avoid division by zero
    Returns: optimizer"""
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon)

    return optimizer
