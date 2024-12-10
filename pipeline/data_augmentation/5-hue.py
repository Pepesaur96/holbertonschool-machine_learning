#!/usr/bin/env python3
"""
This module contains the function change_hue
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.
    Args:
        image: A 3D tf.Tensor containing the image to
        change.
        delta: A float representing the amount the hue
        should change.
    Returns:
        A 3D tf.Tensor of the hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)
