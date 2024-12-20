#!/usr/bin/env python3
"""
This module contains the function rotate_image
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.
    Args:
        image: A 3D tf.Tensor containing the image to rotate.
    Returns:
        A 3D tf.Tensor of the rotated image.
    """
    return tf.image.rot90(image)
