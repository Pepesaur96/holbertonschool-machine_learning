#!/usr/bin/env python3
"""
This module contains the function flip_image
"""
import tensorflow as tf

def flip_image(image):
    """
    Flips an image horizontally.
    
    Args:
        image: A 3D tf.Tensor containing the image to flip.
    
    Returns:
        A 3D tf.Tensor of the flipped image.
    """
    return tf.image.flip_left_right(image)
