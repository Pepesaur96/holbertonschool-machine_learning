#!/usr/bin/env python3
""" Convolution with Grayscale Valid Convolution """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images
    Args:
    - images: a numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            - m: the number of images
            - h: the height in pixels of the images
            - w: the width in pixels of the images
    - kernel: a numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
            - kh: the height of the kernel
            - kw: the width of the kernel
    Returns:
    A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved_images = np.zeros((m, output_h, output_w))

    for i in range(m):
        for x in range(output_h):
            for y in range(output_w):
                convolved_images[i, x, y] = np.sum(
                    images[i, x:x+kh, y:y+kw] * kernel
                )

    return convolved_images
