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
    m, hm, wm = images.shape
    hk, wk = kernel.shape
    ch = hm - hk + 1
    cw = wm - wk + 1
    # creating the output matrix with shape (m, ch, cw)
    convoluted = np.zeros((m, ch, cw))
    # iterating over the images and applying the kernel
    for h in range(ch):
        for w in range(cw):
            square = images[:, h: h + hk, w: w + wk]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
