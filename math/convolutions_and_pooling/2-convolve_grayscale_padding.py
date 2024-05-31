#!/usr/bin/env python3
""" Convolution with Grayscale Padding Convolution """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a convolution on grayscale images with
    custom padding
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
    - padding: a tuple of (ph, pw)
            - ph: the padding for the height of the image
            - pw: the padding for the width of the image
    Returns:
    A numpy.ndarray containing the convolved images
    """
    kh, kw = kernel.shape
    m, hm, wm = images.shape
    ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    ch = hm + (2 * ph) - kh + 1
    cw = wm + (2 * pw) - kw + 1
    # creating the output matrix with shape (m, ch, cw)
    convoluted = np.zeros((m, ch, cw))
    # iterating over the images and applying the kernel
    for h in range(ch):
        for w in range(cw):
            square = padded[:, h: h + kh, w: w + kw]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
