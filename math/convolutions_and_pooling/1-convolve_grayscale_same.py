#!/usr/bin/env python3
""" Convolution with Grayscale Same Convolution """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images
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
    kh, kw = kernel.shape
    m, hm, wm = images.shape
    # padding for same convolution
    ph = int(kh / 2)
    pw = int(kw / 2)
    # creating the output matrix with shape (m, hm, wm)
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convoluted = np.zeros((m, hm, wm))
    # iterating over the images and applying the kernel
    for h in range(hm):
        for w in range(wm):
            square = padded[:, h: h + kh, w: w + kw]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
