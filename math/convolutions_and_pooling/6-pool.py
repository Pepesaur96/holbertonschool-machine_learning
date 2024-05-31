#!/usr/bin/env python3
""" Pooling """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images
    Args:
    - images: a numpy.ndarray with shape (m, h, w, c) containing multiple
                images
            - m: the number of images
            - h: the height in pixels of the images
            - w: the width in pixels of the images
            - c: the number of channels in the image
    - kernel_shape: a tuple of (kh, kw) containing the kernel shape for the
                    pooling
            - kh: the height of the kernel
            - kw: the width of the kernel
    - stride: a tuple of (sh, sw)
            - sh: the stride for the height of the image
            - sw: the stride for the width of the image
    - mode: a string containing either max or avg, indicating whether to
            perform maximum or average pooling, respectively
    Returns:
    A numpy.ndarray containing the pooled images
    """
    m, hm, wm, cm = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    # creating the output matrix with shape (m, hm, wm, cm)
    ch = int((hm - kh) / sh) + 1
    cw = int((wm - kw) / sw) + 1
    convoluted = np.zeros((m, ch, cw, cm))
    # iterating over the images and applying the kernel
    for h in range(ch):
        for w in range(cw):
            square = images[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
            if mode == 'max':
                insert = np.max(square, axis=(1, 2))
            else:
                insert = np.average(square, axis=(1, 2))
            convoluted[:, h, w, :] = insert
    return convoluted
