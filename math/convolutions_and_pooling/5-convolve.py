#!/usr/bin/env python3
""" Convolution with Multiple Kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images using multiple kernels
    Args:
    - images: a numpy.ndarray with shape (m, h, w, c) containing multiple
                images
            - m: the number of images
            - h: the height in pixels of the images
            - w: the width in pixels of the images
            - c: the number of channels in the image
    - kernels: a numpy.ndarray with shape (kh, kw, c, nc) containing the
                kernels for the convolution
            - kh: the height of the kernel
            - kw: the width of the kernel
            - nc: the number of kernels
    - padding: a tuple of (ph, pw)
            - ph: the padding for the height of the image
            - pw: the padding for the width of the image
    - stride: a tuple of (sh, sw)
            - sh: the stride for the height of the image
            - sw: the stride for the width of the image
    Returns:
    A numpy.ndarray containing the convolved images
    """
    kh, kw, kc, nc = kernels.shape
    m, hm, wm, cm = images.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((hm - 1) * sh + kh - hm) / 2) + 1
        pw = int(((wm - 1) * sw + kw - wm) / 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    ch = int((hm + 2 * ph - kh) / sh) + 1
    cw = int((wm + 2 * pw - kw) / sw) + 1
    convoluted = np.zeros((m, ch, cw, nc))
    for c in range(nc):
        for h in range(ch):
            for w in range(cw):
                square = padded[:, h * sh: h * sh + kh, w * sw: w * sw + kw, :]
                insert = np.sum(square * kernels[..., c], axis=(1, 2, 3))
                convoluted[:, h, w, c] = insert
    return convoluted
