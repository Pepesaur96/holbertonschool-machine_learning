#!/usr/bin/env python3
"""Module that creates a plot for two radioactive elements"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Function that creates a plot for two radioactive elements
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y1, 'r--')

    plt.plot(x, y2, 'g-')

    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')

    plt.title('Exponential Decay of Radioactive Elements')

    plt.legend(loc='upper right')

    plt.show()
