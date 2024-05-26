#!/usr/bin/env python3
"""This module creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """This function creates a confusion matrix

    Args:
        labels (numpy.ndarrya): Contains the correct labels for each data point
        logits (numpy.ndarray): Contains the predicted labels
    Returns:
        numpy.ndarray: The confusion matrix oof shape (classes, classes)
    """
    return np.dot(labels.T, logits)
