#!/usr/bin/env python3
"""
Compute the Monte-Carlo policy gradient
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes a stochastic policy by taking a weighted combination of the
    state and weight matrices and applying a softmax function.
    """
    weighted_states = (matrix @ weight)
    e_x = np.exp(weighted_states - np.max(weighted_states))
    return e_x / np.sum(e_x)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a
    weight matrix.
    """
    action_probs = policy(state, weight)
    action = np.random.choice(len(action_probs), p=action_probs)
    d_softmax = action_probs.copy()
    d_softmax[action] -= 1
    grad = -np.outer(state, d_softmax)

    return action, grad
