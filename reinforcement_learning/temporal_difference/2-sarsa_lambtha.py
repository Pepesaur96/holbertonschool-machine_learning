#!/usr/bin/env python3
"""
SARSA(λ) algorithm implementation for Q-table learning.

This module provides the function `sarsa_lambtha` for updating the Q-table
using the SARSA(λ) algorithm with eligibility traces.
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm to update the Q-table.

    Args:
        env: The OpenAI Gym environment instance.
        Q (numpy.ndarray): The Q-table, with shape (s, a).
        lambtha (float): The eligibility trace factor (λ).
        episodes (int): Total number of episodes to train (default=5000).
        max_steps (int): Maximum steps per episode (default=100).
        alpha (float): The learning rate (default=0.1).
        gamma (float): The discount rate (default=0.99).
        epsilon (float): Initial value for epsilon-greedy strategy (default=1).
        min_epsilon (float): Minimum value for epsilon (default=0.1).
        epsilon_decay (float): Decay rate for epsilon (default=0.05).

    Returns:
        numpy.ndarray: The updated Q-table.
    """

    def epsilon_greedy(state, epsilon):
        """Chooses an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return np.random.choice(env.action_space.n)
        return np.argmax(Q[state])

    for episode in range(episodes):
        state, _ = env.reset()  # reset environment for a new episode
        action = epsilon_greedy(state, epsilon)
        eligibility_trace = np.zeros_like(Q)

        for _ in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)

            # Compute TD error
            delta = reward + gamma * Q[next_state, next_action] * (1 - done) - Q[state, action]

            # Update eligibility trace for the current state-action pair
            eligibility_trace[state, action] += 1

            # Update Q-values and eligibility traces for all state-action pairs
            Q += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha  # decay eligibility traces

            # Transition to next state and action
            state, action = next_state, next_action

            if done or truncated:
                break

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q
