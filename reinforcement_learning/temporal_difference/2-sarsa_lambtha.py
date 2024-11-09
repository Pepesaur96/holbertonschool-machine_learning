#!/usr/bin/env python3
"""This module contains the function for the SARSA(λ) algorithm."""
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) algorithm to update the Q table.

    Args:
        env: The OpenAI Gym environment instance.
        Q: A numpy.ndarray of shape (s,a) containing the Q table.
        lambtha: The eligibility trace factor.
        episodes: Total number of episodes to train over (default is 5000).
        max_steps: Maximum number of steps per episode (default is 100).
        alpha: The learning rate (default is 0.1).
        gamma: The discount rate (default is 0.99).
        epsilon: The initial threshold for epsilon-greedy strategy (default is 1).
        min_epsilon: The minimum value that epsilon should decay to (default is 0.1).
        epsilon_decay: The decay rate for updating epsilon between episodes (default is 0.05).

    Returns:
        Q: The updated Q table.
    """

    def epsilon_greedy(state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(env.action_space.n)  # random action
        return np.argmax(Q[state])  # best action according to Q

    for episode in range(episodes):
        state = env.reset()[0]  # reset environment and get initial state
        action = epsilon_greedy(state, epsilon)
        eligibility_trace = np.zeros_like(Q)  # initialize eligibility trace

        for step in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)  # perform action

            # Select next action using epsilon-greedy policy
            next_action = epsilon_greedy(next_state, epsilon)

            # Compute TD error (delta)
            delta = reward + gamma * Q[next_state, next_action] * (1 - done) - Q[state, action]

            # Update eligibility trace for the current state-action pair
            eligibility_trace[state, action] += 1

            # Update Q-values and eligibility traces for all state-action pairs
            Q += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha  # decay eligibility traces

            # Transition to next state and action
            state = next_state
            action = next_action

            # If done, exit the loop for this episode
            if done:
                break

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))

    return Q
