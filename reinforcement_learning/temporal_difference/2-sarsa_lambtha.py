#!/usr/bin/env python3
"""This module contains the function for the SARSA(λ) algorithm."""
import numpy as np

def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm to estimate the Q-value function.
    
    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes
    
    Returns:
        Q: the updated Q table
    """
    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy_policy(Q, state, epsilon)
        eligibility_traces = np.zeros_like(Q)
        
        for step in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)
            
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            eligibility_traces[state, action] += 1
            
            Q += alpha * delta * eligibility_traces
            eligibility_traces *= gamma * lambtha
            
            if done:
                break
            
            state = next_state
            action = next_action
        
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))
    
    return Q