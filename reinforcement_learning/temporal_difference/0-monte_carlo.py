#!/usr/bin/env python3
"""
Performs the Monte Carlo algorithm to estimate the value function
"""
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
    
    Returns:
        V: the updated value estimate
    """
    # Loop over episodes
    for episode in range(episodes):
        state = env.reset()[0]
        episode_data = []
        # Loop over steps
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, reward))
            if done:
                break
            state = next_state
        
        # Update the value estimate
        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])
    
    return V
