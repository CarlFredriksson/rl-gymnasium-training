import random
import numpy as np
import utils
import gymnasium as gym

def select_action_eps_greedy(env, Q, state, eps):
    if random.random() < eps:
        return env.action_space.sample().item()
    return np.argmax(Q[state])

def train_episodic_sarsa(env, init_action_values, rng_seed, num_episodes, gamma, eps_start, eps_end, learning_rate):
    returns = []
    Q = init_action_values.copy()
    for episode in range(num_episodes):
        # Initialize episode
        # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
        seed = rng_seed if episode == 0 else None
        state, info = env.reset(seed=seed)
        eps = utils.compute_eps_linear_decay(eps_start, eps_end, num_episodes, episode)
        action = select_action_eps_greedy(env, Q, state, eps)
        truncated = False
        terminated = False
        undiscounted_return = 0

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, info = env.step(action)
            undiscounted_return += reward
            next_action = select_action_eps_greedy(env, Q, next_state, eps)
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            Q[state, action] += learning_rate * delta
            state = next_state
            action = next_action
        
        returns.append(undiscounted_return)
    return returns, Q

def train_episodic_sarsa_lambda(env, init_action_values, rng_seed, num_episodes, gamma, trace_decay_rate, eps_start, eps_end, learning_rate):
    returns = []
    Q = init_action_values.copy()
    for episode in range(num_episodes):
        # Initialize episode
        # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
        seed = rng_seed if episode == 0 else None
        state, info = env.reset(seed=seed)
        eps = utils.compute_eps_linear_decay(eps_start, eps_end, num_episodes, episode)
        action = select_action_eps_greedy(env, Q, state, eps)
        truncated = False
        terminated = False
        undiscounted_return = 0
        z = np.zeros_like(Q) # Eligibility trace

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, info = env.step(action)
            undiscounted_return += reward
            next_action = select_action_eps_greedy(env, Q, next_state, eps)
            z *= gamma * trace_decay_rate
            z[state, action] += 1
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            Q += learning_rate * delta * z
            state = next_state
            action = next_action
        
        returns.append(undiscounted_return)
    return returns, Q

def generate_frames(environment_id, num_episodes, rng_seed, Q):
    env = gym.make(environment_id, render_mode="rgb_array")
    frames = []
    for episode in range(num_episodes):
        state, info = env.reset(seed=rng_seed if episode == 0 else None)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            frames.append(env.render())
            action = select_action_eps_greedy(env, Q, state, 0)
            state, reward, terminated, truncated, info = env.step(action)
    env.close()
    return frames
