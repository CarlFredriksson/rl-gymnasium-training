import numpy as np
import torch
from torch import nn
from collections import deque
import random

# For big projects, consider using torchrl.data.ReplayBuffer.
# For more info, see https://pytorch.org/rl/tutorials/rb_tutorial.html.
class ReplayMemory():
    def __init__(self, size):
        self._memory = deque(maxlen=size)

    def __len__(self):
        return len(self._memory)

    def store(self, state, action, reward, next_state):
        self._memory.append((state, action, reward, next_state))

    def sample_batch(self, batch_size):
        batch = random.sample(self._memory, batch_size)
        # Transpose list of (state, action, reward, next_state) to
        # (state_batch, action_batch, reward_batch, next_state_batch). For more info, see
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip/19343#19343.
        return zip(*batch)

def select_action_eps_greedy(env, model, state, eps):
    if np.random.random() < eps:
       return env.action_space.sample().item()
    with torch.no_grad():
        return torch.argmax(model(state)).item()

def update_eps(eps, decay=0.995, min_value=0.01):
    return max(eps * decay, min_value)

def update_model(loss_func, optimizer, predictions, targets, model=None, grad_clip_value=None):
    loss = loss_func(predictions, targets)
    optimizer.zero_grad()
    if model != None and grad_clip_value != None:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
    loss.backward()
    optimizer.step()

def train_episodic_semi_grad_sarsa(
        env, model, loss_func, optimizer, device, num_episodes, eps_start, eps_min, eps_decay, grad_clip_value=None):
    eps = eps_start
    returns = []
    for ep in range(num_episodes):
        observation, info = env.reset()
        state = torch.tensor(observation, device=device)
        action = select_action_eps_greedy(env, model, state, eps)
        truncated = False
        terminated = False
        G = 0

        while not (terminated or truncated):
            state_action_value = model(state)[action] # Q(s,a)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(observation, device=device)
            G += reward

            if terminated:
                target = torch.tensor(reward, device=device)
                update_model(loss_func, optimizer, state_action_value, target, model, grad_clip_value)
            else:
                next_action = select_action_eps_greedy(env, model, next_state, eps)
                with torch.no_grad():
                    next_state_action_value = model(next_state)[next_action] # Q(s',a')
                    target = reward + next_state_action_value
                update_model(loss_func, optimizer, state_action_value, target, model, grad_clip_value)
                state = next_state
                action = next_action

        eps = update_eps(eps, eps_decay, eps_min)
        returns.append(G)
    return returns

def train_episodic_semi_grad_qlearning(
        env, model, loss_func, optimizer, device, num_episodes, eps_start, eps_min, eps_decay, grad_clip_value=None):
    eps = eps_start
    returns = []
    for ep in range(num_episodes):
        observation, info = env.reset()
        state = torch.tensor(observation, device=device)
        truncated = False
        terminated = False
        G = 0

        while not (terminated or truncated):
            action = select_action_eps_greedy(env, model, state, eps)
            state_action_value = model(state)[action] # Q(s,a)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(observation, device=device)
            G += reward

            if terminated:
                target = torch.tensor(reward, device=device)
                update_model(loss_func, optimizer, state_action_value, target, model, grad_clip_value)
            else:
                with torch.no_grad():
                    max_next_state_action_value = torch.max(model(next_state)) # max_a' Q(s',a')
                    target = reward + max_next_state_action_value
                update_model(loss_func, optimizer, state_action_value, target, model, grad_clip_value)
                state = next_state

        eps = update_eps(eps, eps_decay, eps_min)
        returns.append(G)
    return returns

def train_episodic_semi_grad_qlearning_exp_replay(
        env, model, loss_func, optimizer, device, num_episodes, eps_start, eps_min, eps_decay,
        memory_size, batch_size, grad_clip_value=None):
    eps = eps_start
    returns = []
    replay_memory = ReplayMemory(memory_size)
    for ep in range(num_episodes):
        observation, info = env.reset()
        state = torch.tensor(observation, device=device).unsqueeze(0)
        truncated = False
        terminated = False
        G = 0

        while not (terminated or truncated):
            action = select_action_eps_greedy(env, model, torch.tensor(observation, device=device), eps)
            observation, reward, terminated, truncated, info = env.step(action)
            G += reward
            next_state = torch.tensor(observation, device=device).unsqueeze(0)
            action = torch.tensor([[action]], device=device)
            reward = torch.tensor([reward], device=device)
            replay_memory.store(state, action, reward, None if terminated else next_state)

            if len(replay_memory) >= batch_size:
                state_batch, action_batch, reward_batch, next_state_batch = replay_memory.sample_batch(batch_size)
                state_batch = torch.cat(state_batch)
                action_batch = torch.cat(action_batch)
                reward_batch = torch.cat(reward_batch)
                state_action_values = model(state_batch).gather(1, action_batch).squeeze()
                non_terminal_next_states_mask = torch.tensor([s != None for s in next_state_batch], device=device)
                non_terminal_next_states = torch.cat([s for s in next_state_batch if s is not None])
                max_next_state_action_values = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    max_next_state_action_values[non_terminal_next_states_mask] = torch.max(model(non_terminal_next_states), dim=1).values
                    targets = reward_batch + max_next_state_action_values
                update_model(loss_func, optimizer, state_action_values, targets, model, grad_clip_value)

            state = next_state

        eps = update_eps(eps, eps_decay, eps_min)
        returns.append(G)
    return returns
