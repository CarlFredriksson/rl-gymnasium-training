import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from rl_utils import *

class LinearModel(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.linear = nn.Linear(num_input, num_output)
    
    def forward(self, X):
        return self.linear(X)

def select_action_eps_greedy(env, model, state, eps):
    if np.random.random() < eps:
       return env.action_space.sample().item()
    with torch.no_grad():
        return torch.argmax(model(state)).item()

def update_eps(eps, decay=0.995, min_value=0.1):
    return max(eps * decay, min_value)

def update_model(loss_func, optimizer, predictions, targets, model=None, grad_clip_value=None):
    loss = loss_func(predictions, targets)
    optimizer.zero_grad()
    if model != None and grad_clip_value != None:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
    loss.backward()
    optimizer.step()

def plot_returns(returns):
    print("Average return per episode:", np.sum(returns) / len(returns))
    plt.plot(np.arange(len(returns)), returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")

def train_episodic_semi_grad_sarsa(env, model, loss_func, optimizer, device, num_episodes, grad_clip_value=None):
    eps = 1
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

        eps = update_eps(eps)
        returns.append(G)
    return returns

def train_episodic_semi_grad_qlearning(env, model, loss_func, optimizer, device, num_episodes, grad_clip_value=None):
    eps = 1
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

        eps = update_eps(eps)
        returns.append(G)
    return returns

def train_episodic_semi_grad_qlearning_experience_replay(env, model, loss_func, optimizer, device, num_episodes, memory_size, batch_size, grad_clip_value=None):
    eps = 1
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

        eps = update_eps(eps)
        returns.append(G)
    return returns
