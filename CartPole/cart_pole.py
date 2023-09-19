import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

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

def update_model(loss_func, optimizer, target, prediction):
    loss = loss_func(target, prediction)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def plot_returns(returns):
    plt.plot(np.arange(len(returns)), returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")

def train_episodic_semi_gradient_sarsa(env, model, loss_func, optimizer, device, num_episodes):
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
            estimated_value = model(state)[action] # Q(s,a)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(observation, device=device)
            G += reward

            if terminated:
                update_model(loss_func, optimizer, torch.tensor(reward, device=device), estimated_value)
            else:
                next_action = select_action_eps_greedy(env, model, next_state, eps)
                with torch.no_grad():
                    estimated_next_value = model(next_state)[next_action] # Q(s',a')
                update_model(loss_func, optimizer, reward + estimated_next_value, estimated_value)
                state = next_state
                action = next_action

        eps = update_eps(eps)
        returns.append(G)
    return returns

def train_episodic_semi_gradient_qlearning(env, model, loss_func, optimizer, device, num_episodes):
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
            estimated_value = model(state)[action] # Q(s,a)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = torch.tensor(observation, device=device)
            G += reward

            if terminated:
                update_model(loss_func, optimizer, torch.tensor(reward, device=device), estimated_value)
            else:
                with torch.no_grad():
                    max_estimated_next_value = torch.max(model(next_state)) # max_a' Q(s',a')
                update_model(loss_func, optimizer, reward + max_estimated_next_value, estimated_value)
                state = next_state

        eps = update_eps(eps)
        returns.append(G)
    return returns

def run_cart_pole_simulation(model, loss_func, optimizer, num_episodes, agent_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch device:", device)
    env = gym.make("CartPole-v1")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    #returns = 
    returns = train_episodic_semi_gradient_sarsa(env, model, loss_func, optimizer, device, 1000)
    env.close()
    print(np.sum(returns) / len(returns))
    plot_returns(returns)
