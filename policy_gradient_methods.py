import torch
import numpy as np

def select_action_softmax(model, state):
    with torch.no_grad():
        action_probabilities = model(state.unsqueeze(dim=0))
    return torch.distributions.categorical.Categorical(action_probabilities).sample().item()

def generate_episode(env, policy_model, device, rng_seed=None):
    # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
    observation, info = env.reset(seed=rng_seed)
    state = torch.tensor(observation, device=device)
    truncated = False
    terminated = False
    states = [state]
    actions = []
    rewards = []
    while not (terminated or truncated):
        action = select_action_softmax(policy_model, state)
        observation, reward, terminated, truncated, info = env.step(action)
        state = torch.tensor(observation, device=device)
        states.append(state)
        actions.append(torch.tensor([action], device=device))
        rewards.append(reward)
    return torch.stack(states[:-1], dim=0), torch.stack(actions, dim=0), rewards

def compute_discounted_returns(rewards, gamma, device):
    discounted_returns = []
    discount_factors = []
    G = 0
    for t in reversed(range(len(rewards))):
        G = gamma * G + rewards[t]
        discounted_returns.append(G)
        discount_factors.append(gamma**t)
    discounted_returns.reverse()
    discount_factors.reverse()
    return torch.tensor(discounted_returns, device=device), torch.tensor(discount_factors, device=device)

def update_model(loss, optimizer, model=None, grad_clip_value=None):
    optimizer.zero_grad()
    loss.backward()
    if model != None and grad_clip_value != None:
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
    optimizer.step()

def train_episodic_reinforce(env, policy_model, policy_optimizer, device, rng_seed, num_episodes, gamma, grad_clip_value=None):
    returns = []
    for episode in range(num_episodes):
        states, actions, rewards = generate_episode(env, policy_model, device, rng_seed if episode == 0 else None)
        returns.append(np.sum(rewards))
        discounted_returns, discount_factors = compute_discounted_returns(rewards, gamma, device)
        loss = -torch.sum(discount_factors * discounted_returns * torch.log(policy_model(states).gather(1, actions)).squeeze())
        update_model(loss, policy_optimizer, policy_model, grad_clip_value)
    return returns

def train_episodic_reinforce_with_baseline(
        env, policy_model, value_model, policy_optimizer, value_optimizer, device, rng_seed, num_episodes, gamma, grad_clip_value=None):
    returns = []
    for episode in range(num_episodes):
        states, actions, rewards = generate_episode(env, policy_model, device, rng_seed if episode == 0 else None)
        returns.append(np.sum(rewards))
        discounted_returns, discount_factors = compute_discounted_returns(rewards, gamma, device)
        with torch.no_grad():
            state_values = value_model(states).squeeze()
        delta = discounted_returns - state_values
        value_loss = -torch.sum(delta * value_model(states).squeeze())
        update_model(value_loss, value_optimizer, value_model, grad_clip_value)
        policy_loss = -torch.sum(discount_factors * delta * torch.log(policy_model(states).gather(1, actions)).squeeze())
        update_model(policy_loss, policy_optimizer, policy_model, grad_clip_value)
    return returns
