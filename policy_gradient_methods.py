import torch
import utils

def select_action_softmax(model, state):
    with torch.no_grad():
        action_probabilities = model(state.unsqueeze(dim=0))
    return torch.distributions.categorical.Categorical(action_probabilities).sample().item()

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

def train_episodic_reinforce(env, policy_model, optimizer, device, rng_seed, num_episodes, gamma, grad_clip_value=None):
    returns = []
    for episode in range(num_episodes):
        # Initiate episode
        # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
        seed = rng_seed if episode == 0 else None
        observation, info = env.reset(seed=seed)
        state = torch.tensor(observation, device=device)
        truncated = False
        terminated = False
        G = 0
        states = [state]
        actions = []
        rewards = []

        # Generate episode
        while not (terminated or truncated):
            action = select_action_softmax(policy_model, state)
            observation, reward, terminated, truncated, info = env.step(action)
            state = torch.tensor(observation, device=device)
            states.append(state)
            actions.append(torch.tensor([action], device=device))
            rewards.append(reward)
            G += reward
        returns.append(G)

        # Learn from episode
        discounted_returns = []
        G = 0
        for t in reversed(range(len(rewards))):
            G = gamma * G + rewards[t]
            discounted_returns.append(gamma**t * G)
        discounted_returns.reverse()
        discounted_returns = torch.tensor(discounted_returns, device=device)
        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        loss = -torch.sum(discounted_returns * torch.log(policy_model(states).gather(1, actions)).squeeze())
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_value != None:
            torch.nn.utils.clip_grad_value_(policy_model.parameters(), grad_clip_value)
        optimizer.step()
        
    return returns, states, actions, rewards

def train_episodic_reinforce_with_baseline(
        env, policy_model, value_model, policy_optimizer, value_optimizer, device, rng_seed, num_episodes, gamma, grad_clip_value=None):
    returns = []
    for episode in range(num_episodes):
        # Initiate episode
        # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
        seed = rng_seed if episode == 0 else None
        observation, info = env.reset(seed=seed)
        state = torch.tensor(observation, device=device)
        truncated = False
        terminated = False
        G = 0
        states = [state]
        actions = []
        rewards = []

        # Generate episode
        while not (terminated or truncated):
            action = select_action_softmax(policy_model, state)
            observation, reward, terminated, truncated, info = env.step(action)
            state = torch.tensor(observation, device=device)
            states.append(state)
            actions.append(torch.tensor([action], device=device))
            rewards.append(reward)
            G += reward
        returns.append(G)

        # Learn from episode
        discounted_returns, discount_factors = compute_discounted_returns(rewards, gamma, device)
        states = torch.stack(states[:-1], dim=0)
        actions = torch.stack(actions, dim=0)
        with torch.no_grad():
            state_values = value_model(states).squeeze()
        delta = discounted_returns - state_values
        policy_loss = -torch.sum(discount_factors * delta * torch.log(policy_model(states).gather(1, actions)).squeeze())
        policy_optimizer.zero_grad()
        policy_loss.backward()
        value_loss = -torch.sum(delta * value_model(states).squeeze())
        value_optimizer.zero_grad()
        value_loss.backward()
        if grad_clip_value != None:
            torch.nn.utils.clip_grad_value_(policy_model.parameters(), grad_clip_value)
            torch.nn.utils.clip_grad_value_(value_model.parameters(), grad_clip_value)
        policy_optimizer.step()
        value_optimizer.step()
        
    return returns, states, actions, rewards
