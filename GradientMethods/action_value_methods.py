import torch
from collections import deque
import random
import utils

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
    if random.random() < eps:
        return env.action_space.sample().item()
    with torch.no_grad():
        return torch.argmax(model(state)).item()

def update_model(loss_func, optimizer, predictions, targets, model=None, grad_clip_value=None):
    loss = loss_func(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    if model != None and grad_clip_value != None:
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
    optimizer.step()

def experience_replay(policy_model, target_model, replay_memory, batch_size, loss_func, optimizer, device, gamma, grad_clip_value):
    if len(replay_memory) >= batch_size:
        # Prepare batch
        state_batch, action_batch, reward_batch, next_state_batch = replay_memory.sample_batch(batch_size)
        state_batch = torch.stack(state_batch, dim=0)
        action_batch = torch.stack(action_batch, dim=0)
        reward_batch = torch.stack(reward_batch, dim=0)
        non_terminal_next_states = torch.stack([s for s in next_state_batch if s is not None], dim=0)
        non_terminal_next_states_mask = torch.tensor([s != None for s in next_state_batch], device=device)

        # Update policy model using batch
        state_action_values = policy_model(state_batch).gather(1, action_batch).squeeze()
        max_next_state_action_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            max_next_state_action_values[non_terminal_next_states_mask] = torch.max(target_model(non_terminal_next_states), dim=1).values
            targets = reward_batch + gamma * max_next_state_action_values
        update_model(loss_func, optimizer, state_action_values, targets, policy_model, grad_clip_value)

def train_episodic_semi_grad_sarsa(
        env, model, loss_func, optimizer, device, rng_seed, num_episodes, gamma, eps_start, eps_end, grad_clip_value=None):
    returns = []
    for episode in range(num_episodes):
        # Initialize episode
        # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
        seed = rng_seed if episode == 0 else None
        observation, info = env.reset(seed=seed)
        state = torch.tensor(observation, device=device)
        eps = utils.compute_eps_linear_decay(eps_start, eps_end, num_episodes, episode)
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
                    target = reward + gamma * next_state_action_value
                update_model(loss_func, optimizer, state_action_value, target, model, grad_clip_value)
                state = next_state
                action = next_action

        returns.append(G)
    return returns

def train_episodic_semi_grad_qlearning(
        env, model, loss_func, optimizer, device, rng_seed, num_episodes, gamma, eps_start, eps_end, grad_clip_value=None):
    returns = []
    for episode in range(num_episodes):
        # Initialize episode
        # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
        seed = rng_seed if episode == 0 else None
        observation, info = env.reset(seed=seed)
        state = torch.tensor(observation, device=device)
        eps = utils.compute_eps_linear_decay(eps_start, eps_end, num_episodes, episode)
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
                    target = reward + gamma * max_next_state_action_value
                update_model(loss_func, optimizer, state_action_value, target, model, grad_clip_value)
                state = next_state

        returns.append(G)
    return returns

def train_episodic_semi_grad_qlearning_exp_replay(
        env, policy_model, target_model, loss_func, optimizer, device, rng_seed, num_episodes, gamma, eps_start, eps_end,
        memory_size, batch_size, num_steps_between_target_model_updates, grad_clip_value=None):
    returns = []
    replay_memory = ReplayMemory(memory_size)
    for episode in range(num_episodes):
        # Initialize episode
        # Set seed only one time per training run. For more info, see https://gymnasium.farama.org/api/env/.
        seed = rng_seed if episode == 0 else None
        observation, info = env.reset(seed=seed)
        state = torch.tensor(observation, device=device)
        eps = utils.compute_eps_linear_decay(eps_start, eps_end, num_episodes, episode)
        truncated = False
        terminated = False
        G = 0
        t = 0

        while not (terminated or truncated):
            # Update the target model to duplicate the policy model
            if t % num_steps_between_target_model_updates == 0:
                target_model.load_state_dict(policy_model.state_dict())

            # Take action
            action = select_action_eps_greedy(env, policy_model, state, eps)
            observation, reward, terminated, truncated, info = env.step(action)
            G += reward

            # Store transition in replay memory
            next_state = torch.tensor(observation, device=device)
            action = torch.tensor([action], device=device)
            reward = torch.tensor(reward, device=device)
            replay_memory.store(state, action, reward, None if terminated else next_state)

            # Update policy model using experience replay
            experience_replay(policy_model, target_model, replay_memory, batch_size, loss_func, optimizer, device, gamma, grad_clip_value)

            # Prepare next time step t
            state = next_state
            t += 1

        returns.append(G)
    return returns
