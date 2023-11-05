import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import gymnasium as gym

def compute_rolling_average(numbers_to_average, window_size):
    rolling_averages = []
    for i in range(len(numbers_to_average)):
        window_start = max(i - window_size + 1, 0)
        window_end = i + 1
        window = numbers_to_average[window_start:window_end]
        rolling_averages.append(sum(window) / window_size)
    return rolling_averages

def plot_returns(returns, target=None):
    print("Average return per episode:", np.sum(returns) / len(returns))
    episodes = np.arange(len(returns))
    plt.plot(episodes, returns, label="Return per episode")
    plt.plot(episodes, compute_rolling_average(returns, 100), label="Rolling average return last 100 episodes")
    if target is not None:
        plt.plot(episodes, np.ones(len(returns)) * target, label="Rolling average return \"solving\" target")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()

def plot_returns_multiple_runs(returns_per_run, target=None):
    if len(returns_per_run) == 1:
        plot_returns(returns_per_run[0], target)
        return
    num_rows = math.ceil(math.sqrt(len(returns_per_run)))
    num_cols = num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 5))
    i = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if i < len(returns_per_run):
                returns = returns_per_run[i]
                episodes = np.arange(len(returns))
                axs[row, col].plot(episodes, returns, label="Return per episode")
                axs[row, col].plot(episodes, compute_rolling_average(returns, 100), label="Rolling average return last 100 episodes")
                if target is not None:
                    axs[row, col].plot(episodes, np.ones(len(returns)) * target, label="Rolling average return \"solving\" target")
            else:
                fig.delaxes(axs[row, col])
            i += 1
    fig.supxlabel("Episode")
    fig.supylabel("Return")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.04))
    plt.tight_layout()

def save_frames_as_gif(frames, path, fig_size_ratio=1/128):
    fig = plt.figure(figsize=(frames[0].shape[1] * fig_size_ratio, frames[0].shape[0] * fig_size_ratio))
    plt.axis("off")
    patch = plt.imshow(frames[0])
    anim = animation.FuncAnimation(fig, lambda i : patch.set_data(frames[i]), frames=range(len(frames)), interval=50)
    anim.save(path, writer="pillow", fps=60)

def generate_random_agent_frames(environment_id, num_episodes, rng_seed):
    env = gym.make(environment_id, render_mode="rgb_array")
    env.action_space.seed(rng_seed)
    frames = []
    for episode in range(num_episodes):
        observation, info = env.reset(seed=rng_seed if episode == 0 else None)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            frames.append(env.render())
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
    env.close()
    return frames
