import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_returns(returns, target=None):
    print("Average return per episode:", np.sum(returns) / len(returns))
    episodes = np.arange(len(returns))
    plt.plot(episodes, returns, label="Return per episode")
    plt.plot(episodes, rolling_average(returns, 100), label="Rolling average return last 100 episodes")
    if target is not None:
        plt.plot(episodes, np.ones(len(returns)) * target, label="Rolling average return \"solving\" target")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()

def rolling_average(numbers_to_average, window_size):
    rolling_averages = []
    for i in range(len(numbers_to_average)):
        window_start = max(i - window_size + 1, 0)
        window_end = i + 1
        window = numbers_to_average[window_start:window_end]
        rolling_averages.append(sum(window) / window_size)
    return rolling_averages
