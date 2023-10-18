import numpy as np
import matplotlib.pyplot as plt
import math

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
