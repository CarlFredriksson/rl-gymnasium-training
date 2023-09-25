import numpy as np
import matplotlib.pyplot as plt

def plot_returns(returns):
    print("Average return per episode:", np.sum(returns) / len(returns))
    plt.plot(np.arange(len(returns)), returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
