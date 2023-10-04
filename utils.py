import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_returns(returns):
    print("Average return per episode:", np.sum(returns) / len(returns))
    plt.plot(np.arange(len(returns)), returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")

def update_model(loss_func, optimizer, predictions, targets, model=None, grad_clip_value=None):
    loss = loss_func(predictions, targets)
    optimizer.zero_grad()
    if model != None and grad_clip_value != None:
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
    loss.backward()
    optimizer.step()
