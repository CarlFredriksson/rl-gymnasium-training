from collections import deque
import random
import numpy as np

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
