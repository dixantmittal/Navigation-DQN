import random

import numpy as np
from constants import *


def to_one_hot(index, n_classes):
    one_hot = np.zeros(n_classes)
    one_hot[index] = 1
    return one_hot


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self.memory = []

    def push(self, state, action, next_state, reward, terminate):
        # Keep on appending new space until capacity is reached
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # Combine the experience into 1 big array and store it on next position. Ordering is important
        self.memory[self.position] = np.hstack((state.reshape(-1), to_one_hot(action, nACTIONS), next_state.reshape(-1), reward,
                                                terminate))

        # Increment the position pointer. We reuse the space by moving to front after limit is reached.
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Change batch size if memory is not big enough
        batch_size = min(batch_size, len(self.memory))

        # Return a sampled batch
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
