from random import sample
from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class MinorStateMemory:
    def __init__(self, length=8):
        self.length = length
        self.memory = None

    def append(self, state):
        """Automatically fill to brim if empty, only relevant for initial state"""
        if self.memory is None:
            self.memory = state
        while self.memory.shape[1] < self.length:
            self.memory = torch.cat((self.memory, state), dim=1)

        self.memory = torch.cat((self.memory, state), dim=1)[:, -self.length:, :, :]

    def get(self):
        if self.memory is None:
            return False
        # Returns a copy of self.memory
        return self.memory.clone().detach()

    def __repr__(self):
        return self.memory.__repr__()

    def empty(self):
        self.memory = None


class DualReplayMemory:
    """Two cyclic buffers of size capacity//2"""

    def __init__(self, capacity, initial_weight=0.8, final_weight=0.5, iterations=1000):
        self.capacity = capacity // 2
        self.memory1 = []
        self.memory2 = []
        # python's only way to store pointers
        self.position1 = [0]
        self.position2 = [0]
        self.weight = initial_weight
        self.weight_reduction = (initial_weight - final_weight) / iterations
        self.iterations = iterations
        self.current_iter = 0

    def push(self, *args, primary_buffer=True):
        """Saves a transition."""
        memory = self.memory1 if primary_buffer else self.memory2
        position = self.position1 if primary_buffer else self.position2
        if len(memory) < self.capacity:
            memory.append(Transition(*args))
        else:
            memory[position[0]] = Transition(*args)
        position[0] = (position[0] + 1) % self.capacity

    def sample(self, batch_size):
        """Retrieves a random batch from memory following the weight distribution, and adjusts the weights"""
        favour = int(batch_size * self.weight)
        if self.current_iter < self.iterations:
            self.weight -= self.weight_reduction
            self.current_iter += 1
        return sample(self.memory1, favour) + sample(self.memory2, batch_size - favour)

    def __len__(self):
        return min(len(self.memory1), len(self.memory2))

    def __repr__(self):
        return "M1 with {} entries and M2 with {} entries".format(len(self.memory1), len(self.memory2))
