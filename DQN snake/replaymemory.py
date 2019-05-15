from random import sample
from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', "terminal", 'reward'))


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

    def __init__(self, capacity, initial_weight=0.8, final_weight=0.5, iterations=1000, history_size=100):
        self.capacity = capacity // 2
        self.memory = [[], []]
        self.position = [0, 0]
        self.weight = initial_weight
        self.weight_reduction = (initial_weight - final_weight) / iterations
        self.iterations = iterations
        self.current_iter = 0

        self.history = []
        self.history_pointer = 0
        self.history_size = history_size

    def push(self, *args, primary_buffer=True):
        """Saves a transition."""
        if len(self.memory[primary_buffer]) < self.capacity:
            self.memory[primary_buffer].append(Transition(*args))
        else:
            self.memory[primary_buffer][self.position[primary_buffer]] = Transition(*args)
        self.position[primary_buffer] = (self.position[primary_buffer] + 1) % self.capacity

        if len(self.history) < self.history_size:
            self.history.append(primary_buffer)
        else:
            self.history[self.history_pointer] = primary_buffer
        self.history_pointer = (self.history_pointer + 1) % self.history_size

    def reduce_score_of_previous_n_by_p(self, n, p):
        backtrack_position = self.position.copy()
        for i in range(self.history_pointer - 1, self.history_pointer - n - 1, -1):
            try:
                memory = self.history[i]
                backtrack_position[memory] -= 1
                *others, terminal, reward = self.memory[memory][backtrack_position[memory]]
                if terminal.item():  # Don't reduce values of earlier runs
                    break
                reward[0] = reward[0] - p
                # Put all reduced into primary memory
                if memory:  # Already in primary
                    self.memory[memory][backtrack_position[memory]] = Transition(*others, terminal, reward)
                else:
                    self.memory[False].pop()
                    if len(self.memory[True]) < self.capacity:
                        self.memory[True].append(Transition(*others, terminal, reward))
                    else:
                        self.memory[True][self.position[True]] = Transition(*others, terminal, reward)
                    self.position[False] = (self.position[False] - 1) % self.capacity
                    self.position[True] = (self.position[True] + 1) % self.capacity
            except IndexError:
                # p > len(history), may be safely ignored, insignificant
                break

    def iterate_ratio(self):
        if self.current_iter < self.iterations:
            self.weight -= self.weight_reduction
            self.current_iter += 1

    def sample(self, batch_size):
        """Retrieves a random batch from memory following the weight distribution, and adjusts the weights"""
        favour = int(batch_size * self.weight)
        return sample(self.memory[True], favour) + sample(self.memory[False], batch_size - favour)

    def __len__(self):
        return min(len(self.memory[False]), len(self.memory[True]))

    def __repr__(self):
        return "M1 with {} entries and M2 with {} entries".format(len(self.memory[True]), len(self.memory[False]))
