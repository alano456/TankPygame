from abc import ABC, abstractmethod
import pickle
import numpy as np

class BaseAgent(ABC):
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size

    @abstractmethod
    def get_action(self, state, epsilon=0.0):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
