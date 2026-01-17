import numpy as np
import random
from collections import defaultdict
from .base_agent import BaseAgent

class ZeroInitializer:
    def __init__(self, size):
        self.size = size
    def __call__(self):
        return np.zeros(self.size)

class QLearningAgent(BaseAgent):
    def __init__(self, action_space_size, alpha=0.1, gamma=0.99):
        super().__init__(action_space_size)
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(ZeroInitializer(action_space_size))

    def get_state_key(self, state):
        # state: (px, py, p_dir, p_hp, ex, ey, e_dir, e_hp, ps1, ps2, ps3, es1, es2, es3)
        if not isinstance(state, tuple):
            state = tuple(state)
            
        px, py, p_dir, p_hp, ex, ey, e_dir, e_hp, ps1, ps2, ps3, es1, es2, es3 = state
        
        # Discretize Position (Sector 4x4)
        sector_x = int(px // 4)
        sector_y = int(py // 4)
        
        # Relative Position
        dx = ex - px
        dy = ey - py
        
        # Sign of relative position
        sign_x = 0
        if dx > 0.5: sign_x = 1
        elif dx < -0.5: sign_x = -1
        
        sign_y = 0
        if dy > 0.5: sign_y = 1
        elif dy < -0.5: sign_y = -1
        
        # Is Aligned? (Bonus feature for shooting)
        aligned_x = abs(dx) < 0.6
        aligned_y = abs(dy) < 0.6
        
        # Construct Key
        # (SectorX, SectorY, SignX, SignY, Aligned?, Dir, WallF, WallL, WallR)
        return (sector_x, sector_y, sign_x, sign_y, aligned_x or aligned_y, p_dir, ps1, ps2, ps3)

    def get_action(self, state, epsilon=0.0):
        state_key = self.get_state_key(state)
        if random.random() < epsilon:
            return random.randint(0, self.action_space_size - 1)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.gamma * self.q_table[next_state_key][best_next_action] * (not done)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error
