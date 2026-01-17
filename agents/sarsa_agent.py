import numpy as np
import random
from collections import defaultdict
from .base_agent import BaseAgent

class ZeroInitializer:
    def __init__(self, size):
        self.size = size
    def __call__(self):
        return np.zeros(self.size)

class SarsaAgent(BaseAgent):
    def __init__(self, action_space_size, alpha=0.1, gamma=0.99):
        super().__init__(action_space_size)
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(ZeroInitializer(action_space_size))

    def get_relative_angle(self, p_dir, px, py, ex, ey):
        import math
        dx = ex - px
        dy = ey - py
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Normalize player dir (0-360)
        p_dir = p_dir % 360
        if target_angle < 0:
            target_angle += 360
            
        diff = target_angle - p_dir
        # Normalize diff to (-180, 180)
        if diff > 180: diff -= 360
        if diff < -180: diff += 360
        
        return diff

    def get_state_key(self, state_input):
        # We expect state_input to be (dir_bin, dist_bin, obstacle_ahead, has_los, cooldown)
        # OR legacy formats.
        if isinstance(state_input, tuple):
             if len(state_input) == 5:
                 # Already discrete: (dir, dist, obst, los, cool)
                 return state_input
             if len(state_input) == 4:
                 return state_input # Legacy 4-item
             elif len(state_input) == 3:
                 return state_input # Legacy 3-item
             elif len(state_input) < 8:
                  return state_input
        
        
        # Parse Obs
        obs = state_input
        # Obs: [px, py, p_dir, p_hp, ex, ey, e_dir, e_hp, ...]
        if not isinstance(obs, (list, tuple, np.ndarray)) or len(obs) < 8:
             # Ensure we return a hashable tuple even if fallback
             if isinstance(state_input, (list, np.ndarray)):
                 return tuple(state_input)
             return state_input # Fallback

        px, py, p_dir = obs[0], obs[1], obs[2]
        ex, ey, e_dir = obs[4], obs[5], obs[6]

        # Define Cooldown default for raw obs (not present in 14-dim obs)
        cooldown = 0 
        
        # 1. My Relative Angle (Where is enemy?)
        angle_diff = self.get_relative_angle(p_dir, px, py, ex, ey)
        
        # 1. My Relative Angle (Where is enemy?)
        angle_diff = self.get_relative_angle(p_dir, px, py, ex, ey)
        
        # 8-Sector Binning (45 degrees each)
        # Shift so -22.5..22.5 -> 0..45 (Bin 0)
        # 22.5..67.5 -> 45..90 (Bin 1)
        # etc.
        angle_bin = int((angle_diff + 22.5) % 360 / 45)

        # 2. Distance
        dist = abs(ex - px) + abs(ey - py)
        if dist < 4: dist_bin = 0
        elif dist < 10: dist_bin = 1
        elif dist < 20: dist_bin = 2
        else: dist_bin = 3

        # 3. Cooldown
        can_shoot = 0 if cooldown == 0 else 1
        
        # 4. Enemy Threat (Is he aiming at me?)
        enemy_angle_diff = self.get_relative_angle(e_dir, ex, ey, px, py)
        is_threat = 0
        if abs(enemy_angle_diff) < 30: 
            is_threat = 1
            
        # 5. Line of Sight (0 or 1)
        # Passed externally because we need Grid access
        
        # 6. Obstacle Ahead (0 or 1)
        # Passed externally or calculated if bounds known. Best to pass externally.
        obstacle_ahead = 0
        if len(state_input) >= 4:
            obstacle_ahead = state_input[3]

        # 7. Is Trapped (0 or 1) - Corner Detection
        is_trapped = 0
        if len(state_input) >= 7:
             is_trapped = state_input[6]
            
        return (angle_bin, dist_bin, can_shoot, is_threat, has_los, obstacle_ahead, is_trapped)

    def get_action(self, state, epsilon=0.0):
        state_key = self.get_state_key(state)
        if random.random() < epsilon:
            return random.randint(0, self.action_space_size - 1)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done, next_action=None):
        if next_action is None:
            raise ValueError("SARSA update requires next_action")
        
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
            
        td_target = reward + self.gamma * self.q_table[next_state_key][next_action] * (not done)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.alpha * td_error
