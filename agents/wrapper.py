import gymnasium as gym
import numpy as np
import math
from game.grid import Grid

class HunterDeepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.env = env
        self.last_angle_err = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_angle_err = self._get_angle_error() 
        return self._get_neural_state(role="player"), info

    def step(self, action):
        # We assume training happens as 'player'. 
        # For 'enemy' training we would need to pass role to step too?
        # Usually we train as player.
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- NORMALIZATION & REWARD Logic (Same as before) ---
        current_angle_err = self._get_angle_error()
        angle_improvement = self.last_angle_err - current_angle_err
        self.last_angle_err = current_angle_err
        
        if angle_improvement > 0: reward = 0.01
        elif angle_improvement < 0: reward = -0.01
        else: reward = 0.0

        if terminated:
            if self.env.enemy.hp <= 0: reward = 1.0 
            elif self.env.player.hp <= 0: reward = -0.5 
        
        if action == 3 and self.env.cooldowns['player'] > 0:
             reward -= 0.05 

        return self._get_neural_state(role="player"), reward, terminated, truncated, info

    def _get_angle_error(self):
        px, py = self.env.player.x, self.env.player.y
        ex, ey = self.env.enemy.x, self.env.enemy.y
        target_rad = math.atan2(ey - py, ex - px)
        target_deg = math.degrees(target_rad)
        p_dir = self.env.player.direction
        dir_map = {"N": -90, "S": 90, "E": 0, "W": 180}
        current_deg = dir_map.get(p_dir, 0)
        return abs((target_deg - current_deg + 180) % 360 - 180)

    def _get_neural_state(self, role="player"):
        # Select actor based on role
        if role == "player":
            actor = self.env.player
            opponent = self.env.enemy
            cooldown = self.env.cooldowns['player']
        else:
            actor = self.env.enemy
            opponent = self.env.player
            cooldown = self.env.cooldowns['enemy']

        px, py = actor.x, actor.y
        ex, ey = opponent.x, opponent.y
        
        target_rad = math.atan2(ey - py, ex - px)
        p_dir = actor.direction
        dir_rad_map = {"N": -1.57, "S": 1.57, "E": 0, "W": 3.14}
        curr_rad = dir_rad_map.get(p_dir, 0)
        
        rel_rad = target_rad - curr_rad
        
        sin_a = math.sin(rel_rad)
        cos_a = math.cos(rel_rad)
        
        # Dynamic map size normalization
        diag = math.hypot(self.env.width, self.env.height)
        dist = math.hypot(ex - px, ey - py) / 17.0 
        # Wait, if we use dynamic diag in main.py, we MUST use it here too for consistency!
        # In train_dqn.py we hardcoded / 17.0.
        # User said "12x12" and I changed train_dqn.py to 17.0 manually.
        # To be robust, let's stick to 17.0 IF we are training on 12x12.
        # OR use generic math.hypot(width, height) but we must know width/height are grid units (12).
        # TankEnv stores width/height as Grid Size (12). Tank x,y are Pixels (if not changed?)
        # Let's check logic:
        # In wrapper below: px, py are actor.x (Pixels).
        # But we divide by 40 in Raycast logic.
        # So math.hypot is PIXELS.
        # 17.0 is TINY for pixels.
        # If I trained with 17.0 on pixels (observed val ~300), inputs were huge (average 17).
        # Neural nets typically clip > 1 or saturate.
        # If I want correct normalization, I should normalize pixels to 0-1 range.
        # dist_norm = dist_pixels / (diag_blocks * 40).
        # BUT I must respect what was *actually trained*.
        # In train_dqn.py I wrote: `dist = math.hypot(...) / 17.0`.
        # And `px/40` for grid checks.
        # This confirms px IS pixels.
        # So the agent was trained with `dist` values around 10-20.
        # This is surprisingly high for NN (usually desire 0-1), but if it works (100% winrate), I MUST replicate it.
        # So I will keep `/ 17.0`.
        
        cd = 1.0 if cooldown > 0 else 0.0
        
        # Raycast
        dx, dy = 0, 0
        if p_dir == "N": dy = -1
        elif p_dir == "S": dy = 1
        elif p_dir == "E": dx = 1
        elif p_dir == "W": dx = -1
        
        # Grid logic
        # Convert pixels to grid
        gpx, gpy = int(px/40), int(py/40)
        
        if p_dir == "N": gpy -= 1
        elif p_dir == "S": gpy += 1
        elif p_dir == "E": gpx += 1
        elif p_dir == "W": gpx -= 1
        
        wall = 0.0
        if not (0 <= gpx < self.env.width and 0 <= gpy < self.env.height):
            wall = 1.0
        elif self.env.grid.cells[gpy][gpx] == Grid.WALL:
            wall = 1.0

        return np.array([sin_a, cos_a, dist, cd, wall], dtype=np.float32)
