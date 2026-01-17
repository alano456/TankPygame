import gymnasium as gym
from gymnasium import spaces
from game.grid import Grid
from game.tank import Tank
from game.enemy import EnemyBot


class TankEnv(gym.Env):
    def __init__(self, width=20, height=20):
        super().__init__()
        self.width = width
        self.height = height

        self.action_space = spaces.Discrete(4) # Size 4: Left, Right, Fwd, Shoot
        # Obs: [px, py, p_dir, p_hp, ex, ey, e_dir, e_hp, 
        #       p_wall_F, p_wall_L, p_wall_R, e_wall_F, e_wall_L, e_wall_R]
        self.observation_space = spaces.Box(
            low=0, high=max(width, height), shape=(14,), dtype=int
        )

        self.enemy_bot = EnemyBot()
    def reset(self, seed=None, options=None, grid_layout=None):
        import random
        # Create grid
        self.grid = Grid(self.width, self.height)
        
        # If custom layout provided, apply it
        if grid_layout:
            for y in range(self.height):
                for x in range(self.width):
                    if y < len(grid_layout) and x < len(grid_layout[0]):
                         self.grid.cells[y][x] = grid_layout[y][x]
        
        # Valid spawn search
        attempts = 0
        while True:
            attempts += 1
            px, py = random.randint(1, self.width-2), random.randint(1, self.height-2)
            ex, ey = random.randint(1, self.width-2), random.randint(1, self.height-2)
            
            # Ensure safe spawn on empty cells
            if (self.grid.cells[py][px] == Grid.EMPTY and 
                self.grid.cells[ey][ex] == Grid.EMPTY and
                (abs(px - ex) + abs(py - ey) > 5)):
                break
                
            # Fallback if map is too crowded
            if attempts > 100:
                px, py = 1, 1
                ex, ey = self.width-2, self.height-2
                break
        
        self.player = Tank(px, py, random.choice(["N", "E", "S", "W"]))
        self.enemy = Tank(ex, ey, random.choice(["N", "E", "S", "W"]))
        
        # State tracking for penalties
        self.prev_pos = {"player": (px, py), "enemy": (ex, ey)}
        self.last_shot_pos = {"player": None, "enemy": None}
        
        # Reward Shaping Init
        self.prev_dist = self._get_manhattan_dist()

        self.steps = 0
        
        # Cooldown management
        self.cooldowns = {"player": 0, "enemy": 0}
        self.RELOAD_TIME = 10 # steps
        
        self.projectiles = []  # List of {x, y, dx, dy, role, owner}
        self.stats = {
            "player": {"shots": 0, "hits": 0},
            "enemy": {"shots": 0, "hits": 0}
        }
        
        return self._get_obs(), {}

    def step(self, action, enemy_action=None):
        self.steps += 1
        reward = -0.01 # Explore incentive
        enemy_reward = -0.01
        self.step_penalties = {"player": 0.0, "enemy": 0.0}
        terminated = False
        
        # Decrease cooldowns
        for role in self.cooldowns:
            if self.cooldowns[role] > 0:
                self.cooldowns[role] -= 1

        # Track hits before updating projectiles
        prev_hp_p = self.player.hp
        prev_hp_e = self.enemy.hp
        
        self._update_projectiles()
        
        # Hit Rewards/Penalties
        if self.player.hp < prev_hp_p:
            reward -= 50
            enemy_reward += 50
            
        if self.enemy.hp < prev_hp_e:
            enemy_reward -= 50
            reward += 50

        # Collision & Shot Costs Logic
        # Check Collision for Player
        # New Actions: 0:Left, 1:Right, 2:Fwd, 3:Shoot
        if action == 2: # Move Forward
            if not self._can_move(self.player, self.enemy):
                reward -= 0.5 # Collision Penalty
        elif action == 3: # Shoot
             if self.cooldowns["player"] == 0:
                 reward -= 0.05 # Shot Cost

        # Check Collision for Enemy
        act_e = 0
        if self.enemy.alive:
            if enemy_action is not None:
                act_e = enemy_action
            else:
                # RuleBased returns 0-4. Map to 0-3 if needed.
                raw_act = self.enemy_bot.choose_action(self.enemy, self.player, self.grid, self.cooldowns["enemy"])
                if raw_act > 0: act_e = raw_act - 1
                else: act_e = 0 # strict mapping
                
            if act_e == 2: # Move Forward
                if not self._can_move(self.enemy, self.player):
                    enemy_reward -= 0.5
            elif act_e == 3:
                if self.cooldowns["enemy"] == 0:
                     enemy_reward -= 0.05

            self._apply_action(self.enemy, self.player, act_e, "enemy")

        # Player Action Application
        self._apply_action(self.player, self.enemy, action, "player")

        # Penalties: Stagnation (standing still)
        if (self.player.x, self.player.y) == self.prev_pos["player"]:
            reward -= 0.1
        self.prev_pos["player"] = (self.player.x, self.player.y)

        if (self.enemy.x, self.enemy.y) == self.prev_pos["enemy"]:
            enemy_reward -= 0.1
        self.prev_pos["enemy"] = (self.enemy.x, self.enemy.y)

        # REWARD SHAPING (Distance)
        # Calculate new distance
        curr_dist = self._get_manhattan_dist()
        delta = self.prev_dist - curr_dist # Positive if closer
        # Apply shaping
        reward += delta * 0.05
        enemy_reward += delta * 0.05 # Assumes mutual approach is good
        self.prev_dist = curr_dist

        # Terminal States
        if not self.enemy.alive:
            reward += 1000
            enemy_reward -= 1000
            terminated = True
        if not self.player.alive:
            reward -= 1000
            enemy_reward += 1000
            terminated = True
            
        # Apply gathered penalties
        reward += self.step_penalties["player"]
        enemy_reward += self.step_penalties["enemy"]
            
        return self._get_obs(), reward, terminated, False, {"enemy_reward": enemy_reward}

    def _can_move(self, actor, opponent=None):
        """Check if actor can move forward. Checks walls and opponent collision."""
        dx, dy = self._get_direction_vector(actor.direction)
        nx, ny = actor.x + dx, actor.y + dy
        
        # Check Map Bounds and Walls
        if not (self.grid.in_bounds(nx, ny) and self.grid.is_free(nx, ny)):
            return False
            
        # Check Opponent Collision
        if opponent and opponent.alive:
             if nx == opponent.x and ny == opponent.y:
                 return False
                 
        return True

    def _get_manhattan_dist(self):
        return abs(self.player.x - self.enemy.x) + abs(self.player.y - self.enemy.y)

    def _update_projectiles(self):
        # projectile speed increased to 4 (High Speed)
        # We must use sub-stepping to prevent tunneling through walls/tanks
        speed = 4.0
        step_size = 0.5 
        
        for p in self.projectiles[:]:
            dist_remaining = speed
            hit = False
            
            while dist_remaining > 0:
                current_step = min(step_size, dist_remaining)
                dist_remaining -= current_step
                
                # Propose new position
                new_x = p["x"] + p["dx"] * current_step
                new_y = p["y"] + p["dy"] * current_step
                
                # 1. Check Bounds
                if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                    self.projectiles.remove(p)
                    hit = True; break
                
                # 2. Check Wall Collision
                # We check the cell we are currently entering/inside
                if self.grid.cells[int(new_y)][int(new_x)] == Grid.WALL:
                    self.projectiles.remove(p)
                    hit = True; break
                
                # Update position temporarily for collision check
                p["x"] = new_x
                p["y"] = new_y
                
                # 3. Check Tank Collision
                # Identify Target
                target = None
                shooter_stats = None
                
                if p["role"] == "enemy": 
                    target = self.player
                    shooter_stats = self.stats["enemy"]
                elif p["role"] == "player": 
                    target = self.enemy
                    shooter_stats = self.stats["player"]
                
                if target and target.alive:
                    # Simple AABB / Radius check
                    # Hitbox is roughly 1.0 unit (tank size). Threshold 0.6 implies center-to-center check.
                    if abs(p["x"] - target.x) < 0.6 and abs(p["y"] - target.y) < 0.6:
                        target.hp -= 1
                        if shooter_stats: shooter_stats["hits"] += 1
                        
                        try:
                            self.projectiles.remove(p)
                        except ValueError: pass # Already removed?
                        
                        hit = True; break
            
            if hit: continue

    def _apply_action(self, actor, target, action, role):
        if action == 0:
            actor.turn_left()
        elif action == 1:
            actor.turn_right()
        elif action == 2:
            actor.move_forward(self.grid)
        elif action == 3:
            self._shoot(actor, role)

    def _shoot(self, shooter, role):
        if self.cooldowns[role] > 0:
            return 
        
        # Camping Penalty
        current_pos = (shooter.x, shooter.y)
        if self.last_shot_pos[role] == current_pos:
            self.step_penalties[role] -= 0.1
        self.last_shot_pos[role] = current_pos
            
        self.cooldowns[role] = self.RELOAD_TIME
        self.stats[role]["shots"] += 1
        dx, dy = self._get_direction_vector(shooter.direction)
        # Spawn bullet at center of tank + offset
        self.projectiles.append({
            "x": shooter.x + dx * 0.6, 
            "y": shooter.y + dy * 0.6,
            "dx": dx, 
            "dy": dy,
            "role": role
        })

    def _get_direction_vector(self, direction):
        if direction == "N": return (0, -1)
        if direction == "S": return (0, 1)
        if direction == "E": return (1, 0)
        if direction == "W": return (-1, 0)
        return (0,0)

    def _get_sensor_data(self, actor):
        """Returns [Wall_Fwd, Wall_Left, Wall_Right] as 0/1"""
        obs = []
        dirs = ["N", "E", "S", "W"]
        curr_idx = dirs.index(actor.direction)
        
        # Front, Left, Right indices relative to current
        check_dirs = [
            dirs[curr_idx], # Front
            dirs[(curr_idx - 1) % 4], # Left
            dirs[(curr_idx + 1) % 4]  # Right
        ]
        
        for d in check_dirs:
            dx, dy = self._get_direction_vector(d)
            nx, ny = actor.x + dx, actor.y + dy
            if not (self.grid.in_bounds(nx, ny) and self.grid.is_free(nx, ny)):
                obs.append(1) # Blocked
            else:
                obs.append(0) # Free
        return obs

    def game_time(self):
        return round(self.steps * (1.0/6.0), 1)

    def _get_obs(self):
        # [p_x, p_y, p_dir, p_hp, e_x, e_y, e_dir, e_hp] + [p_s1, p_s2, p_s3] + [e_s1, e_s2, e_s3]
        base_p = [
            self.player.x,
            self.player.y,
            ["N", "E", "S", "W"].index(self.player.direction),
            self.player.hp,
            self.enemy.x,
            self.enemy.y,
            ["N", "E", "S", "W"].index(self.enemy.direction),
            self.enemy.hp
        ]
        
        sensors_p = self._get_sensor_data(self.player)
        sensors_e = self._get_sensor_data(self.enemy)
        
        return base_p + sensors_p + sensors_e
