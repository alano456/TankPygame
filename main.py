import time
import random
import pygame
import sys
import json
import os
import pickle
import numpy as np
from env.tank_env import TankEnv
from render.pygame_renderer import PygameRenderer, Button
from game.grid import Grid
from game.enemy import EnemyBot
from agents.dqn_agent import DQNAgent
from stable_baselines3 import DQN
from agents.wrapper import HunterDeepWrapper
from agents.sarsa_agent import SarsaAgent
from collections import deque
from agents.q_agent import QLearningAgent
import math

# --- HUNTER AGENT & HELPERS ---
class HunterAgent:
    def __init__(self, action_space_size, alpha=0.1, gamma=0.95):
        self.q_table = {}
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return str(state)

    def get_action(self, state, epsilon=0.0):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            # Default to zeros if unknown state (exploration or safe fallback)
            return random.randint(0, self.action_space_size - 1)
        
        values = self.q_table[state_key]
        max_val = np.max(values)
        best_actions = [i for i, v in enumerate(values) if v == max_val]
        return random.choice(best_actions)

class QLearningAgent:
    def __init__(self, action_space_size, alpha=0.1, gamma=0.99):
        self.q_table = {}
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return str(state)

    def get_action(self, state, epsilon=0.0):
        # Action Masking (blokada strzału na cooldownie)
        can_shoot = state[-1] # Ostatni element stanu to cooldown (1=tak, 0=nie) - Wait, in train_qlearning it is (0=tak if env.cooldown==0)
   
        valid_actions = [0, 1, 2]
        if can_shoot == 1: valid_actions.append(3)

        if random.uniform(0, 1) < epsilon:
            return random.choice(valid_actions)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            return random.choice(valid_actions)
        
        values = self.q_table[state_key]
        masked_values = np.copy(values)
        if can_shoot == 0:
            masked_values[3] = -float('inf')

        max_val = np.max(masked_values)
        best_actions = [i for i, v in enumerate(masked_values) if v == max_val]
        return random.choice(best_actions)

def get_angle_to_enemy(p_dir_idx, px, py, ex, ey):

    dir_angles = {0: -90, 1: 0, 2: 90, 3: 180}
    current_angle = dir_angles.get(p_dir_idx, 0)

    dx = ex - px
    dy = ey - py
    target_angle_rad = math.atan2(dy, dx)
    target_angle_deg = math.degrees(target_angle_rad)
    
    diff = (target_angle_deg - current_angle + 180) % 360 - 180
    return diff

def check_hitscan_inference(px, py, p_dir_idx, ex, ey, env):

    
    dx, dy = 0, 0
    # Dir Idx: 0=N, 1=E, 2=S, 3=W
    if p_dir_idx == 0: dy = -1
    elif p_dir_idx == 2: dy = 1
    elif p_dir_idx == 1: dx = 1
    elif p_dir_idx == 3: dx = -1
    
    cx, cy = px + dx, py + dy
    
    # Raymarching (limit 20 steps)
    for _ in range(20):
        if not (0 <= cx < env.width and 0 <= cy < env.height):
            break # Out of bounds
            
        if env.grid.cells[int(cy)][int(cx)] == Grid.WALL:
            return False # Hit Wall
            
        # Hit Enemy check (Distance < 1.0)
        if abs(cx - ex) < 1.0 and abs(cy - ey) < 1.0:
            return True # Hit Enemy
            
        cx += dx
        cy += dy
        
    return False

def get_hunter_state(obs, env):
    px, py, p_dir_idx = obs[0], obs[1], obs[2]
    ex, ey = obs[4], obs[5]
    
    # 1. Angle
    angle_diff = get_angle_to_enemy(p_dir_idx, px, py, ex, ey)
    angle_bin = int((angle_diff + 180 + 11.25) % 360 // 22.5)
    
    # 2. LOS (Hitscan)
    has_los = 1 if check_hitscan_inference(px, py, p_dir_idx, ex, ey, env) else 0
    
    # 3. Distance
    dist = abs(px - ex) + abs(py - ey)
    if dist < 3: dist_bin = 0
    elif dist < 8: dist_bin = 1
    else: dist_bin = 2
   
    pass
    

def get_q_hunter_state_inference(obs, env, role):
    # obs is already relative (swapped if enemy)
    px, py, p_dir_idx = obs[0], obs[1], obs[2]
    ex, ey = obs[4], obs[5]

    # 1. Angle (16 sectors)
    angle_diff = get_angle_to_enemy(p_dir_idx, px, py, ex, ey)
    angle_bin = int((angle_diff + 180 + 11.25) % 360 // 22.5)
    
    # 2. LOS
    has_los = check_hitscan_inference(px, py, p_dir_idx, ex, ey, env)
    
    # 3. Dist
    dist = abs(px - ex) + abs(py - ey)
    if dist < 3: dist_bin = 0
    elif dist < 9: dist_bin = 1
    else: dist_bin = 2
    
    # 4. Wall Ahead
    dx, dy = 0, 0
    if p_dir_idx == 0: dy = -1
    elif p_dir_idx == 2: dy = 1
    elif p_dir_idx == 1: dx = 1
    elif p_dir_idx == 3: dx = -1
    
    wall_ahead = 1
   
    # So `px + dx` works directly.
    fx, fy = int(px + dx), int(py + dy)
    if 0 <= fx < env.width and 0 <= fy < env.height:
        if env.grid.cells[fy][fx] != Grid.WALL: wall_ahead = 0

    # 5. Cooldown
    # We rely on env.cooldowns[role]
    can_shoot = 1 if env.cooldowns[role] == 0 else 0
    
    return (angle_bin, has_los, dist_bin, wall_ahead, can_shoot)

# Helper for State Discretization (Must match train_sarsa.py!)
def get_discrete_state(obs, env, role="player"):
    px, py, p_dir = obs[0], obs[1], obs[2]
    ex, ey = obs[4], obs[5]

    # 1. Angle to target (Relative)
    dx = ex - px
    dy = ey - py
    angle_to_target = math.degrees(math.atan2(dy, dx))
    angle_diff = (angle_to_target - p_dir + 180) % 360 - 180
    
    # 8-Sector Binning (matches SarsaAgent)
    dir_bin = int((angle_diff + 22.5) % 360 / 45)

    # 2. Distance (Manhattan Simplified) - Grid Units!
    dist = abs(dx) + abs(dy)
    if dist < 4:   dist_bin = 0  
    elif dist < 10: dist_bin = 1 
    else:            dist_bin = 2  

    # 3. Obstacle Ahead (Raycast 1 tile) - Grid Units!
    rad = math.radians(p_dir)
    # Check point 1.0 tiles ahead
    front_x = px + math.cos(rad) * 1.0 
    front_y = py + math.sin(rad) * 1.0
    
    obstacle_ahead = 0
    # Check Bounds (env.width is grid units)
    if not (0 <= front_x < env.width and 0 <= front_y < env.height): 
             obstacle_ahead = 1
    
    # Check Wall (Grid Coords)
    if obstacle_ahead == 0:
        gx, gy = int(front_x), int(front_y) # Direct cast to int
        # bounds check again just in case
        if 0 <= gx < env.width and 0 <= gy < env.height:
             if env.grid.cells[gy][gx] == 1:
                 obstacle_ahead = 1
        else:
             obstacle_ahead = 1

    # 4. Line of Sight
    has_los = 1 if env.grid.clear_line(px, py, ex, ey) else 0

    # 5. Cooldown
    # Now using role!
    cooldown = 1 if env.cooldowns[role] > 0 else 0
  
    gpx, gpy = px/40, py/40
    dist_x = min(gpx, env.width - gpx)
    dist_y = min(gpy, env.height - gpy)
    is_trapped = 1 if (dist_x < 2 and dist_y < 2) else 0

    return (dir_bin, dist_bin, obstacle_ahead, has_los, cooldown, is_trapped)

def get_dqn_inference_state(obs, env, role):
    # obs: [px, py, p_dir_idx, p_cooldown, ex, ey, e_dir_idx, e_cooldown] (Relative!)
    px, py, p_dir_idx = obs[0], obs[1], obs[2]
    ex, ey = obs[4], obs[5]

    # 1. Sin/Cos Angle
    dx = ex - px
    dy = ey - py
    target_rad = math.atan2(dy, dx)
   
    
    curr_rad = 0
    if p_dir_idx == 0: curr_rad = -1.57
    elif p_dir_idx == 1: curr_rad = 0
    elif p_dir_idx == 2: curr_rad = 1.57
    elif p_dir_idx == 3: curr_rad = 3.14
    
    rel_rad = target_rad - curr_rad
    sin_a = math.sin(rel_rad)
    cos_a = math.cos(rel_rad)
    

    
    dist_pixels = math.hypot(dx, dy)
    dist_blocks = dist_pixels / 40.0

    
    # 3. Cooldown
    cd = 1.0 if env.cooldowns[role] > 0 else 0.0
    
    # 4. Wall
    wall = 0.0 # Placeholder
    
    return np.array([sin_a, cos_a, dist_norm, cd, wall], dtype=np.float32)

# State Constants
MENU = "MENU"
EDITOR = "EDITOR"
GAME = "GAME"

class GameApp:
    def __init__(self):
        # Default Config
        self.map_size = 20
        self.p1_type = "Human"
        self.p2_type = "RuleBased"
        
        self.agent_types = ["Human", "RuleBased", "DQN", "SARSA", "Q-Learning"]
        self.bot_types = ["RuleBased", "DQN", "SARSA", "Q-Learning"]
        
        self.dqn_wrapper = None
        
        # Global Objects
        self.renderer = PygameRenderer(self.map_size, self.map_size)
        self.env = None
        self.editor_grid = Grid(self.map_size, self.map_size)
        
        self.state = MENU
        self.init_menu()

    def init_menu(self):
        cx = self.renderer.screen.get_width() // 2 - 100
        cy = 150
        self.buttons = [
            Button(cx, cy, 200, 40, "Play", self.start_game),
            Button(cx, cy+50, 200, 40, "Map Editor", self.go_to_editor),
            Button(cx, cy+100, 200, 40, f"P1: {self.p1_type}", self.toggle_p1),
            Button(cx, cy+150, 200, 40, f"P2: {self.p2_type}", self.toggle_p2),
            Button(cx, cy+200, 200, 40, f"Size: {self.map_size}", self.toggle_size),
            Button(cx, cy+250, 200, 40, "Quit", self.quit_game)
        ]

    def update_menu_text(self):
        self.buttons[2].text = f"P1: {self.p1_type}"
        self.buttons[3].text = f"P2: {self.p2_type}"
        self.buttons[4].text = f"Size: {self.map_size}"

    def toggle_p1(self):
        idx = self.agent_types.index(self.p1_type)
        self.p1_type = self.agent_types[(idx + 1) % len(self.agent_types)]
        self.update_menu_text()

    def toggle_p2(self):
        idx = self.bot_types.index(self.p2_type)
        self.p2_type = self.bot_types[(idx + 1) % len(self.bot_types)]
        self.update_menu_text()

    def toggle_size(self):
        sizes = [12, 16, 20, 32]
        try:
            idx = sizes.index(self.map_size)
            self.map_size = sizes[(idx + 1) % len(sizes)]
        except:
            self.map_size = 16
        
        self.update_menu_text()
        # Re-init renderer and editor grid for new size
        pygame.quit()
        self.renderer = PygameRenderer(self.map_size, self.map_size)
        self.editor_grid = Grid(self.map_size, self.map_size)
        self.init_menu() # Re-center buttons

    def load_agent(self, agent_type):
        if agent_type == "Human": return None
        if agent_type == "RuleBased": return EnemyBot()
        
        state_dim = 14
        action_size = 4
        agent = None
        
        if agent_type == "DQN":
             path = os.path.join("logs_dqn", "dqn_hunter_model.zip")
             try:
                 if os.path.exists(path):
                     # Load using SB3
                     agent = DQN.load(path)
                     print(f"Loaded SB3 DQN from {path}")
                     # Attach a memory buffer for FrameStacking
                     agent.frame_buffer = deque(maxlen=4)
                 else:
                     print(f"DQN not found at {path}, using random.")
                     agent = DQNAgent(state_dim, action_size) 
                     agent.frame_buffer = deque(maxlen=4)
             except Exception as e:
                 print(f"Error loading DQN: {e}")
                 agent = DQNAgent(state_dim, action_size)
                 agent.frame_buffer = deque(maxlen=4)

        elif agent_type == "SARSA":
             try:
                # Try Hunter first
                path = os.path.join("logs_hunter", "hunter_agent.pkl")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        agent = pickle.load(f)
                    print(f"Hunter Agent loaded from {path}")
                else:
                    # Fallback
                    with open("logs_sarsa/sarsa_agent.pkl", "rb") as f:
                        agent = pickle.load(f)
                    print("SARSA Agent loaded from logs_sarsa.")
             except FileNotFoundError:
                print("Agent not found. Using random.")
                agent = HunterAgent(action_size) # Default empty hunter
             except Exception as e:
                 print(f"Error loading SARSA: {e}")
                 agent = HunterAgent(action_size)

        elif agent_type == "Q-Learning":
             try:
                path = os.path.join("logs_q_hunter", "q_hunter_agent.pkl")
                if os.path.exists(path):
                    agent = QLearningAgent(action_size)
                    with open(path, "rb") as f:
                        q_data = pickle.load(f)
                        if isinstance(q_data, dict):
                            agent.q_table = q_data
                        else:
                            agent = q_data 
                    print(f"Q-Hunter loaded from {path}")
                else:
                    print("Q-Hunter not found. Using random.")
                    agent = QLearningAgent(action_size)
             except Exception as e:
                 print(f"Error loading Q-Learning: {e}")
                 agent = QLearningAgent(action_size)
                 
        return agent

    def get_bot_action(self, agent, agent_type, obs, env, role):
        if agent is None: return 0
        
        if agent_type == "RuleBased":
            # Map 1-4 to 0-3 (or keep 0 as -1?)
            # Original RuleBased returns: 0(Idle), 1(L), 2(R), 3(F), 4(S)
            # We want: 0(L), 1(R), 2(F), 3(S).
            # If 0(Idle) -> return -1? (Idle)
            # If 1 -> 0
            act = 0
            if role == "player":
                act = agent.choose_action(env.player, env.enemy, env.grid, env.cooldowns["player"])
            else:
                act = agent.choose_action(env.enemy, env.player, env.grid, env.cooldowns["enemy"])
            
            if act == 0: return -1 # Idle
            return act - 1 # Shift 1-4 down to 0-3
        
        elif agent_type in ["DQN", "SARSA", "Q-Learning"]:
            state = obs
            if role == "enemy":
                # Invert state for P2: Swap Base (0-3 <-> 4-7) AND Sensors (8-10 <-> 11-13)
                state = [
                    obs[4], obs[5], obs[6], obs[7], # Enemy as Player
                    obs[0], obs[1], obs[2], obs[3], # Player as Enemy
                    obs[11], obs[12], obs[13],      # Enemy Sensors as Player Sensors
                    obs[8], obs[9], obs[10]         # Player Sensors as Enemy Sensors
                ]
            
            if agent_type == "DQN":
                # Use Shared Wrapper
                if hasattr(self, 'dqn_wrapper') and self.dqn_wrapper:
                    dqn_state = self.dqn_wrapper._get_neural_state(role)
                else:
                    dqn_state = HunterDeepWrapper(env)._get_neural_state(role)
                
                # Update Buffer
                if not hasattr(agent, "frame_buffer"): agent.frame_buffer = deque(maxlen=4)
                
                if len(agent.frame_buffer) == 0:
                    for _ in range(4): agent.frame_buffer.append(dqn_state)
                else:
                    agent.frame_buffer.append(dqn_state)
                
           
                stacked_state = np.concatenate(list(agent.frame_buffer))
                
                # Predict
                action, _ = agent.predict(stacked_state, deterministic=True)
                act = int(action)
                return act

            elif agent_type == "SARSA":
                pass # Handled below with discrete state
            else:
                act = agent.get_action(tuple(state), epsilon=0)
            
        # SARSA / Hunter Logic
        if agent_type == "SARSA":
         
             q_table = agent.q_table if hasattr(agent, "q_table") else agent
             
             # Calculate state
             can_shoot = 1 if env.cooldowns[role] == 0 else 0
             
             px, py, p_dir = state[0], state[1], state[2]
             ex, ey = state[4], state[5]
             
             angle_diff = get_angle_to_enemy(p_dir, px, py, ex, ey)
             angle_bin = int((angle_diff + 180 + 11.25) % 360 // 22.5)
             
             has_los = 1 if check_hitscan_inference(px, py, p_dir, ex, ey, env) else 0
             
             dist = abs(px - ex) + abs(py - ey)
             if dist < 3: dist_bin = 0
             elif dist < 8: dist_bin = 1
             else: dist_bin = 2
             
             hunter_state = (angle_bin, has_los, dist_bin, can_shoot)
             state_key = str(hunter_state)
             
             if state_key in q_table:
                 values = q_table[state_key]
                 act = int(np.argmax(values))
             else:
                 act = random.randint(0, 3)
                 
             return act

        if agent_type == "Q-Learning":
             # Use Q-Hunter State
             qs = get_q_hunter_state_inference(state, env, role)
             return agent.get_action(qs, epsilon=0)
        
        return 0

    def start_game(self):
        self.state = GAME
        self.env = TankEnv(self.map_size, self.map_size)
        # Pass editor layout to reset
        layout = self.editor_grid.cells
        self.env.reset(grid_layout=layout)
        
        self.p1_bot = self.load_agent(self.p1_type)
        self.p2_bot = self.load_agent(self.p2_type)
        
        # Init Wrapper (single instance, reuses env)
        self.dqn_wrapper = HunterDeepWrapper(self.env)
        
        # Reset DQN Buffers
        if self.p1_type == "DQN" and hasattr(self.p1_bot, "frame_buffer"): self.p1_bot.frame_buffer.clear()
        if self.p2_type == "DQN" and hasattr(self.p2_bot, "frame_buffer"): self.p2_bot.frame_buffer.clear()

    def go_to_editor(self):
        self.state = EDITOR
        # Keep current editor_grid

    def quit_game(self):
        pygame.quit()
        sys.exit()

    def run(self):
        while True:
            if self.state == MENU:
                self.run_menu()
            elif self.state == EDITOR:
                self.run_editor()
            elif self.state == GAME:
                self.run_game()

    def run_menu(self):
        self.renderer.clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
            
            for btn in self.buttons:
                if btn.handle_event(event):
                    break
                    
        self.renderer.draw_menu(f"{self.p1_type} vs {self.p2_type}", self.map_size)
        for btn in self.buttons:
            btn.draw(self.renderer.screen, self.renderer.font)
        pygame.display.flip()

    def run_editor(self):
        self.renderer.clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.start_game()
                if event.key == pygame.K_c:
                    self.editor_grid = Grid(self.map_size, self.map_size)
                if event.key == pygame.K_ESCAPE:
                    self.state = MENU
                
                # Save Map
                if event.key == pygame.K_s:
                    try:
                        with open("custom_map.json", "w") as f:
                            json.dump(self.editor_grid.cells, f)
                        print("Map saved to custom_map.json")
                    except Exception as e:
                        print(f"Error saving map: {e}")

                # Load Map
                if event.key == pygame.K_l:
                    try:
                        with open("custom_map.json", "r") as f:
                            data = json.load(f)
                            # Verify size compatibility roughly or resize grid
                            if len(data) == self.map_size and len(data[0]) == self.map_size:
                                self.editor_grid.cells = data
                                print("Map loaded.")
                            else:
                                print(f"Map size mismatch! Current: {self.map_size}, File: {len(data)}")
                    except FileNotFoundError:
                        print("No saved map found.")
                    except Exception as e:
                        print(f"Error loading map: {e}")
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    mx, my = event.pos
                    gx, gy = int(mx // self.renderer.cell_size), int(my // self.renderer.cell_size)
                    if self.editor_grid.in_bounds(gx, gy):
                        # Toggle Wall
                        if self.editor_grid.cells[gy][gx] == Grid.WALL:
                            self.editor_grid.cells[gy][gx] = Grid.EMPTY
                        else:
                            self.editor_grid.cells[gy][gx] = Grid.WALL

        self.renderer.draw_editor(self.editor_grid)
        pygame.display.flip()

    def run_game(self):
        LOGIC_FPS = 6
        STEP = 1 / LOGIC_FPS
        last = time.time()
        
        while self.state == GAME:
            dt = self.renderer.clock.tick(60) / 1000
            
            # Input
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT: self.quit_game()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    self.state = MENU
                    return # Exit game loop
                if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    # Restart with same map
                    layout = self.editor_grid.cells
                    self.env.reset(grid_layout=layout)

            # Logic Update
            if time.time() - last > STEP:
                
                obs = self.env._get_obs()
                
                # Player 1 Action
                action_p1 = -1 # Default Idle (TankEnv ignores < 0)
                if self.p1_type == "Human":
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_a]: action_p1 = 0
                    elif keys[pygame.K_d]: action_p1 = 1
                    elif keys[pygame.K_w]: action_p1 = 2
                    elif keys[pygame.K_SPACE]: action_p1 = 3
                else:
                    action_p1 = self.get_bot_action(self.p1_bot, self.p1_type, obs, self.env, "player")

                # Player 2 Action
                action_p2 = self.get_bot_action(self.p2_bot, self.p2_type, obs, self.env, "enemy")
                
                self.env.step(action_p1, action_p2)
                last = time.time()
                
                # Check Game Over
                if self.env.player.hp <= 0 or self.env.enemy.hp <= 0:
                     winner = "Player 1" if self.env.enemy.hp <= 0 else "Player 2"
                     
                     # Draw final state
                     self.renderer.draw(self.env.grid, self.env.player, self.env.enemy)
                     self.renderer.draw_projectiles(self.env.projectiles)
                     
                     # Draw Game Over Text
                     font = pygame.font.SysFont('Arial', 64, bold=True)
                     text = font.render("GAME OVER", True, (255, 0, 0))
                     text_rect = text.get_rect(center=(self.renderer.screen.get_width()//2, self.renderer.screen.get_height()//2 - 50))
                     
                     win_text = self.renderer.font.render(f"{winner} Wins!", True, (255, 255, 255))
                     win_rect = win_text.get_rect(center=(self.renderer.screen.get_width()//2, self.renderer.screen.get_height()//2 + 20))
                     
                     self.renderer.screen.blit(text, text_rect)
                     self.renderer.screen.blit(win_text, win_rect)
                     pygame.display.flip()
                     
                     time.sleep(2)
                     self.state = MENU
                     return

            # Render
            self.renderer.update(dt, self.env.player, self.env.enemy)
            self.renderer.draw(self.env.grid, self.env.player, self.env.enemy)
            self.renderer.draw_projectiles(self.env.projectiles)
            self.renderer.draw_panel(self.env)
            pygame.display.flip()

if __name__ == "__main__":
    app = GameApp()
    app.run()

