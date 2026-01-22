import numpy as np
import csv
import os
import time
import math
import random
import pickle
from env.tank_env import TankEnv, Grid
from analyze_training import analyze_custom

ALPHA = 0.1          
GAMMA = 0.99         
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPISODES = 50000 
MAX_STEPS = 400      

epsilon_decrement = (EPSILON_START - EPSILON_MIN) / (EPISODES * 0.8)

LOG_DIR = "logs_q_hunter"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

class QLearningAgent:
    def __init__(self, action_space_size, alpha, gamma):
        self.q_table = {}
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return str(state)

    def get_action(self, state, epsilon):
        # Action Masking (blokada strzału na cooldownie)
        can_shoot = state[-1] # Ostatni element stanu to cooldown (0=tak, 1=nie)
        
        valid_actions = [0, 1, 2]
        if can_shoot == 1: valid_actions.append(3)

        # Eksploracja
        if random.uniform(0, 1) < epsilon:
            return random.choice(valid_actions)
        
        # Eksploatacja
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        
        values = self.q_table[state_key]
        
        # Maskowanie w Q-table (dla wyboru najlepszej akcji)
        masked_values = np.copy(values)
        if can_shoot == 0:
            masked_values[3] = -float('inf')

        max_val = np.max(masked_values)
        best_actions = [i for i, v in enumerate(masked_values) if v == max_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_size)

        # --- SERCE Q-LEARNINGU ---
        # Bierzemy MAX z przyszłego stanu (niezależnie od tego co zrobimy)
        # Ale uwaga: Jeśli w next_state mamy cooldown, to max nie powinien uwzględniać strzału!
        
        next_can_shoot = next_state[-1]
        next_values = np.copy(self.q_table[next_state_key])
        if next_can_shoot == 0:
            next_values[3] = -float('inf')
            
        best_next_q = np.max(next_values) if not done else 0.0
        current_q = self.q_table[state_key][action]
        
        # Równanie Bellmana dla Q-Learning
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[state_key][action] = new_q

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

# --- FIZYKA I POMOCNICZE ---

def get_angle_to_enemy(p_dir_idx, px, py, ex, ey):
    dir_angles = {0: -90, 1: 0, 2: 90, 3: 180} 
    current_angle = dir_angles.get(p_dir_idx, 0)
    dx = ex - px; dy = ey - py
    target_angle = math.degrees(math.atan2(dy, dx))
    return (target_angle - current_angle + 180) % 360 - 180

def check_hitscan(env):
    px, py = env.player.x, env.player.y
    ex, ey = env.enemy.x, env.enemy.y
    p_dir = env.player.direction
    dx, dy = 0, 0
    if p_dir == "N": dy = -1
    elif p_dir == "S": dy = 1
    elif p_dir == "E": dx = 1
    elif p_dir == "W": dx = -1
    
    cx, cy = px + dx, py + dy
    while 0 <= cx < env.width and 0 <= cy < env.height:
        if env.grid.cells[int(cy)][int(cx)] == Grid.WALL: return False
        if int(cx) == int(ex) and int(cy) == int(ey): return True
        cx += dx; cy += dy
    return False

def get_q_hunter_state(obs, env):
    px, py, p_dir_idx = obs[0], obs[1], obs[2]
    ex, ey = obs[4], obs[5]
    
    # 1. Kąt (16 sektorów)
    angle_diff = get_angle_to_enemy(p_dir_idx, px, py, ex, ey)
    angle_bin = int((angle_diff + 180 + 11.25) % 360 // 22.5)
    
    # 2. Line of Sight
    has_los = check_hitscan(env)
    
    # 3. Dystans (3 strefy)
    dist = abs(px - ex) + abs(py - ey)
    if dist < 3: dist_bin = 0
    elif dist < 9: dist_bin = 1 # Sweet spot powiększony
    else: dist_bin = 2
    
    # 4. Ściana przed nosem (Ważne dla Q-Learningu, żeby nie wbijał się w mur)
    p_dir = env.player.direction
    dx, dy = 0, 0
    if p_dir == "N": dy = -1
    elif p_dir == "S": dy = 1
    elif p_dir == "E": dx = 1
    elif p_dir == "W": dx = -1
    
    fx, fy = int(px + dx), int(py + dy)
    wall_ahead = 1
    if 0 <= fx < env.width and 0 <= fy < env.height:
        if env.grid.cells[fy][fx] != Grid.WALL: wall_ahead = 0
            
    # 5. Cooldown (Gotowy=1)
    can_shoot = 1 if env.cooldowns['player'] == 0 else 0
    
    return (angle_bin, has_los, dist_bin, wall_ahead, can_shoot)

# --- PĘTLA TRENINGOWA ---

def train_q_hunter():
    env = TankEnv()
    agent = QLearningAgent(action_space_size=4, alpha=ALPHA, gamma=GAMMA)
    
    # Run ID Detection
    run_id = 1
    while os.path.exists(os.path.join(LOG_DIR, f"report_{run_id}.md")):
        run_id += 1
    print(f"--- Q-LEARNING HUNTER ENGAGED (RUN #{run_id}) ---")
    
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["episode", "reward", "win", "epsilon", "q_size"])

    STEPS_LOG_FILE = os.path.join(LOG_DIR, "steps_log.csv")
    with open(STEPS_LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["episode", "step", "px", "py", "died"])

    epsilon = EPSILON_START
    start_time = time.time()

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        state = get_q_hunter_state(obs, env)
        
        total_reward = 0
        done = False
        steps = 0
        
        steps = 0
        
        last_angle_abs = abs(get_angle_to_enemy(obs[2], obs[0], obs[1], obs[4], obs[5]))

        episode_steps = []
        should_log_steps = (episode % 50 == 0)
        if should_log_steps: episode_steps.append([episode, steps, obs[0], obs[1], 0])

        while not done and steps < MAX_STEPS:
            # W Q-Learningu wybieramy akcję TU I TERAZ
            action = agent.get_action(state, epsilon)
            
            # --- Hitscan Logic ---
            real_action = action
            hit_enemy = False
            
            if action == 3: # Strzał
                if env.cooldowns['player'] == 0:
                    hit_enemy = check_hitscan(env)
                    env.cooldowns['player'] = env.RELOAD_TIME
                    real_action = -1 # Hack na env
                else:
                    real_action = -1 

            if real_action == -1: pass
            
            # Krok środowiska
            next_obs, _, terminated, _, _ = env.step(action if action != 3 else 3)
            
            # --- NAGRODY ---
            reward = 0.0
            
            # 1. Kąt (Kompas)
            curr_angle_abs = abs(get_angle_to_enemy(next_obs[2], next_obs[0], next_obs[1], next_obs[4], next_obs[5]))
            if curr_angle_abs < last_angle_abs: reward += 0.2
            elif curr_angle_abs > last_angle_abs: reward -= 0.3
            last_angle_abs = curr_angle_abs
            
            # 2. Inteligentny Dystans (Kiting)
            if curr_angle_abs < 25: # Tylko jak patrzę na wroga
                curr_dist = abs(next_obs[4] - next_obs[0]) + abs(next_obs[5] - next_obs[1])
                if 3 <= curr_dist <= 9: reward += 0.1 # Utrzymuj dystans
                
            # 3. Ściana
            if action == 2 and state[3] == 1: # Próbował wjechać w ścianę
                reward -= 2.0 

            # 4. Walka
            if hit_enemy:
                reward += 100.0
                terminated = True
                env.enemy.hp = 0
            
            if action == 3 and not hit_enemy:
                reward -= 1.0 # Kara za pudło

            if env.player.hp <= 0:
                reward -= 50.0
                terminated = True

            reward -= 0.01

            # --- Q-UPDATE ---
            next_state = get_q_hunter_state(next_obs, env)
            
            # Kluczowa różnica: tutaj NIE wybieramy next_action epsilon-chciwie.
            # Metoda update() sama znajdzie MAX Q dla next_state.
            agent.update(state, action, reward, next_state, terminated)
            
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated
            
            if should_log_steps:
                 died_flag = 1 if (done and env.player.hp <= 0) else 0
                 episode_steps.append([episode, steps, next_obs[0], next_obs[1], died_flag])

        if epsilon > EPSILON_MIN:
            epsilon -= epsilon_decrement
            
        win = 1 if env.enemy.hp <= 0 else 0
        
        if should_log_steps and episode_steps:
             with open(STEPS_LOG_FILE, 'a', newline='') as f:
                 writer = csv.writer(f)
                 writer.writerows(episode_steps)
        
        if episode % 100 == 0:
            q_size = len(agent.q_table)
            print(f"Q-Hunter Ep {episode} | R={total_reward:.1f} | Win={win} | Q={q_size} | Eps={epsilon:.2f}")
            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([episode, round(total_reward, 2), win, round(epsilon, 3), q_size])

        # Feature: Frequent updates for heatmaps (every 5000 eps)
        if episode % 5000 == 0:
             try:
                analyze_custom(LOG_DIR, time.time() - start_time, run_id=run_id)
             except Exception as e:
                print(f"Analysis update failed: {e}")

    agent.save(os.path.join(LOG_DIR, "q_hunter_agent.pkl"))
    
    # Final Analysis with Run ID
    try:
        analyze_custom(LOG_DIR, run_id=run_id)
        # Optional: Save Log Copy
        import shutil
        shutil.copy(LOG_FILE, os.path.join(LOG_DIR, f"training_log_{run_id}.csv"))
        print(f"Saved artifacts for Run {run_id}")
    except Exception as e:
        print(f"Analysis Error: {e}")

if __name__ == "__main__":
    train_q_hunter()
