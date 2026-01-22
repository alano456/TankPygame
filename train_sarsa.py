import numpy as np
import csv
import os
import time
import math
import random
import pickle
from env.tank_env import TankEnv, Grid
from analyze_training import analyze_custom

#KONFIGURACJA 
ALPHA = 0.1          
GAMMA = 0.95         
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPISODES = 30000     
MAX_STEPS = 300      

# Liniowy spadek Epsilona
epsilon_decrement = (EPSILON_START - EPSILON_MIN) / (EPISODES * 0.90)

LOG_DIR = "logs_hunter"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")

# Run ID Detection
run_id = 1
while os.path.exists(os.path.join(LOG_DIR, f"report_{run_id}.md")):
    run_id += 1

class HunterAgent:
    def __init__(self, action_space_size, alpha, gamma):
        self.q_table = {}
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, state):
        return str(state)

    def get_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        
        # Randomizacja przy równych wartościach (żeby nie faworyzował akcji 0)
        values = self.q_table[state_key]
        max_val = np.max(values)
        best_actions = [i for i, v in enumerate(values) if v == max_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done, next_action):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space_size)

        current_q = self.q_table[state_key][action]
        next_q = self.q_table[next_state_key][next_action]
        
        # Wzór SARSA
        if done:
             target = reward
        else:
             target = reward + self.gamma * next_q
             
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state_key][action] = new_q

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

# --- MATEMATYKA FIZYKI ---

def get_angle_to_enemy(p_dir_idx, px, py, ex, ey):
    """
    Oblicza błąd kątowy: 0 = idealnie na wprost.
    Zwraca wartość od -180 (cel z tyłu/lewo) do 180 (cel z tyłu/prawo).
    """
    # Mapowanie indeksu kierunku (0-3) na wektor. 
    # TankEnv: ["N", "E", "S", "W"] -> 0, 1, 2, 3
    # Ale uwaga: W Pygame Y rośnie w dół!
    # N(0,-1), E(1,0), S(0,1), W(-1,0)
    
    dir_angles = {0: -90, 1: 0, 2: 90, 3: 180} # Stopnie na kole trygonometrycznym
    current_angle = dir_angles.get(p_dir_idx, 0)

    dx = ex - px
    dy = ey - py
    
    # Atan2 zwraca kąt wektora do celu
    target_angle_rad = math.atan2(dy, dx) 
    target_angle_deg = math.degrees(target_angle_rad)
    
    # Różnica
    diff = (target_angle_deg - current_angle + 180) % 360 - 180
    return diff

def check_hitscan(env):
    px, py = env.player.x, env.player.y
    ex, ey = env.enemy.x, env.enemy.y
    p_dir = env.player.direction # "N", "E", "S", "W"
    
    dx, dy = 0, 0
    if p_dir == "N": dy = -1
    elif p_dir == "S": dy = 1
    elif p_dir == "E": dx = 1
    elif p_dir == "W": dx = -1
    
    cx, cy = px + dx, py + dy
    
    # Raymarching
    while 0 <= cx < env.width and 0 <= cy < env.height:
        # Sprawdź ścianę
        if env.grid.cells[int(cy)][int(cx)] == Grid.WALL:
            return False, False # Hit Wall
        
        # Sprawdź wroga (zakładamy hitbox 1x1)
        if int(cx) == int(ex) and int(cy) == int(ey):
            return True, False # Hit Enemy
            
        cx += dx
        cy += dy
        
    return False, False # Miss (out of bounds)

# STAN I NAGRODY 

def get_hunter_state(obs, env):
    """
  
    """
    px, py, p_dir_idx = obs[0], obs[1], obs[2]
    ex, ey = obs[4], obs[5]
    
    # 1. Kąt (Precyzja to klucz)
    # Dzielimy 360 stopni na 16 sektorów.
    # Sektor 0 = Idealnie prosto.
    angle_diff = get_angle_to_enemy(p_dir_idx, px, py, ex, ey)
    angle_bin = int((angle_diff + 180 + 11.25) % 360 // 22.5)
    
    # 2. Czy widzę wroga? (Line of Sight)
    has_los, _ = check_hitscan(env) # Wykorzystujemy funkcję hitscan jako sensor
    
    # 3. Dystans (Blisko/Średnio/Daleko)
    dist = abs(px - ex) + abs(py - ey)
    if dist < 3: dist_bin = 0
    elif dist < 8: dist_bin = 1
    else: dist_bin = 2
    
    # 4. Cooldown
    can_shoot = 1 if env.cooldowns['player'] == 0 else 0
    
    return (angle_bin, has_los, dist_bin, can_shoot)

def train_hunter():
    # Inicjalizacja Env
    env = TankEnv()
    
    # Inicjalizacja Agenta
    agent = HunterAgent(action_space_size=4, alpha=ALPHA, gamma=GAMMA)
    
    STEPS_LOG_FILE = os.path.join(LOG_DIR, "steps_log.csv")
    
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["episode", "reward", "win", "epsilon", "q_size", "avg_q", "steps"])
    
    
    with open(STEPS_LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["episode", "step", "px", "py", "died"])

    epsilon = EPSILON_START
    start_time = time.time()
    print(f"--- HUNTER PROTOCOL ENGAGED (RUN #{run_id}) ---")

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        
        state = get_hunter_state(obs, env)
        action = agent.get_action(state, epsilon)
        
        total_reward = 0
        done = False
        steps = 0
        
        # Do shapingu (nagroda za poprawę kąta)
        last_angle_abs = abs(get_angle_to_enemy(obs[2], obs[0], obs[1], obs[4], obs[5]))
        
        episode_steps = []
        should_log_steps = (episode % 50 == 0)
        if should_log_steps: episode_steps.append([episode, steps, obs[0], obs[1], 0])

        while not done and steps < MAX_STEPS:
            
            real_action = action
            hit_enemy = False
            
            if action == 3: # Strzał
                if env.cooldowns['player'] == 0:
                    hit_enemy, hit_wall = check_hitscan(env)
                    env.cooldowns['player'] = env.RELOAD_TIME
                    real_action = -1 # Pusta akcja dla env (tylko ruch wroga)
                else:
                    real_action = -1 # Próba strzału na cooldownie
            

            if real_action == -1: 
              
                pass

            next_obs, _, terminated, _, _ = env.step(action if action != 3 else 3) # Env tworzy pocisk, my go ignorujemy
            
            # NALICZANIE NAGRÓD (Shaping) 
            reward = 0.0
            
            # 1. Kąt (Kompas)
            # Czy po ruchu patrzę bardziej na wroga?
            curr_angle_abs = abs(get_angle_to_enemy(next_obs[2], next_obs[0], next_obs[1], next_obs[4], next_obs[5]))
            
            if curr_angle_abs < last_angle_abs:
                reward += 0.2  
            elif curr_angle_abs > last_angle_abs:
                reward -= 0.3  
                
            last_angle_abs = curr_angle_abs
            
            # 2. Hitscan Kill
            if hit_enemy:
                reward += 100.0 
                terminated = True
                env.enemy.hp = 0 
            
            # 3. Strzał w nic
            if action == 3 and not hit_enemy:
                reward -= 2.0 
            
            # 4. Kolizje (zakładamy że pozycja się nie zmieniła przy akcji ruchu)
            if action in [0,1,2] and obs[0] == next_obs[0] and obs[1] == next_obs[1]:
                reward -= 1.0 # Ściana
            
            # 5. Śmierć gracza (jeśli wróg nas trafił normalnym pociskiem z env)
            if env.player.hp <= 0:
                reward -= 50.0
                terminated = True

            reward -= 0.01 # Presja czasu

            # SARSA Update
            next_state = get_hunter_state(next_obs, env)
            next_action = agent.get_action(next_state, epsilon)
            
            agent.update(state, action, reward, next_state, terminated, next_action)
            
            state = next_state
            action = next_action
            obs = next_obs
            
            total_reward += reward
            steps += 1
            done = terminated
            
            if should_log_steps:
                 died_flag = 1 if (done and env.player.hp <= 0) else 0
                 episode_steps.append([episode, steps, next_obs[0], next_obs[1], died_flag])

        # Decay
        if epsilon > EPSILON_MIN:
            epsilon -= epsilon_decrement
            
        win = 1 if env.enemy.hp <= 0 else 0
        
        if should_log_steps and episode_steps:
             with open(STEPS_LOG_FILE, 'a', newline='') as f:
                 writer = csv.writer(f)
                 writer.writerows(episode_steps)
        
        if episode % 100 == 0:
            q_size = len(agent.q_table)
            
            # Avg Q calc
            avg_q = 0
            if q_size > 0:
                 keys = list(agent.q_table.keys())
                 # sampling if too large
                 sample = keys[:100] if len(keys)>100 else keys
                 avg_q = np.mean([np.max(agent.q_table[k]) for k in sample])

            print(f"Hunter Ep {episode} | R={total_reward:.1f} | Win={win} | Q={q_size} | Eps={epsilon:.2f}")
            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([episode, round(total_reward, 2), win, round(epsilon, 3), q_size, round(avg_q, 3), steps])

        # Regular Analysis
        if episode % 5000 == 0:
            try:
                analyze_custom(LOG_DIR, time.time() - start_time, run_id=run_id)
            except Exception as e:
                print(f"Analysis failed: {e}")

    agent.save(os.path.join(LOG_DIR, "hunter_agent.pkl"))
    analyze_custom(LOG_DIR, time.time() - start_time, run_id=run_id)
    
    import shutil
    shutil.copy(LOG_FILE, os.path.join(LOG_DIR, f"training_log_{run_id}.csv"))
    
    print("Training Complete. Hunter is ready.")

if __name__ == "__main__":
    train_hunter()