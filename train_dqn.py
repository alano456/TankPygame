import gymnasium as gym
import numpy as np
import math
import os
import csv
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env.tank_env import TankEnv, Grid
from analyze_training import analyze_custom
from agents.wrapper import HunterDeepWrapper

# --- KONFIGURACJA LOGÓW ---
LOG_DIR = "logs_dqn"
os.makedirs(LOG_DIR, exist_ok=True)

run_id = 1
while os.path.exists(os.path.join(LOG_DIR, f"report_{run_id}.md")):
    run_id += 1

LOG_FILE = os.path.join(LOG_DIR, "training_log.csv")
STEPS_LOG_FILE = os.path.join(LOG_DIR, "steps_log.csv")


class AnalysisCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, run_id):
        super().__init__(verbose=1)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.run_id = run_id
        self.start_time = time.time()
        self.last_len = 0
        self.last_analysis_step = 0

    def _on_step(self):
        if self.num_timesteps % 20 == 0:
             # get_attr from VecEnv
             players = self.training_env.get_attr("player")
             px, py = players[0].x, players[0].y
             
             died = 0
             dones = self.locals['dones']
             if dones[0]:
                  if players[0].hp <= 0: died = 1

             with open(STEPS_LOG_FILE, 'a', newline='') as f:
                 csv.writer(f).writerow([1, self.num_timesteps, px, py, died])
        return True

    def _on_rollout_end(self):
        # Access Monitor info from VecEnv?
        # VecEnv doesn't expose Monitor directly usually.
        # Use info buffer?
        # Or get_attr("episode_returns") from Monitor wrapper inside VecEnv
        
        # self.training_env is VecFrameStack -> DummyVecEnv -> Monitor
        # .get_attr("episode_returns") returns list of lists (one per env)
        
        try:
            # List of lists (for each env)
            all_returns = self.training_env.get_attr("episode_returns")
            all_lengths = self.training_env.get_attr("episode_lengths")
            
            # Assuming 1 env
            returns = all_returns[0]
            lengths = all_lengths[0]
            
            current_len = len(returns)
            
            if current_len > self.last_len:
                new_eps = current_len - self.last_len
                for i in range(new_eps):
                    idx = self.last_len + i
                    rew = returns[idx]
                    win = 1 if rew > 0.8 else 0 # Threshold 0.8 since max is 1.0
                    epsilon = self.model.exploration_rate
                    steps = lengths[idx]
                    
                    with open(LOG_FILE, 'a', newline='') as f:
                        csv.writer(f).writerow([idx+1, round(rew, 2), win, round(epsilon, 3), steps])
                
                self.last_len = current_len
        except Exception as e:
            pass

        if self.num_timesteps > 0 and self.num_timesteps % 5000 == 0:
             if self.last_analysis_step != self.num_timesteps:
                self.last_analysis_step = self.num_timesteps
                try:
                    analyze_custom(self.log_dir, time.time() - self.start_time, run_id=self.run_id)
                except: pass

def train_dqn_pro():
    # 1. Tworzenie środowiska
    def make_env():
        env = TankEnv(12, 12) # Use 12x12 Map
        env = HunterDeepWrapper(env)
        env = Monitor(env) 
        return env

    # 2. Wektoryzacja i Stosowanie Klatek
    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4) 

    # Inicjalizacja CSV
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["episode", "total_reward", "win", "epsilon", "steps"])
    with open(STEPS_LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["episode", "step", "px", "py", "died"])

    # 3. Model DQN (Dostrojony)
    model = DQN(
        "MlpPolicy", 
        vec_env,
        verbose=1,
        learning_rate=1e-4,      
        buffer_size=100000,      
        learning_starts=5000,    
        batch_size=128,          
        gamma=0.99,
        train_freq=4,            
        gradient_steps=1,
        target_update_interval=1000, 
        exploration_fraction=0.6, 
        exploration_final_eps=0.05,
        tensorboard_log=LOG_DIR
    )

    print(f"--- STARTING DQN PRO (STACKED FRAMES) RUN #{run_id} ---")
    
    callback = AnalysisCallback(check_freq=1000, log_dir=LOG_DIR, run_id=run_id)
    model.learn(total_timesteps=200000, callback=callback) 
    
    model.save(os.path.join(LOG_DIR, "dqn_hunter_model"))
    print("Model saved.")
    
    try:
        analyze_custom(LOG_DIR, time.time() - callback.start_time, run_id=run_id)
        import shutil
        shutil.copy(LOG_FILE, os.path.join(LOG_DIR, f"training_log_{run_id}.csv"))
    except: pass

if __name__ == "__main__":
    train_dqn_pro()
