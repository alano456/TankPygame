import time
import gymnasium as gym
import numpy as np
import math
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from env.tank_env import TankEnv, Grid
from agents.wrapper import HunterDeepWrapper

# Wrapper should be imported from agents.wrapper

def test_agent():
    # 1. Tworzymy środowisko DOKŁADNIE tak jak w treningu
    env = TankEnv(12, 12) # Match training map size
    env = HunterDeepWrapper(env) # Nakładamy "okulary" (normalizacja)
    
    # 2. Jeśli w treningu był FrameStack, tu też musi być!
    # SB3 wymaga opakowania w DummyVecEnv dla FrameStack
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4) # PAMIĘĆ KRÓTKOTRWAŁA

    # 3. Ładujemy model
    model_path = os.path.join("logs_dqn", "dqn_hunter_model") # Correct path
    print(f"Loading model from {model_path}...")
    
    try:
        model = DQN.load(model_path)
    except Exception as e:
        print(f"Błąd: Nie znaleziono pliku modelu lub błąd ładowania: {e}")
        return

    # 4. Pętla gry
    obs = env.reset()
    
    print("--- ROZPOCZYNAMY POKAZ (Tylko konsola, bez GUI) ---")
    
    total_matches = 0
    
    for i in range(1000): # Run for 1000 steps max
        # KLUCZOWE: deterministic=True
        # To wyłącza losowość. Agent robi to, co uważa za najlepsze.
        action, _states = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, infos = env.step(action)
        
        # Logowanie co 10 kroków
        if i % 10 == 0:
            env_unwrapped = env.envs[0].unwrapped # VecFrameStack->Dummy->Wrapper->Env
            print(f"Step {i}: Total Reward: {rewards}, Player HP: {env_unwrapped.player.hp}, Enemy HP: {env_unwrapped.enemy.hp}")
        
        time.sleep(0.01) 
        
        if dones[0]:
            total_matches += 1
            env_unwrapped = env.envs[0].unwrapped
            print(f"--- Koniec rundy {total_matches}.", end=" ")
            if env_unwrapped.enemy.hp <= 0: print("WIN!")
            elif env_unwrapped.player.hp <= 0: print("LOSS!")
            else: print("DRAW/TIMEOUT")
            obs = env.reset()
            if total_matches >= 5: break

if __name__ == "__main__":
    test_agent()
