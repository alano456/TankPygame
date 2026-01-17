import math
from env.tank_env import TankEnv
from game.grid import Grid
from game.tank import Tank
from game.enemy import EnemyBot

def test_sniper_alignment():
    print("--- Testing Approaching Sniper Alignment ---")
    env = TankEnv(15, 15)
    env.reset()
    
    # 1. Test Approaching
    # Player at (5, 5), Enemy at (12, 12). Dist ~9.9
    env.player.x, env.player.y = 5, 5
    env.enemy.x, env.enemy.y = 12, 12
    env.enemy.direction = "W" # Facing West (towards 5, 12)
    
    print(f"Start: P(5,5), E(12,12). Goal: Get to distance 5.")
    
    steps_to_close = 0
    for _ in range(30):
        obs, _, _, _, _ = env.step(-1) # Player idle
        dist = math.hypot(env.player.x - env.enemy.x, env.player.y - env.enemy.y)
        steps_to_close += 1
        if dist <= 5.5:
            print(f"Reached distance {dist:.2f} in {steps_to_close} steps.")
            break
            
    # 2. Test Alignment
    # Now at distance ~5. Let's see if it tries to align.
    print(f"Phase 2: Aligning. Current E pos: ({env.enemy.x:.2f}, {env.enemy.y:.2f})")
    
    aligned = False
    for i in range(20):
        obs, _, _, _, _ = env.step(-1)
        dx = env.player.x - env.enemy.x
        dy = env.player.y - env.enemy.y
        if abs(dx) < 0.5 or abs(dy) < 0.5:
            print(f"Aligned at step {i+1}! dx={dx:.2f}, dy={dy:.2f}")
            aligned = True
            break
            
    # 3. Test Shooting
    if aligned:
        print("Phase 3: Shooting. Waiting for projectile.")
        shot_detected = False
        for i in range(10):
            env.cooldowns['enemy'] = 0 # Ensure ready to shoot
            obs, _, _, _, _ = env.step(-1)
            if len(env.projectiles) > 0:
                print(f"Shot detected at step {i+1}!")
                shot_detected = True
                break
        if not shot_detected:
            print("FAILURE: No shot detected after alignment.")
    else:
        print("FAILURE: Did not align within 20 steps.")

if __name__ == "__main__":
    test_sniper_alignment()
