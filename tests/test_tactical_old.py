from env.tank_env import TankEnv
import time

def verify_tactical_behavior():
    print("--- Verifying TacticalBot Behavior ---")
    env = TankEnv(12, 12)
    # Start at distance 5 (Ready to shoot)
    env.reset()
    env.player.x, env.player.y = 5, 5
    env.enemy.x, env.enemy.y = 5, 10 # Dist 5
    env.enemy.direction = "N"
    env.player.hp = 10 # More HP to observe behavior
    
    print(f"Start: Player at {env.player.x, env.player.y}, Enemy at {env.enemy.x, env.enemy.y}")
    
    for i in range(15):
        # Force cooldown to 0 for enemy to see shots
        env.cooldowns['enemy'] = 0
        
        obs, _, terminated, _, _ = env.step(-1)
        dist = abs(env.player.x - env.enemy.x) + abs(env.player.y - env.enemy.y)
        
        proj = "Yes" if len(env.projectiles) > 0 else "No"
        print(f"Step {i+1}: Dist={dist}, EnemyPos={env.enemy.x, env.enemy.y}, Proj={proj}")
        
        if dist < 4:
            print(">> ALERT: Enemy got too close (D_MIN=4 violated)")
            
    # Test Retreating
    print("\n--- Testing Retreating (Force close proximity) ---")
    env.player.x, env.player.y = 5, 5
    env.enemy.x, env.enemy.y = 5, 6 # Dist 1 (Too close!)
    env.enemy.direction = "N" # Facing player
    
    for i in range(10):
        obs, _, _, _, _ = env.step(-1)
        dist = abs(env.player.x - env.enemy.x) + abs(env.player.y - env.enemy.y)
        print(f"Retreat Step {i+1}: Dist={dist}, EnemyPos={env.enemy.x, env.enemy.y}, Dir={env.enemy.direction}")
        if dist > 1:
            print(">> SUCCESS: Enemy is increasing distance.")
            break

if __name__ == "__main__":
    verify_tactical_behavior()
