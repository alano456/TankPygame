import math
from env.tank_env import TankEnv
from game.grid import Grid

def test_sniper_bypass():
    print("--- Testing Sniper Obstacle Bypass ---")
    env = TankEnv(15, 15)
    env.reset()
    
    # Player at (5, 5)
    env.player.x, env.player.y = 5, 5
    
    # Enemy at (10, 5) - Directly aligned on X, but we'll put a wall in between
    env.enemy.x, env.enemy.y = 10, 5
    env.enemy.direction = "W"
    
    # Wall at (7, 5) and (7, 4), (7, 6) - A vertical bar
    env.grid.cells[5][7] = Grid.WALL
    env.grid.cells[4][7] = Grid.WALL
    env.grid.cells[6][7] = Grid.WALL
    
    shot_detected = False
    for i in range(200):
        env.cooldowns['enemy'] = 0
        obs, _, _, _, _ = env.step(-1)
        
        dist = math.hypot(env.player.x - env.enemy.x, env.player.y - env.enemy.y)
        is_aligned = abs(env.player.x - env.enemy.x) < 0.1 or abs(env.player.y - env.enemy.y) < 0.1
        has_los = env.enemy_bot.has_line_of_sight(env.enemy, env.player, env.grid)
        
        # Check for shot
        if len(env.projectiles) > 0:
            print(f"SUCCESS: Shot detected at step {i+1}!")
            shot_detected = True
            break
            
        if i % 5 == 0:
            print(f"Step {i}: E({env.enemy.x:.1f}, {env.enemy.y:.1f}), Aligned:{is_aligned}, LOS:{has_los}, Dist:{dist:.1f}")

    if not shot_detected:
        print("FAILURE: Sniper could not bypass wall to shoot.")

if __name__ == "__main__":
    test_sniper_bypass()
