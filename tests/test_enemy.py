from env.tank_env import TankEnv
from game.grid import Grid
import time

def test_enemy_aggression():
    print("--- Testing EnemyBot Aggression ---")
    env = TankEnv(12, 12)
    obs, _ = env.reset()
    
    # Force positions for alignment test
    # Player at (5, 5), Enemy at (5, 9) (South of Player, facing N ideally)
    env.player.x, env.player.y = 5, 5
    env.enemy.x, env.enemy.y = 5, 9
    
    # Force directions
    env.player.direction = "S"
    env.enemy.direction = "N" # Facing player
    
    print(f"Initial State: Player({env.player.x}, {env.player.y}), Enemy({env.enemy.x}, {env.enemy.y}, {env.enemy.direction})")
    
    # Run for 20 steps
    for i in range(20):
        # Player does nothing (action=10 -> invalid/idle if mapping handles it? Or just 0/1/2/3)
        # Action 0 is Left Turn. Action 4? 
        # Env step takes 0-3.
        # Check rulebased agent mapping in main.py: "if act == 0: return -1".
        # But step method doesn't handle -1.
        # "self._apply_action" checks 0,1,2,3. If -1, does nothing.
        # But logic: "if action == 2: ... elif action == 3: ...".
        # "self._apply_action" has if/elif. If unknown, does nothing.
        # So action=-1 is safe Idle.
        
        obs, reward, terminated, _, info = env.step(-1)
        
        # Check Projectiles
        proj_count = len(env.projectiles)
        
        # Check Enemy Action (Logic is inside step, hidden)
        # But we can see if projectile appeared.
        
        print(f"Step {i+1}: Enemy Pos({env.enemy.x}, {env.enemy.y}), Dir({env.enemy.direction}), Proj({proj_count}), Player HP({env.player.hp})")
        
        if proj_count > 0:
            print(">> Projectile detected!")
        
        if env.player.hp <= 0:
            print(">> Player DIED! Enemy works.")
            return
            
        if terminated:
            print(">> Terminated.")
            break
            
    print("--- Test Failed? Player survived 20 steps aligned. ---")

if __name__ == "__main__":
    test_enemy_aggression()
