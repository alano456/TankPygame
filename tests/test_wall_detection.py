import math
from game.grid import Grid
from game.tank import Tank
from game.enemy import EnemyBot

def test_wall_detection():
    print("--- Testing Wall Detection ---")
    grid = Grid(5, 5)
    # Put walls at (2,1)[N], (2,3)[S], (3,2)[E], (1,2)[W]
    grid.cells[1][2] = Grid.WALL # North of (2,2)
    grid.cells[3][2] = Grid.WALL # South of (2,2)
    grid.cells[2][3] = Grid.WALL # East of (2,2)
    grid.cells[2][1] = Grid.WALL # West of (2,2)
    
    bot = EnemyBot()
    me = Tank(2, 2) # Center
    
    directions = ["N", "S", "E", "W"]
    results = []
    
    for d in directions:
        me.direction = d
        wall = bot.is_wall_ahead(me, grid)
        print(f"Direction {d}: Wall? {wall}")
        results.append(wall)
        
    if all(results):
        print(">> SUCCESS: Walls detected in all 4 directions centered at (2,2).")
    else:
        print(f">> FAILURE: Some walls missed. Results: {results}")

    # Test edge of map
    print("\n--- Testing Map Edge Detection ---")
    me.x, me.y = 0, 0
    me.direction = "N"
    wall_n = bot.is_wall_ahead(me, grid)
    print(f"Pos (0,0) Dir N (Edge): Wall? {wall_n}")
    
    me.direction = "W"
    wall_w = bot.is_wall_ahead(me, grid)
    print(f"Pos (0,0) Dir W (Edge): Wall? {wall_w}")
    
    if wall_n and wall_w:
        print(">> SUCCESS: Map edges detected as walls.")
    else:
        print(">> FAILURE: Map edges NOT detected.")

if __name__ == "__main__":
    test_wall_detection()
