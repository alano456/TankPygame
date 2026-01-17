import random
import math
from game.grid import Grid

class EnemyBot:
    def choose_action(self, me, target, grid, cooldown=0):
        """
        Strategia: Approaching Sniper (Base Agent - V3)
        1. Shot Check: Aligned + LOS + Cooldown -> Shoot.
        2. Goal Selection: Target distance 5 and axial alignment.
        3. Smart Move: Evaluate all 4 directions, pick best free one.
        """
        dx = target.x - me.x
        dy = target.y - me.y
        dist = math.hypot(dx, dy)
        has_los = self.has_line_of_sight(me, target, grid)
        
        # --- 1. SHOT CHECK ---
        is_aligned_x = abs(dx) < 0.1
        is_aligned_y = abs(dy) < 0.1
        
        if is_aligned_x or is_aligned_y:
            target_deg = (90 if dy > 0 else -90) if is_aligned_x else (0 if dx > 0 else 180)
            my_deg = {"N": -90, "S": 90, "E": 0, "W": 180}.get(me.direction, 0)
            diff = (target_deg - my_deg + 180) % 360 - 180
            
            if abs(diff) < 5:
                if has_los and cooldown == 0:
                    return 4 # Shoot
            else:
                return 2 if diff > 0 else 1 # Align rotation

        # --- 2. GOAL SELECTION ---
        # Determinuje w którą stronę bot CHCIAŁBY iść
        if dist < 4 and has_los:
            # Ucieczka
            goal_deg = math.degrees(math.atan2(-dy, -dx))
        elif dist > 6:
            # Zbliżanie
            goal_deg = math.degrees(math.atan2(dy, dx))
        else:
            # Próba wyrównania (align)
            if not has_los:
                # Jeśli brak LOS, musimy zejść z linii (jeśli na niej jesteśmy) lub dążyć do celu okrężnie
                # Dodajemy 90 stopni do wektora do celu żeby "obchodzić" przeszkodę
                goal_deg = math.degrees(math.atan2(dy, dx)) + 90
            else:
                # Wyrównaj mniejszą oś
                if abs(dx) < abs(dy): goal_deg = 0 if dx > 0 else 180
                else: goal_deg = 90 if dy > 0 else -90

        # --- 3. SMART MOVE ---
        return self.navigate_to_deg(me, goal_deg, grid)

    def navigate_to_deg(self, me, target_deg, grid):
        """Wybiera najlepszą DOPUSZCZALNĄ akcję by osiągnąć target_deg"""
        possible_degs = [-90, 0, 90, 180]
        # Sortuj kierunki od najlepszego (najbliższego target_deg)
        candidates = sorted(possible_degs, key=lambda d: abs((target_deg - d + 180) % 360 - 180))
        
        my_deg = {"N": -90, "S": 90, "E": 0, "W": 180}.get(me.direction, 0)
        
        for cand_deg in candidates:
            if self.is_direction_free(me, cand_deg, grid):
                if cand_deg == my_deg:
                    return 3 # Forward
                diff = (cand_deg - my_deg + 180) % 360 - 180
                return 2 if diff > 0 else 1
                
        # Jeśli zablokowany całkowicie, obróć się
        return random.choice([1, 2])

    def is_direction_free(self, me, deg, grid):
        dx, dy = 0, 0
        if deg == -90: dy = -1
        elif deg == 90: dy = 1
        elif deg == 0: dx = 1
        elif deg == 180: dx = -1
        nx, ny = int(me.x + dx), int(me.y + dy)
        if not (0 <= nx < grid.width and 0 <= ny < grid.height): return False
        return grid.cells[ny][nx] == Grid.EMPTY

    def is_wall_ahead(self, me, grid):
        my_deg = {"N": -90, "S": 90, "E": 0, "W": 180}.get(me.direction, 0)
        return not self.is_direction_free(me, my_deg, grid)

    def has_line_of_sight(self, me, target, grid):
        x0, y0 = int(me.x), int(me.y)
        x1, y1 = int(target.x), int(target.y)
        points = self.get_line(x0, y0, x1, y1)
        for px, py in points:
            if px == x0 and py == y0: continue
            if px == x1 and py == y1: continue
            if grid.cells[py][px] == Grid.WALL: return False
        return True

    def get_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0: y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0: x += sx; err += dy
                y += sy
        points.append((x, y))
        return points

# --- SEKCA TESTOWA (Self-Test) ---
if __name__ == "__main__":
    import math
    from env.tank_env import TankEnv
    from game.tank import Tank
    
    def run_all_tests():
        print("=== Baza Agent: Uruchamianie Testów Samosprawdzających ===")
        
        # 1. Test Wykrywania Ścian
        print("\n[TEST 1] Wykrywanie ścian i krawędzi...")
        grid = Grid(5, 5)
        # Ściany wokół (2,2)
        grid.cells[1][2] = Grid.WALL; grid.cells[3][2] = Grid.WALL
        grid.cells[2][1] = Grid.WALL; grid.cells[2][3] = Grid.WALL
        bot = EnemyBot()
        me = Tank(2, 2)
        for d in ["N", "S", "E", "W"]:
            me.direction = d
            if not bot.is_wall_ahead(me, grid):
                print(f"FAILED: Nie wykryto ściany na {d}")
        print("OK: Ściany i krawędzie wykrywane poprawnie.")

        # 2. Test Celowania (Alignment)
        print("\n[TEST 2] Sniper Alignment (Podejście i wyrównanie)...")
        env = TankEnv(15, 15)
        env.reset()
        env.player.x, env.player.y = 5, 5
        env.enemy.x, env.enemy.y = 12, 12
        env.enemy.direction = "W"
        
        steps = 0
        aligned = False
        shot = False
        for i in range(100):
            env.cooldowns['enemy'] = 0
            env.step(-1)
            dist = math.hypot(env.player.x - env.enemy.x, env.player.y - env.enemy.y)
            if not aligned and (abs(env.player.x - env.enemy.x) < 0.1 or abs(env.player.y - env.enemy.y) < 0.1):
                aligned = True
                print(f"Step {i}: Wyrównano oś! Dist: {dist:.1f}")
            if len(env.projectiles) > 0:
                shot = True
                print(f"Step {i}: ODDANO STRZAŁ!")
                break
        if shot: print("OK: Sniper namierzył i wystrzelił.")
        else: print("FAILED: Sniper nie oddał strzału.")

        # 3. Test Omijania (Bypass)
        print("\n[TEST 3] Sniper Bypass (Omijanie przeszkody)...")
        env.reset()
        env.player.x, env.player.y = 5, 5
        env.enemy.x, env.enemy.y = 10, 5
        env.enemy.direction = "W"
        # Mur na drodze
        env.grid.cells[5][7] = Grid.WALL; env.grid.cells[4][7] = Grid.WALL; env.grid.cells[6][7] = Grid.WALL
        
        success = False
        for i in range(100):
            env.cooldowns['enemy'] = 0
            env.step(-1)
            if len(env.projectiles) > 0:
                success = True
                print(f"Step {i}: Omijanie udane, oddano strzał!")
                break
        if success: print("OK: Przeszkoda ominięta.")
        else: print("FAILED: Bot utknął na ścianie.")

        print("\n=== Wszystkie testy zakończne ===")

    run_all_tests()
