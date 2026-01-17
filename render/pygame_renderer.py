import pygame
from game.grid import Grid

# Global defaults only for non-instance usage if any (none here really)
ANIM_SPEED = 10.0
PANEL_WIDTH = 240
FONT_SIZE = 18
MAX_HEIGHT = 900  # Maximum window height

class PygameRenderer:
    def __init__(self, w, h):
        pygame.init()
        self.font = pygame.font.SysFont("consolas", FONT_SIZE)
        
        # Dynamic Cell Size Calculation
        # We want h * cell_size <= MAX_HEIGHT
        target_size = MAX_HEIGHT // h
        self.cell_size = min(32, target_size) # Cap at 32px
        self.cell_size = max(4, self.cell_size) # Min 4px to be visible
        
        self.panel_x = w * self.cell_size
        self.screen = pygame.display.set_mode(
            (self.panel_x + PANEL_WIDTH, max(h * self.cell_size, 600)) # Ensure at least 600 height for panel
        )
        pygame.display.set_caption(f"Tank Battle RL (Map: {w}x{h}, Cell: {self.cell_size}px)")
        self.clock = pygame.time.Clock()
        self.player_pos = None
        self.enemy_pos = None
        self.shots = []

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def get_action_from_keyboard(self):
        k = pygame.key.get_pressed()
        if k[pygame.K_a]: return 1
        if k[pygame.K_d]: return 2
        if k[pygame.K_w]: return 3
        if k[pygame.K_SPACE]: return 4
        return 0

    def update(self, dt, player, enemy):
        if self.player_pos is None:
            self.player_pos = [player.x * self.cell_size, player.y * self.cell_size]
            self.enemy_pos = [enemy.x * self.cell_size, enemy.y * self.cell_size]

        self._interp(self.player_pos, player, dt)
        self._interp(self.enemy_pos, enemy, dt)
        self._update_shots(dt)

    def _interp(self, pos, tank, dt):
        tx, ty = tank.x * self.cell_size, tank.y * self.cell_size
        pos[0] += (tx - pos[0]) * ANIM_SPEED * dt
        pos[1] += (ty - pos[1]) * ANIM_SPEED * dt

    def _update_shots(self, dt):
        for s in self.shots[:]:
            s["time"] -= dt
            if s["time"] <= 0:
                self.shots.remove(s)

    def draw(self, grid, player, enemy):
        self.screen.fill((0, 0, 0))

        for y in range(grid.height):
            for x in range(grid.width):
                c = (80, 80, 80) if grid.cells[y][x] else (30, 30, 30)
                pygame.draw.rect(
                    self.screen, c,
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )

        self._draw_tank(self.player_pos, player.direction, (0, 200, 0))
        self._draw_tank(self.enemy_pos, enemy.direction, (200, 0, 0))
        self._draw_hp(self.player_pos, player.hp)
        self._draw_hp(self.enemy_pos, enemy.hp)

    def _draw_tank(self, pos, d, color):
        cx, cy = pos[0] + self.cell_size//2, pos[1] + self.cell_size//2
        s = self.cell_size//2
        pts = {
            "N": [(cx, cy-s), (cx-s, cy+s), (cx+s, cy+s)],
            "S": [(cx, cy+s), (cx-s, cy-s), (cx+s, cy-s)],
            "E": [(cx+s, cy), (cx-s, cy-s), (cx-s, cy+s)],
            "W": [(cx-s, cy), (cx+s, cy-s), (cx+s, cy+s)]
        }[d]
        pygame.draw.polygon(self.screen, color, pts)

    def _draw_hp(self, pos, hp):
        w = self.cell_size
        h = max(2, self.cell_size // 6)
        x, y = pos[0], pos[1] - h - 2
        pygame.draw.rect(self.screen, (100, 0, 0), (x, y, w, h))
        pygame.draw.rect(self.screen, (0, 200, 0), (x, y, w * hp / 3, h))

    def draw_text(self, text, x, y, color=(220, 220, 220)):
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    def draw_projectiles(self, projectiles):
        for p in projectiles:
            px = p["x"] * self.cell_size + self.cell_size/2
            py = p["y"] * self.cell_size + self.cell_size/2
            r = max(2, self.cell_size // 8)
            pygame.draw.circle(self.screen, (255, 255, 0), (int(px), int(py)), r)

    def draw_panel(self, env):
        # Background for panel
        x = self.panel_x
        pygame.draw.rect(self.screen, (40, 40, 40), (x, 0, PANEL_WIDTH, self.screen.get_height()))

        x += 10
        y = 10

        self.draw_text("=== GAME INFO ===", x, y); y += 30
        self.draw_text(f"Time: {env.game_time()} s", x, y); y += 20
        self.draw_text(f"Steps: {env.steps}", x, y); y += 20
        
        self.draw_text("--- PLAYER ---", x, y); y += 25
        self.draw_text(f"HP: {env.player.hp}", x, y); y += 20
        # Cooldown viz
        cd = env.cooldowns['player']
        status = "READY" if cd == 0 else f"RELOAD {cd}"
        self.draw_text(f"Gun: {status}", x, y, (100, 255, 100) if cd==0 else (255, 100, 100)); y += 20
        
        self.draw_text(f"Shots: {env.stats['player']['shots']}", x, y); y += 20
        self.draw_text(f"Hits: {env.stats['player']['hits']}", x, y); y += 30

        self.draw_text("--- ENEMY ---", x, y); y += 25
        self.draw_text(f"HP: {env.enemy.hp}", x, y); y += 20
        self.draw_text(f"Shots: {env.stats['enemy']['shots']}", x, y); y += 20
        self.draw_text(f"Hits: {env.stats['enemy']['hits']}", x, y); y += 30

        self.draw_text("[R] Restart game", x, y + 10, (180, 180, 255))
        self.draw_text("[ESC] Menu", x, y + 30, (180, 180, 255))

    # --- UI Components ---
    
    def draw_menu(self, mode, size):
        self.screen.fill((30, 30, 40))
        cx, cy = self.screen.get_width() // 2, self.screen.get_height() // 2
        
        self.draw_text("TANK BATTLE RL", cx - 100, 50, (255, 255, 0))
        
        # Buttons are drawn by main loop, but we can helper text here
        self.draw_text(f"Current Mode: {mode}", cx - 100, 100)
        self.draw_text(f"Map Size: {size}x{size}", cx - 100, 130)

    def draw_editor(self, grid):
        self.screen.fill((0, 0, 0))
        for y in range(grid.height):
            for x in range(grid.width):
                c = (80, 80, 80) if grid.cells[y][x] else (30, 30, 30)
                pygame.draw.rect(
                    self.screen, c,
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )
        
        # Draw grid lines for clarity
        for x in range(grid.width):
            pygame.draw.line(self.screen, (50, 50, 50), (x*self.cell_size, 0), (x*self.cell_size, grid.height*self.cell_size))
        for y in range(grid.height):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y*self.cell_size), (grid.width*self.cell_size, y*self.cell_size))

        # Editor UI
        panel_x = grid.width * self.cell_size + 10
        self.draw_text("=== MAP EDITOR ===", panel_x, 20, (255, 200, 0))
        self.draw_text("LMB: Toggle Wall", panel_x, 60)
        self.draw_text("ENTER: Play", panel_x, 90)
        self.draw_text("S: Save Map", panel_x, 120)
        self.draw_text("L: Load Map", panel_x, 150)
        self.draw_text("C: Clear Map", panel_x, 180)
        self.draw_text("ESC: Menu", panel_x, 210)

class Button:
    def __init__(self, x, y, w, h, text, callback, color=(100, 100, 200)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = (min(color[0]+30, 255), min(color[1]+30, 255), min(color[2]+30, 255))
    
    def draw(self, screen, font):
        mouse_pos = pygame.mouse.get_pos()
        c = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(screen, c, self.rect)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        
        txt_surf = font.render(self.text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False