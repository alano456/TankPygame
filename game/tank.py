DIRECTIONS = ["N", "E", "S", "W"]

DIR_VECTORS = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0)
}


class Tank:
    def __init__(self, x, y, direction="N", hp=1):
        self.x = x
        self.y = y
        self.direction = direction
        self.hp = hp

    @property
    def alive(self):
        return self.hp > 0

    def turn_left(self):
        self.direction = DIRECTIONS[(DIRECTIONS.index(self.direction) - 1) % 4]

    def turn_right(self):
        self.direction = DIRECTIONS[(DIRECTIONS.index(self.direction) + 1) % 4]

    def move_forward(self, grid):
        dx, dy = DIR_VECTORS[self.direction]
        nx, ny = self.x + dx, self.y + dy
        if grid.in_bounds(nx, ny) and grid.is_free(nx, ny):
            self.x = nx
            self.y = ny
