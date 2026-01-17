# game/grid.py

class Grid:
    EMPTY = 0
    WALL = 1

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[Grid.EMPTY for _ in range(width)] for _ in range(height)]
        self._create_walls()

    def _create_walls(self):
        for x in range(self.width):
            self.cells[0][x] = Grid.WALL
            self.cells[self.height - 1][x] = Grid.WALL

        for y in range(self.height):
            self.cells[y][0] = Grid.WALL
            self.cells[y][self.width - 1] = Grid.WALL

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x, y):
        return self.cells[y][x] == Grid.EMPTY

    def clear_line(self, x1, y1, x2, y2):
        # tylko linie proste (grid)
        if x1 == x2:
            step = 1 if y2 > y1 else -1
            for y in range(y1 + step, y2, step):
                if self.cells[y][x1] == Grid.WALL:
                    return False
            return True

        if y1 == y2:
            step = 1 if x2 > x1 else -1
            for x in range(x1 + step, x2, step):
                if self.cells[y1][x] == Grid.WALL:
                    return False
            return True

        return False
