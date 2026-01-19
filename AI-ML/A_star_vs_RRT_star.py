import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math
from heapq import heappush, heappop

# ==================================================
# Load map
# ==================================================
def load_image_to_grid(image_path, threshold=128):
    img = Image.open(image_path).convert('L')
    return (np.array(img) > threshold).astype(int)


# ==================================================
# Distance utility
# ==================================================
def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    return sum(
        np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        for i in range(1, len(path))
    )


# ==================================================
# A* (corner-safe)
# ==================================================
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0}

    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (1, 1), (-1, 1), (1, -1)
    ]

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy

            if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] == 1:

                # 🚫 prevent diagonal corner cutting
                if dx != 0 and dy != 0:
                    if grid[current[1], current[0] + dx] == 0 or \
                       grid[current[1] + dy, current[0]] == 0:
                        continue

                neighbor = (nx, ny)
                cost = g[current] + math.hypot(dx, dy)

                if neighbor not in g or cost < g[neighbor]:
                    g[neighbor] = cost
                    f = cost + heuristic(neighbor, goal)
                    heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

    return None


# ==================================================
# RRT*
# ==================================================
class Node:
    def __init__(self, p):
        self.p = p
        self.parent = None
        self.cost = 0.0


class RRTStar:
    def __init__(self, grid, start, goal,
                 max_iter=5000, step=6, goal_radius=8):

        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = Node(start)
        self.goal = Node(goal)
        self.nodes = [self.start]

        self.max_iter = max_iter
        self.step = step
        self.goal_radius = goal_radius

    def sample(self):
        for _ in range(100):
            x = random.randint(0, self.cols - 1)
            y = random.randint(0, self.rows - 1)
            if self.grid[y, x] == 1:
                return (x, y)
        return None

    def nearest(self, p):
        return min(self.nodes, key=lambda n: np.linalg.norm(np.array(n.p) - np.array(p)))

    def steer(self, n, p):
        d = np.linalg.norm(np.array(p) - np.array(n.p))
        if d <= self.step:
            return p
        u = (np.array(p) - np.array(n.p)) / d
        return tuple((np.array(n.p) + u * self.step).astype(int))

    def collision_free(self, a, b):
        a, b = np.array(a), np.array(b)
        steps = max(int(np.linalg.norm(b - a) / 0.5), 1)
        for i in range(steps + 1):
            p = a + (b - a) * i / steps
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < self.cols and 0 <= y < self.rows):
                return False
            if self.grid[y, x] == 0:
                return False
        return True

    def run(self):
        for _ in range(self.max_iter):
            rnd = self.goal.p if random.random() < 0.15 else self.sample()
            if rnd is None:
                continue

            near = self.nearest(rnd)
            new_p = self.steer(near, rnd)

            if not self.collision_free(near.p, new_p):
                continue

            n = Node(new_p)
            n.parent = near
            n.cost = near.cost + np.linalg.norm(np.array(near.p) - np.array(new_p))
            self.nodes.append(n)

            if np.linalg.norm(np.array(n.p) - np.array(self.goal.p)) <= self.goal_radius:
                if self.collision_free(n.p, self.goal.p):
                    self.goal.parent = n
                    return self.path()

        return None

    def path(self):
        p, n = [], self.goal
        while n:
            p.append(n.p)
            n = n.parent
        return p[::-1]


# ==================================================
# GUI
# ==================================================
class PlannerGUI:
    def __init__(self, grid):
        self.grid = grid
        self.start = None
        self.goal = None

        img = np.zeros((*grid.shape, 3), dtype=np.uint8)
        img[grid == 1] = [255, 255, 255]
        img[grid == 0] = [0, 0, 0]

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(img)
        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.click)
        plt.show()

    def click(self, e):
        if e.xdata is None or e.ydata is None:
            return
        x, y = int(e.xdata), int(e.ydata)
        if self.grid[y, x] == 0:
            return

        if self.start is None:
            self.start = (x, y)
            self.ax.plot(x, y, "go")

        elif self.goal is None:
            self.goal = (x, y)
            self.ax.plot(x, y, "ro")
            self.fig.canvas.mpl_disconnect(self.cid)
            self.run()

        self.fig.canvas.draw()

    def run(self):
        p_a = a_star(self.grid, self.start, self.goal)
        p_r = RRTStar(self.grid, self.start, self.goal).run()

        if p_a:
            self.ax.plot(*zip(*p_a), "b--", label=f"A* {path_length(p_a):.2f}")

        if p_r:
            self.ax.plot(*zip(*p_r), "m-", label=f"RRT* {path_length(p_r):.2f}")

        self.ax.legend()
        plt.show()


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    grid = load_image_to_grid("map2v2.png")
    PlannerGUI(grid)
