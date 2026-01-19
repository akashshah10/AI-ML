import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from heapq import heappush, heappop
import math

# ==================================================
# Load map as binary grid (1 = free, 0 = obstacle)
# ==================================================
def load_image_to_grid(image_path, threshold=128):
    img = Image.open(image_path).convert('L')
    grid = (np.array(img) > threshold).astype(int)
    return grid


# ==================================================
# A* Algorithm (8-connected, corner-safe)
# ==================================================
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start))

    came_from = {}
    g_cost = {start: 0}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (1, 1), (-1, 1), (1, -1)
        ]:
            nx, ny = current[0] + dx, current[1] + dy

            if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx] == 1:

                # 🚫 Prevent diagonal corner cutting
                if dx != 0 and dy != 0:
                    if grid[current[1], current[0] + dx] == 0 or \
                       grid[current[1] + dy, current[0]] == 0:
                        continue

                neighbor = (nx, ny)
                new_cost = g_cost[current] + math.hypot(dx, dy)

                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    f = new_cost + heuristic(neighbor, goal)
                    heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

    return None


# ==================================================
# Interactive Point Selector
# ==================================================
class PointSelector:
    def __init__(self, grid):
        self.grid = grid
        self.start = None
        self.goal = None

        self.img_rgb = np.zeros((*grid.shape, 3), dtype=np.uint8)
        self.img_rgb[grid == 1] = [255, 255, 255]
        self.img_rgb[grid == 0] = [0, 0, 0]

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.img_rgb)
        self.ax.set_title("Click START (green), then GOAL (red)")
        self.cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.onclick
        )
        plt.show()

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.grid[y, x] == 0:
            print("Obstacle selected. Choose another point.")
            return

        if self.start is None:
            self.start = (x, y)
            self.ax.plot(x, y, 'go', markersize=8)
            print("Start:", self.start)

        elif self.goal is None:
            self.goal = (x, y)
            self.ax.plot(x, y, 'ro', markersize=8)
            print("Goal:", self.goal)

            self.fig.canvas.mpl_disconnect(self.cid)
            self.find_path()

        self.fig.canvas.draw()

    def find_path(self):
        print("Running A*...")
        path = a_star(self.grid, self.start, self.goal)

        if path:
            px, py = zip(*path)
            self.ax.plot(px, py, 'b-', linewidth=2, label="A* Path")
            self.ax.legend()
            self.ax.set_title("A* Path (Corner-Safe)")
            print(f"Path found! Number of points: {len(path)}")
            plt.show()
        else:
            print("No path found!")


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    grid = load_image_to_grid("map2v2.png")
    PointSelector(grid)
