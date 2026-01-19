import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math

# ==================================================
# Load map (1 = free, 0 = obstacle)
# ==================================================
def load_image_to_grid(image_path, threshold=128):
    img = Image.open(image_path).convert('L')
    return (np.array(img) > threshold).astype(int)


# ==================================================
# Inflate obstacles (planning grid only)
# ==================================================
def inflate_obstacles(grid, radius=1):
    inflated = grid.copy()
    rows, cols = grid.shape

    for y in range(rows):
        for x in range(cols):
            if grid[y, x] == 0:
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            inflated[ny, nx] = 0
    return inflated


# ==================================================
# Node
# ==================================================
class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0.0


# ==================================================
# RRT*
# ==================================================
class RRTStar:
    def __init__(self, grid, start, goal,
                 max_iter=4000,
                 step_size=6,
                 goal_radius=8,
                 neighbor_radius=35):

        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = Node(start)
        self.goal = Node(goal)

        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.neighbor_radius = neighbor_radius

        self.nodes = [self.start]

    # ------------------------------------------------
    def sample_point(self):
        for _ in range(100):   # bounded sampling
            x = random.randint(0, self.cols - 1)
            y = random.randint(0, self.rows - 1)
            if self.grid[y, x] == 1:
                return (x, y)
        return None

    # ------------------------------------------------
    def nearest(self, point):
        return min(
            self.nodes,
            key=lambda n: np.linalg.norm(
                np.array(n.point) - np.array(point)
            )
        )

    # ------------------------------------------------
    def steer(self, from_node, to_point):
        from_pt = np.array(from_node.point, dtype=float)
        to_pt = np.array(to_point, dtype=float)

        dist = np.linalg.norm(to_pt - from_pt)
        if dist <= self.step_size:
            return tuple(to_pt.astype(int))

        direction = (to_pt - from_pt) / dist
        new_pt = from_pt + direction * self.step_size
        return tuple(new_pt.astype(int))

    # ------------------------------------------------
    # STRONG collision checking
    def collision_free(self, p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)

        dist = np.linalg.norm(p2 - p1)
        steps = max(int(dist / 0.4), 1)

        for i in range(steps + 1):
            pt = p1 + (p2 - p1) * i / steps
            x, y = int(round(pt[0])), int(round(pt[1]))

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (
                        nx < 0 or nx >= self.cols or
                        ny < 0 or ny >= self.rows or
                        self.grid[ny, nx] == 0
                    ):
                        return False
        return True

    # ------------------------------------------------
    def run(self):
        for _ in range(self.max_iter):

            # Goal bias (15%)
            if random.random() < 0.15:
                rnd = self.goal.point
            else:
                rnd = self.sample_point()

            if rnd is None:
                continue

            nearest = self.nearest(rnd)
            new_pt = self.steer(nearest, rnd)

            if not self.collision_free(nearest.point, new_pt):
                continue

            new_node = Node(new_pt)
            new_node.parent = nearest
            new_node.cost = nearest.cost + np.linalg.norm(
                np.array(nearest.point) - np.array(new_pt)
            )

            self.nodes.append(new_node)

            # Goal reached
            if (
                np.linalg.norm(
                    np.array(new_node.point) - np.array(self.goal.point)
                ) <= self.goal_radius
                and self.collision_free(new_node.point, self.goal.point)
            ):
                self.goal.parent = new_node
                return self.get_path()

        print("RRT* terminated: goal not reached within iteration limit")
        return None

    # ------------------------------------------------
    def get_path(self):
        path = []
        node = self.goal
        while node:
            path.append(node.point)
            node = node.parent
        return path[::-1]


# ==================================================
# Path distance
# ==================================================
def compute_path_distance(path):
    dist = 0.0
    for i in range(1, len(path)):
        dist += np.linalg.norm(
            np.array(path[i]) - np.array(path[i - 1])
        )
    return dist


# ==================================================
# GUI (original map only)
# ==================================================
class PointSelector:
    def __init__(self, grid_display, grid_planning):
        self.grid_display = grid_display
        self.grid_planning = grid_planning
        self.start = None
        self.goal = None

        img = np.zeros((*grid_display.shape, 3), dtype=np.uint8)
        img[grid_display == 1] = [255, 255, 255]
        img[grid_display == 0] = [0, 0, 0]

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(img)
        self.ax.set_title("Click START (green), then GOAL (red)")
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.grid_display[y, x] == 0:
            print("Obstacle selected")
            return

        if self.start is None:
            self.start = (x, y)
            self.ax.plot(x, y, 'go', markersize=8)

        elif self.goal is None:
            self.goal = (x, y)
            self.ax.plot(x, y, 'ro', markersize=8)
            self.fig.canvas.mpl_disconnect(self.cid)
            self.find_path()

        self.fig.canvas.draw()

    def find_path(self):
        rrt = RRTStar(self.grid_planning, self.start, self.goal)
        path = rrt.run()

        if path:
            px, py = zip(*path)
            self.ax.plot(px, py, 'm-', linewidth=2)
            plt.show()

            print("\n===== RRT* OUTPUT =====")
            print(f"Total nodes generated : {len(rrt.nodes)}")
            print(f"Path distance         : {compute_path_distance(path):.2f} pixels")
        else:
            print("No path found")


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    grid_original = load_image_to_grid("map2v2.png")
    grid_planning = inflate_obstacles(grid_original, radius=1)

    PointSelector(grid_original, grid_planning)
