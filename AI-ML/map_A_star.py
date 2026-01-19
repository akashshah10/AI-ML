import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from heapq import heappush, heappop

# Load binary map image
def load_image_to_grid(image_path, threshold=128):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    grid = (img_array > threshold).astype(int)
    return grid

# A* algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start))
    
    came_from = {}
    cost_so_far = {start: 0}
    
    while open_set:
        _, current_cost, current = heappop(open_set)
        
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for dr, dc in [(-1,1),(-1,-1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]:  # 4 directions
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                neighbor = (nr, nc)
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
    return None

# Interactive click handler
class PointSelector:
    def __init__(self, grid):
        self.grid = grid
        self.start = None
        self.end = None
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(grid, cmap='gray_r')  # obstacles=black, free=white
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.path = None
        self.done = False
        plt.title("Click to select START (first) and END (second) points")
        plt.show()
    
    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        r, c = int(event.ydata), int(event.xdata)
        if self.grid[r, c] == 0:
            print(f"Selected point ({r},{c}) is an obstacle! Choose another.")
            return
        if self.start is None:
            self.start = (r, c)
            self.ax.plot(c, r, 'go')  # green = start
            self.fig.canvas.draw()
            print(f"Start selected at {self.start}")
        elif self.end is None:
            self.end = (r, c)
            self.ax.plot(c, r, 'ro')  # red = end
            self.fig.canvas.draw()
            print(f"End selected at {self.end}")
            self.fig.canvas.mpl_disconnect(self.cid)
            print("Finding path with A*...")
            self.find_path()
    
    def find_path(self):
        self.path = a_star(self.grid, self.start, self.end)
        if self.path:
            print(f"Path found! Length: {len(self.path)}")
            path_rows, path_cols = zip(*self.path)
            self.ax.plot(path_cols, path_rows, 'b.', markersize=1)  # blue path
            plt.title("Path found (yellow)), Start=green, End=red")
            plt.show()
        else:
            print("No path found!")

if __name__ == "__main__":
    image_path = "map2v2.png"
    grid = load_image_to_grid(image_path)
    selector = PointSelector(grid)
