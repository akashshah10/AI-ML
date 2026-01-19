import random
import math
import matplotlib.pyplot as plt

# 1️⃣ Define a Node (a point in the tree)
class Node:
    def __init__(self, x, y):
        self.x = x          # x-coordinate of this point
        self.y = y          # y-coordinate of this point
        self.parent = None  # previous point in the tree (used to find the path)

# 2️⃣ Define the environment
start = (0, 0)               # starting point
goal = (8, 8)                # goal point
obstacles = [(5, 5, 1),      # list of obstacles (x, y, radius)
             (3, 6, 1),
             (6, 3, 1)]
min_x, max_x = -2, 10        # x-boundaries of the map
min_y, max_y = -2, 10        # y-boundaries of the map
STEP_SIZE = 1.0               # maximum step the tree can grow at once
MAX_ITER = 500                # maximum number of iterations

# 3️⃣ Distance function
def distance(n1, n2):
    # Euclidean distance between two nodes
    return math.hypot(n1.x - n2.x, n1.y - n2.y)

# 4️⃣ Random node generator
def get_random_node():
    # Pick a random point within boundaries
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    return Node(x, y)

# 5️⃣ Find nearest node in the tree
def get_nearest_node(tree, rnd_node):
    nearest = tree[0]  # start assuming the first node is nearest
    for node in tree:
        if distance(node, rnd_node) < distance(nearest, rnd_node):
            nearest = node  # update if a closer node is found
    return nearest

# 6️⃣ Steering function (grow tree step by step)
def steer(from_node, to_node):
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    angle = math.atan2(dy, dx)  # direction from from_node to to_node

    new_x = from_node.x + STEP_SIZE * math.cos(angle)
    new_y = from_node.y + STEP_SIZE * math.sin(angle)

    new_node = Node(new_x, new_y)  
    new_node.parent = from_node      # connect to the tree
    return new_node

# 7️⃣ Collision check
def is_collision_free(node):
    for ox, oy, radius in obstacles:
        if math.hypot(node.x - ox, node.y - oy) <= radius:
            return False   # node is inside an obstacle
    return True            # node is safe

# 8️⃣ Extract final path
def get_path(goal_node):
    path = []
    node = goal_node
    while node is not None:     # follow parent links back to start
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]           # reverse path to start->goal

# 9️⃣ Main RRT loop
def main():
    tree = []
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    tree.append(start_node)    # initialize tree with start

    for i in range(MAX_ITER):
        rnd_node = get_random_node()                  # pick a random point
        nearest_node = get_nearest_node(tree, rnd_node)  # find closest tree node
        new_node = steer(nearest_node, rnd_node)         # grow tree step toward random node

        if is_collision_free(new_node):                  # only add if safe
            tree.append(new_node)                        # expand tree

            if distance(new_node, goal_node) < STEP_SIZE:  # check if goal reached
                goal_node.parent = new_node
                tree.append(goal_node)
                print("Goal reached!")
                break

    # 10️⃣ Draw the tree and path
    for node in tree:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")  # green lines = tree

    for ox, oy, radius in obstacles:
        circle = plt.Circle((ox, oy), radius, color="r")  # red circles = obstacles
        plt.gca().add_patch(circle)

    path = get_path(goal_node)
    px, py = zip(*path)
    plt.plot(px, py, "-b", linewidth=2)  # blue line = final path

    plt.plot(start[0], start[1], "ro")  # start point
    plt.plot(goal[0], goal[1], "bo")    # goal point
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
