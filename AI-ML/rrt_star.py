import random
import math
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Node Definition
# ==============================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


# ==============================
# 2️⃣ Environment 
# ==============================
start = (0, 0)
goal = (8, 8)

obstacles = [
    (5, 5, 1),
    (3, 6, 1),
    (6, 3, 1)
]

min_x, max_x = -2, 10
min_y, max_y = -2, 10

STEP_SIZE = 1.0
MAX_ITER = 500
NEAR_RADIUS = 2.0 * STEP_SIZE


# ==============================
# 3️⃣ Utility Functions
# ==============================
def distance(n1, n2):
    return math.hypot(n1.x - n2.x, n1.y - n2.y)


def get_random_node():
    return Node(
        random.uniform(min_x, max_x),
        random.uniform(min_y, max_y)
    )


def get_nearest_node(tree, rnd_node):
    return min(tree, key=lambda node: distance(node, rnd_node))


# ==============================
# 4️⃣ Steering
# ==============================
def steer(from_node, to_node):
    dist = distance(from_node, to_node)

    if dist <= STEP_SIZE:
        new_x, new_y = to_node.x, to_node.y
    else:
        angle = math.atan2(
            to_node.y - from_node.y,
            to_node.x - from_node.x
        )
        new_x = from_node.x + STEP_SIZE * math.cos(angle)
        new_y = from_node.y + STEP_SIZE * math.sin(angle)

    new_node = Node(new_x, new_y)
    new_node.parent = from_node
    new_node.cost = from_node.cost + distance(from_node, new_node)
    return new_node


# ==============================
# 5️⃣ Collision Checking
# ==============================
def is_edge_collision_free(from_node, to_node, step=0.1):
    dist = distance(from_node, to_node)
    steps = max(int(dist / step), 1)

    for i in range(steps + 1):
        t = i / steps
        x = from_node.x + t * (to_node.x - from_node.x)
        y = from_node.y + t * (to_node.y - from_node.y)

        for ox, oy, radius in obstacles:
            if math.hypot(x - ox, y - oy) <= radius:
                return False

    return True
  


# ==============================
# 6️⃣ RRT* Core Functions
# ==============================
def get_near_nodes(tree, new_node):
    return [
        node for node in tree
        if distance(node, new_node) <= NEAR_RADIUS
    ]


def choose_parent(near_nodes, new_node):
    best_parent = None
    min_cost = float("inf")

    for node in near_nodes:
        if is_edge_collision_free(node, new_node):
            temp_cost = node.cost + distance(node, new_node)
            if temp_cost < min_cost:
                best_parent = node
                min_cost = temp_cost

    if best_parent:
        new_node.parent = best_parent
        new_node.cost = min_cost

    return new_node


def rewire(near_nodes, new_node):
    for node in near_nodes:
        new_cost = new_node.cost + distance(new_node, node)
        if new_cost < node.cost and is_edge_collision_free(new_node, node):
            node.parent = new_node
            node.cost = new_cost


# ==============================
# 7️⃣ Path Extraction
# ==============================
def get_path(goal_node):
    path = []
    node = goal_node
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]


# ==============================
# 8️⃣ Main RRT* Algorithm
# ==============================
def main():
    tree = []
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    tree.append(start_node)

    for _ in range(MAX_ITER):
        rnd_node = get_random_node()
        nearest_node = get_nearest_node(tree, rnd_node)
        new_node = steer(nearest_node, rnd_node)

        if not is_edge_collision_free(nearest_node, new_node):
            continue

        near_nodes = get_near_nodes(tree, new_node)
        new_node = choose_parent(near_nodes, new_node)

        tree.append(new_node)
        rewire(near_nodes, new_node)

    # ==============================
    # 9️⃣ Best Goal Connection
    # ==============================
    goal_candidates = []

    for node in tree:
        if distance(node, goal_node) <= STEP_SIZE:
            if is_edge_collision_free(node, goal_node):
                temp_goal = Node(goal_node.x, goal_node.y)
                temp_goal.parent = node
                temp_goal.cost = node.cost + distance(node, temp_goal)
                goal_candidates.append(temp_goal)

    if not goal_candidates:
        print("No path found.")
        return

    goal_node = min(goal_candidates, key=lambda n: n.cost)
    path = get_path(goal_node)

    print(f"Optimal path cost: {goal_node.cost:.2f}")

    # ==============================
    # 🔟 Visualization
    # ==============================
    for node in tree:
        if node.parent:
            plt.plot(
                [node.x, node.parent.x],
                [node.y, node.parent.y],
                "-g",
                linewidth=0.5
            )

    for ox, oy, radius in obstacles:
        circle = plt.Circle((ox, oy), radius, color="r")
        plt.gca().add_patch(circle)

    px, py = zip(*path)
    plt.plot(px, py, "-b", linewidth=2)

    plt.plot(start[0], start[1], "ro")
    plt.plot(goal[0], goal[1], "bo")

    plt.axis("equal")
    plt.grid(True)
    plt.title("RRT* with Proper Collision Checking")
    plt.show()


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    main()
