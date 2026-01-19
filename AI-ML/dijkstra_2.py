import sys
from heapq import heapify, heappush, heappop
import matplotlib.pyplot as plt

def dijsktra(graph, src, dest):
    inf = sys.maxsize
    node_data = {'A':{'cost':inf,'pred':[]},
                 'B':{'cost':inf,'pred':[]},
                 'C':{'cost':inf,'pred':[]},
                 'D':{'cost':inf,'pred':[]},
                 'E':{'cost':inf,'pred':[]},
                 'F':{'cost':inf,'pred':[]}
                }
    node_data[src]['cost'] = 0
    visited = []
    temp = src
    for i in range(5):
        if temp not in visited:
            visited.append(temp)
            min_heap = []
            for j in graph[temp]:
                if j not in visited:
                    cost = node_data[temp]['cost'] + graph[temp][j]
                    if cost < node_data[j]['cost']:
                        node_data[j]['cost'] = cost 
                        node_data[j]['pred'] = node_data[temp]['pred'] + list(temp)
                    heappush(min_heap,(node_data[j]['cost'],j))
        heapify(min_heap)
        temp = min_heap[0][1]

    shortest_path = node_data[dest]['pred'] + list(dest)
    print("Shortest Distance: " + str(node_data[dest]['cost']))
    print("Shortest Path: " + str(shortest_path))
    
    plot_graph(graph, shortest_path)

def plot_graph(graph, shortest_path):
    # Coordinates approximated based on distances (weights)
    # You can adjust positions if you want better visual spacing
    positions = {
        'A': (0, 2),
        'B': (2, 4),
        'C': (2, 0),
        'D': (5, 4),
        'E': (5, 0),
        'F': (8, 2)
    }

    fig, ax = plt.subplots(figsize=(8,6))
    
    # Draw edges
    for node in graph:
        for neighbor, weight in graph[node].items():
            x_values = [positions[node][0], positions[neighbor][0]]
            y_values = [positions[node][1], positions[neighbor][1]]
            # Highlight shortest path
            if (node in shortest_path and neighbor in shortest_path and
                abs(shortest_path.index(node) - shortest_path.index(neighbor)) == 1):
                ax.plot(x_values, y_values, 'r', linewidth=3)
            else:
                ax.plot(x_values, y_values, 'k', linewidth=1)
            # Edge weights at midpoint
            mid_x = (positions[node][0] + positions[neighbor][0]) / 2
            mid_y = (positions[node][1] + positions[neighbor][1]) / 2
            ax.text(mid_x, mid_y, str(weight), color='blue', fontsize=10, fontweight='bold')

    # Draw nodes
    for node, (x, y) in positions.items():
        if node in shortest_path:
            ax.scatter(x, y, s=500, c='orange', zorder=5)
        else:
            ax.scatter(x, y, s=500, c='lightblue', zorder=5)
        ax.text(x, y, node, fontsize=12, ha='center', va='center', fontweight='bold')

    ax.set_title("Graph with Coordinates and Shortest Path Highlighted")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    graph = {
        'A': {'B':2,'C':4},
        'B': {'A':2,'C':3,'D':8},
        'C': {'A':4,'B':3,'E':5,'D':2},
        'D': {'B':8,'C':2,'E':11,'F':22},
        'E': {'C':5,'D':11,'F':1},
        'F': {'D':22,'E':1}
    }

    source = 'A'
    destination = 'F'
    dijsktra(graph, source, destination)
