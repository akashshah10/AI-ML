from PIL import Image
import numpy as np
from heapq import heappush, heappop

def load_image_to_grid(image_path, threshold=128):
    img = Image.open(image_path).convert('L')  
    img_array = np.array(img)
    grid = (img_array > threshold).astype(int)  
    return grid
def grid_to_graph(grid):
    rows, cols = grid.shape
    graph = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                node = (r, c)
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        neighbors.append((nr, nc))
                graph[node] = {n:1 for n in neighbors}  
    return graph

def dijkstra(graph, src, dest):
    inf = float('inf')
    node_data = {node: {'cost': inf, 'pred': []} for node in graph}
    node_data[src]['cost'] = 0
    visited = set()
    min_heap = [(0, src)]
    
    while min_heap:
        cost, node = heappop(min_heap)
        if node in visited:
            continue
        visited.add(node)
        if node == dest:
            break
        for neighbor in graph[node]:
            if neighbor not in visited:
                new_cost = cost + graph[node][neighbor]
                if new_cost < node_data[neighbor]['cost']:
                    node_data[neighbor]['cost'] = new_cost
                    node_data[neighbor]['pred'] = node_data[node]['pred'] + [node]
                    heappush(min_heap, (new_cost, neighbor))
                    
    if node_data[dest]['cost'] == inf:
        print("No path found.")
    else:
        print("Shortest Distance:", node_data[dest]['cost'])
        print("Shortest Path:", node_data[dest]['pred'] + [dest])

image_path = "map2v2.png"
grid = load_image_to_grid(image_path)
graph = grid_to_graph(grid)

source = (start_row, start_col)  
destination = (end_row, end_col)  

dijkstra(graph, source, destination)
