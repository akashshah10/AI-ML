import math

def astar(grid, start, goal):
    open_list = [start]       
    came_from = {}             
    g = {start: 0}             

    def heuristic(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    while open_list:
       
        current = min(open_list, key=lambda x: g[x] + heuristic(x, goal))

        if current == goal:
            
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        open_list.remove(current)
        x, y = current


        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0:
                new_g = g[current] + 1

                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    came_from[(nx, ny)] = current
                    if (nx, ny) not in open_list:
                        open_list.append((nx, ny))

    return None


grid = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0]
]

start = (0, 0)
goal = (2, 3)

path = astar(grid, start, goal)
print("Shortest Path:", path)
