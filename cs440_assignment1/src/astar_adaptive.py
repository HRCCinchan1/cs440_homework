import heapq

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_adaptive(start, goal, grid, heuristic):
    """
    Adaptive A* search.
    heuristic: dictionary storing learned h-values
    Returns: (path, expanded, g_values)
    """

    g = {start: 0}
    parent = {}
    open_list = []
    closed = set()
    expanded = 0
    counter = 0

    h_start = heuristic.get(start, manhattan(start, goal))
    heapq.heappush(open_list, (h_start, 0, counter, start))

    while open_list:
        _, _, _, current = heapq.heappop(open_list)

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        if current == goal:
            path = []
            s = goal
            while s != start:
                path.append(s)
                s = parent[s]
            path.append(start)
            path.reverse()
            return path, expanded, g

        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr][nc] == 0:
                    continue

                neighbor = (nr, nc)
                new_g = g[current] + 1

                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor] = new_g
                    parent[neighbor] = current
                    h = heuristic.get(neighbor, manhattan(neighbor, goal))
                    counter += 1
                    heapq.heappush(
                        open_list,
                        (new_g + h, -new_g, counter, neighbor)
                    )

    return None, expanded, g
