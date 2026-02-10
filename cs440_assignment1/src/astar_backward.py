import heapq

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_backward(start, goal, grid):

    rows, cols = grid.shape

    g = {goal: 0}
    parent = {}

    open_heap = []
    closed = set()
    expanded = 0
    counter = 0

    h_goal = manhattan(goal, start)
    heapq.heappush(open_heap, (h_goal, 0, counter, goal))

    while open_heap:
        f, neg_g, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        expanded += 1

        if current == start:
            path = []
            s = start
            while s != goal:
                path.append(s)
                s = parent[s]
            path.append(goal)
            return path, expanded

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 0:
                    continue

                neighbor = (nr, nc)
                new_g = g[current] + 1

                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor] = new_g
                    parent[neighbor] = current
                    h = manhattan(neighbor, start)
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (new_g + h, -new_g, counter, neighbor)
                    )

    return None, expanded
