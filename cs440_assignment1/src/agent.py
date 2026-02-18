import numpy as np
from astar import astar

FREE = 1
BLOCKED = 0


class Agent:
    def __init__(self, true_grid, start, goal):
        self.true_grid = true_grid
        self.start = start
        self.goal = goal
        self.pos = start
        rows, cols = true_grid.shape
        self.knowledge = np.ones((rows, cols), dtype=int)

    def observe(self):
        r, c = self.pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.true_grid.shape[0] and 0 <= nc < self.true_grid.shape[1]:
                self.knowledge[nr, nc] = self.true_grid[nr, nc]

    def run(self):
        total_expanded = 0
        iterations = 0  

        while self.pos != self.goal:
            iterations += 1  
            
            if iterations > 10000:
                print(f"WARNING: Exceeded iteration limit. Expanded {total_expanded} cells.")
                return False, total_expanded
            
            path, expanded = astar(self.pos, self.goal, self.knowledge)
            total_expanded += expanded

            if path is None:
                print(f"Cannot reach the target. Expanded {total_expanded} cells.")
                return False, total_expanded

            for step in path[1:]:
                self.observe()

                if self.true_grid[step[0], step[1]] == BLOCKED:
                    self.knowledge[step[0], step[1]] = BLOCKED
                    break

                self.pos = step

                if self.pos == self.goal:
                    print(f"I reached the target. Expanded {total_expanded} cells.")
                    return True, total_expanded

        print(f"I reached the target. Expanded {total_expanded} cells.")
        return True, total_expanded