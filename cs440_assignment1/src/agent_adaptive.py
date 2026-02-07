import numpy as np
from astar_adaptive import astar_adaptive

BLOCKED = 0


class AgentAdaptive:
    def __init__(self, true_grid, start, goal):
        self.true_grid = true_grid
        self.start = start
        self.goal = goal
        self.pos = start

        rows, cols = true_grid.shape
        self.knowledge = np.ones((rows, cols), dtype=int)

        # Learned heuristic table
        self.heuristic = {}

    def observe(self):
        r, c = self.pos
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.true_grid.shape[0] and 0 <= nc < self.true_grid.shape[1]:
                self.knowledge[nr][nc] = self.true_grid[nr][nc]

    def run(self):
        total_expanded = 0

        while self.pos != self.goal:
            path, expanded, g_values = astar_adaptive(
                self.pos, self.goal, self.knowledge, self.heuristic
            )

            total_expanded += expanded

            if path is None:
                return False, total_expanded

            goal_g = g_values[self.goal]

            # Adaptive heuristic update
            for state in g_values:
                self.heuristic[state] = goal_g - g_values[state]

            for step in path[1:]:
                self.observe()

                if self.true_grid[step[0]][step[1]] == BLOCKED:
                    self.knowledge[step[0]][step[1]] = BLOCKED
                    break

                self.pos = step

                if self.pos == self.goal:
                    return True, total_expanded

        return True, total_expanded

