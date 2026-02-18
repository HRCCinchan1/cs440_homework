import os
import time
import numpy as np
from collections import deque
from grid_loader import load_grid
from agent import Agent
from agent_small_g import AgentSmallG
from agent_backward import AgentBackward
from agent_adaptive import AgentAdaptive


NUM_GRIDS = 50
os.makedirs("results", exist_ok=True)


def get_reachable_region(grid, start):
    """BFS from start — returns all cells reachable from start."""
    visited = []
    seen = set()
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        visited.append(cur)
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if grid[nr,nc] == 1 and (nr,nc) not in seen:
                    queue.append((nr,nc))
    return visited


def get_start_goal(grid):
    """Pick start and goal guaranteed to be in the same connected component."""
    unblocked = list(zip(*np.where(grid == 1)))
    if len(unblocked) < 2:
        return None, None
    start = unblocked[0]
    reachable = get_reachable_region(grid, start)
    if len(reachable) < 2:
        return None, None
    return start, reachable[-1]

def run_agent(agent):
    start_time = time.time()
    success, expanded = agent.run()
    return success, expanded, time.time() - start_time

print("PART 2: Tie-breaking (Large-g vs Small-g)")
print("=" * 60)

with open("results/tiebreaking.txt", "w") as f:
    f.write("grid,large_g_expanded,small_g_expanded\n")

    for i in range(NUM_GRIDS):
        grid = load_grid(f"maps/grid_{i}.npy")
        start, goal = get_start_goal(grid)

        if start is None:
            print(f"  Grid {i+1}/{NUM_GRIDS}: skipped (not enough unblocked cells)")
            f.write(f"{i},0,0\n")
            continue

        print(f"\n  Grid {i+1}/{NUM_GRIDS}:")

        _, exp_large, t_large = run_agent(Agent(grid, start, goal))
        print(f"    Large-g: {exp_large} expanded ({t_large:.2f}s)")

        _, exp_small, t_small = run_agent(AgentSmallG(grid, start, goal))
        print(f"    Small-g: {exp_small} expanded ({t_small:.2f}s)")

        f.write(f"{i},{exp_large},{exp_small}\n")
        f.flush()

print("\nPart 2 complete.")


print("\nPART 3: Repeated Forward A* vs Repeated Backward A*")
print("=" * 60)

with open("results/forward_vs_backward.txt", "w") as f:
    f.write("grid,forward_expanded,backward_expanded\n")

    for i in range(NUM_GRIDS):
        grid = load_grid(f"maps/grid_{i}.npy")
        start, goal = get_start_goal(grid)

        if start is None:
            print(f"  Grid {i+1}/{NUM_GRIDS}: skipped (not enough unblocked cells)")
            f.write(f"{i},0,0\n")
            continue

        print(f"\n  Grid {i+1}/{NUM_GRIDS}:")

        _, exp_fwd, t_fwd = run_agent(Agent(grid, start, goal))
        print(f"    Forward:  {exp_fwd} expanded ({t_fwd:.2f}s)")

        _, exp_bwd, t_bwd = run_agent(AgentBackward(grid, start, goal))
        print(f"    Backward: {exp_bwd} expanded ({t_bwd:.2f}s)")

        f.write(f"{i},{exp_fwd},{exp_bwd}\n")
        f.flush()

print("\nPart 3 complete.")

print("\nPART 5: Repeated Forward A* vs Adaptive A*")
print("=" * 60)

with open("results/adaptive_astar.txt", "w") as f:
    f.write("grid,forward_expanded,adaptive_expanded\n")

    for i in range(NUM_GRIDS):
        grid = load_grid(f"maps/grid_{i}.npy")
        start, goal = get_start_goal(grid)

        if start is None:
            print(f"  Grid {i+1}/{NUM_GRIDS}: skipped (not enough unblocked cells)")
            f.write(f"{i},0,0\n")
            continue

        print(f"\n  Grid {i+1}/{NUM_GRIDS}:")

        _, exp_fwd, t_fwd = run_agent(Agent(grid, start, goal))
        print(f"    Forward:  {exp_fwd} expanded ({t_fwd:.2f}s)")

        _, exp_adapt, t_adapt = run_agent(AgentAdaptive(grid, start, goal))
        print(f"    Adaptive: {exp_adapt} expanded ({t_adapt:.2f}s)")

        f.write(f"{i},{exp_fwd},{exp_adapt}\n")
        f.flush()

print("\nPart 5 complete.")

print("\n" + "=" * 60)
print("ALL EXPERIMENTS FINISHED SUCCESSFULLY.")
print("Results saved to results/tiebreaking.txt, forward_vs_backward.txt, adaptive_astar.txt")