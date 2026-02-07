from grid_loader import load_grid
from agent import Agent
from agent_backward import AgentBackward
import os

NUM_GRIDS = 50
START = (0, 0)
GOAL = (100, 100)

os.makedirs("results", exist_ok=True)

with open("results/forward_vs_backward.txt", "w") as f:
    f.write("grid,forward_expanded,backward_expanded\n")

    for i in range(NUM_GRIDS):
        grid = load_grid(f"maps/grid_{i}.npy")

        agent_fwd = Agent(grid, START, GOAL)
        _, expanded_fwd = agent_fwd.run()

        print(f"Grid {i} forward done. Expanded {expanded_fwd}", flush=True)

        print(f"Starting grid {i} (backward)...", flush=True)

        agent_bwd = AgentBackward(grid, START, GOAL)
        _, expanded_bwd = agent_bwd.run()

        print(f"Grid {i} backward done. Expanded {expanded_bwd}", flush=True)

        f.write(f"{i},{expanded_fwd},{expanded_bwd}\n")
        f.flush()
        print(f"Grid {i}: forward={expanded_fwd}, backward={expanded_bwd}")

print("Forward vs Backward experiment complete.")




