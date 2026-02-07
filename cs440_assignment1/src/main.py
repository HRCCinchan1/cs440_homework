from grid_loader import load_grid
from agent import Agent
from agent_backward import AgentBackward
import os
import time  # ⭐ ADD THIS

NUM_GRIDS = 50
START = (0, 0)
GOAL = (100, 100)

os.makedirs("results", exist_ok=True)

print(f"Running Forward vs Backward on {NUM_GRIDS} grids...")  # ⭐ ADD THIS
print("=" * 60)  # ⭐ ADD THIS

with open("results/forward_vs_backward.txt", "w") as f:
    f.write("grid,forward_expanded,backward_expanded\n")

    for i in range(NUM_GRIDS):
        print(f"\nGrid {i}/{NUM_GRIDS}:")  # ⭐ CHANGED
        
        grid = load_grid(f"maps/grid_{i}.npy")

        # Forward
        print(f"  Running forward...", end=" ", flush=True)  # ⭐ CHANGED
        agent_fwd = Agent(grid, START, GOAL)
        start_time = time.time()  # ⭐ ADD THIS
        _, expanded_fwd = agent_fwd.run()
        fwd_time = time.time() - start_time  # ⭐ ADD THIS
        print(f"expanded {expanded_fwd} ({fwd_time:.2f}s)")  # ⭐ CHANGED

        # Backward
        print(f"  Running backward...", end=" ", flush=True)  # ⭐ CHANGED
        agent_bwd = AgentBackward(grid, START, GOAL)
        start_time = time.time()  # ⭐ ADD THIS
        _, expanded_bwd = agent_bwd.run()
        bwd_time = time.time() - start_time  # ⭐ ADD THIS
        print(f"expanded {expanded_bwd} ({bwd_time:.2f}s)")  # ⭐ CHANGED

        f.write(f"{i},{expanded_fwd},{expanded_bwd}\n")
        f.flush()

print("\n" + "=" * 60)  # ⭐ ADD THIS
print("Forward vs Backward experiment complete.")