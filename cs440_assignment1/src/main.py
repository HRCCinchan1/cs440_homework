from grid_loader import load_grid
from agent import Agent
from agent_backward import AgentBackward
from agent_adaptive import AgentAdaptive
import os
import time


# Creating 50 grids 
NUM_GRIDS = 50
START = (0, 0)
GOAL = (100, 100)
os.makedirs("results", exist_ok=True)


# 2 & 3: Implementation of Repeated Forward vs Backward A*
print(f"Forward vs Backward A* on {NUM_GRIDS} grids")
print("*" * 60)

with open("results/forward_vs_backward.txt", "w") as f:
    f.write("grid,forward_expanded,backward_expanded\n")

    for i in range(NUM_GRIDS):
        print(f"\nGrid {i}/{NUM_GRIDS - 1}:")

        grid = load_grid(f"maps/grid_{i}.npy")

        # 2. Repeated Forward A* 
        print("  Running forward A* ", end=" ", flush=True)
        agent_fwd = Agent(grid, START, GOAL)
        start_time = time.time()
        _, expanded_fwd = agent_fwd.run()
        fwd_time = time.time() - start_time
        print(f"expanded {expanded_fwd} ({fwd_time:.2f}s)")

        # Repeated Backward A* 
        print("  Running backward A*", end=" ", flush=True)
        agent_bwd = AgentBackward(grid, START, GOAL)
        start_time = time.time()
        _, expanded_bwd = agent_bwd.run()
        bwd_time = time.time() - start_time
        print(f"expanded {expanded_bwd} ({bwd_time:.2f}s)")
        f.write(f"{i},{expanded_fwd},{expanded_bwd}\n")
        f.flush()

print("\n" + "*" * 60)
print("Forward vs Backward A* run complete.")


# 5. Implementation of ADAPTIVE A*
print("\nRunning Adaptive A* on all 50 grids...")
print("*" * 60)

with open("results/adaptive_astar.txt", "w") as f:
    f.write("grid,adaptive_expanded\n")

    for i in range(NUM_GRIDS):
        print(f"\nGrid {i}/{NUM_GRIDS - 1}:")

        grid = load_grid(f"maps/grid_{i}.npy")

        print("  Running adaptive A*", end=" ", flush=True)
        agent_adapt = AgentAdaptive(grid, START, GOAL)
        start_time = time.time()
        _, expanded_adapt = agent_adapt.run()
        adapt_time = time.time() - start_time
        print(f"expanded {expanded_adapt} ({adapt_time:.2f}s)")

        f.write(f"{i},{expanded_adapt}\n")
        f.flush()

print("\n" + "*" * 60)
print("Adaptive A* run complete.")

print("ALL EXPERIMENTS FINISHED SUCCESSFULLY.")
