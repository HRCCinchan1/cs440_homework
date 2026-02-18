import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque
from grid_loader import load_grid
import os


def save_grid_as_png(grid, grid_id):
    """Save a plain grid world as PNG (no path)."""
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray")
    plt.title(f"Grid World {grid_id}")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    os.makedirs("visualizations", exist_ok=True)
    filename = f"visualizations/grid_{grid_id}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def visualize_grid(grid, start=None, goal=None, path=None, title="Grid World", save_path=None):
    """
    Visualize a grid with optional start, goal, and path overlay.

    Parameters:
        grid      : 2D numpy array (1=unblocked, 0=blocked)
        start     : (row, col) tuple for start cell
        goal      : (row, col) tuple for goal cell
        path      : list of (row, col) tuples representing the path
        title     : plot title string
        save_path : if provided, save to this filepath instead of showing
    """
    rows, cols = grid.shape
    img = np.zeros((rows, cols, 3))

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                img[r, c] = [1, 1, 1]   # white = unblocked
            else:
                img[r, c] = [0, 0, 0]   # black = blocked

    # Draw path in light blue
    if path:
        for (r, c) in path:
            img[r, c] = [0.4, 0.7, 1.0]

    # Draw start in green (on top of path)
    if start:
        img[start[0], start[1]] = [0.0, 0.8, 0.0]

    # Draw goal in red (on top of path)
    if goal:
        img[goal[0], goal[1]] = [0.9, 0.1, 0.1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    legend_items = [
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Unblocked"),
        mpatches.Patch(facecolor="black", label="Blocked"),
        mpatches.Patch(facecolor=(0.4, 0.7, 1.0), label="Path / Reachable region"),
        mpatches.Patch(facecolor=(0.0, 0.8, 0.0), label="Start"),
        mpatches.Patch(facecolor=(0.9, 0.1, 0.1), label="Goal"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


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
    """
    Pick start and goal guaranteed to be in the same connected component
    by doing a BFS from the first unblocked cell.
    """
    unblocked = list(zip(*np.where(grid == 1)))
    if len(unblocked) < 2:
        return None, None

    start = unblocked[0]
    reachable = get_reachable_region(grid, start)

    if len(reachable) < 2:
        return None, None

    # Pick the last reachable cell as goal — guaranteed connected
    goal = reachable[-1]
    return start, goal


if __name__ == "__main__":
    from astar import astar

    NUM_GRIDS = 50
    os.makedirs("visualizations", exist_ok=True)

    for i in range(NUM_GRIDS):
        grid = load_grid(f"maps/grid_{i}.npy")
        start, goal = get_start_goal(grid)

        if start is None:
            save_grid_as_png(grid, i)
            print(f"Grid {i}: not enough unblocked cells, saved plain grid.")
            continue

        # Try to find a path with full knowledge of the grid
        path, _ = astar(start, goal, grid)

        if path is not None:
            # Solvable — show the actual shortest path
            visualize_grid(
                grid,
                start=start,
                goal=goal,
                path=path,
                title=f"Grid {i} — Solved (path length {len(path)})",
                save_path=f"visualizations/grid_{i}.png"
            )
            print(f"Grid {i}: solved, path length {len(path)}.")
        else:
            # Unsolvable — flood fill the reachable region from start
            reachable = get_reachable_region(grid, start)
            visualize_grid(
                grid,
                start=start,
                goal=goal,
                path=reachable,
                title=f"Grid {i} — Unsolvable (reachable region shown)",
                save_path=f"visualizations/grid_{i}.png"
            )
            print(f"Grid {i}: unsolvable, reachable region shown.")

    print("Done. All grids saved to visualizations/")