import matplotlib.pyplot as plt
from grid_loader import load_grid
import os


def save_grid_as_png(grid, grid_id):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray")
    plt.title(f"Grid World {grid_id}")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    os.makedirs("visualizations", exist_ok=True)
    filename = f"visualizations/grid_{grid_id}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()   


if __name__ == "__main__":
    NUM_GRIDS = 50

    for i in range(NUM_GRIDS):
        grid = load_grid(f"maps/grid_{i}.npy")
        save_grid_as_png(grid, i)

    print("All 50 grid worlds saved as PNG files.")
