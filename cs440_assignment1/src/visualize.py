import numpy as np
import matplotlib.pyplot as plt
from grid_loader import load_grid


def visualize_grid(grid):
    """
    Visualize the grid world.
    White = free cell
    Black = blocked cell
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray")
    plt.title("Grid World Visualization")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


if __name__ == "__main__":
    # Load one grid and visualize it
    grid = load_grid("maps/grid_0.npy")
    visualize_grid(grid)
