import numpy as np
import random
import os

GRID_SIZE = 101
NUM_GRIDS = 50
BLOCK_PROB = 0.30  # 30% chance blocked, 70% unblocked


def generate_single_grid(size):
   
    grid = -1 * np.ones((size, size), dtype=int)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def in_bounds(r, c):
        return 0 <= r < size and 0 <= c < size

   
    start_r = random.randint(0, size - 1)
    start_c = random.randint(0, size - 1)

    stack = [(start_r, start_c)]
    grid[start_r, start_c] = 1  

    while stack:
        r, c = stack[-1]

        
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid[nr, nc] == -1:
                neighbors.append((nr, nc))

        if neighbors:
            nr, nc = random.choice(neighbors)

           
            if random.random() < BLOCK_PROB:
                grid[nr, nc] = 0  # blocked
            else:
                grid[nr, nc] = 1  # unblocked
                stack.append((nr, nc))
        else:
            
            stack.pop()

    
    grid[grid == -1] = 1

    return grid


def generate_all_grids():
    os.makedirs("maps", exist_ok=True)

    for i in range(NUM_GRIDS):
        grid = generate_single_grid(GRID_SIZE)
        filename = f"maps/grid_{i}.npy"
        np.save(filename, grid)
        print(f"Saved {filename}")


if __name__ == "__main__":
    generate_all_grids()
