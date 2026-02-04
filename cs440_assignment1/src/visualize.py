import numpy as np
import matplotlib.pyplot as plt

grid = np.load("maps/grid_0.npy")

plt.imshow(grid, cmap="gray")
plt.title("Gridworld")
plt.show()

