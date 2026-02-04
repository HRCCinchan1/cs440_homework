import numpy as np
import matplotlib.pyplot as plt

grid = np.random.choice([0, 1], size=(10, 10))

plt.imshow(grid, cmap="gray")
plt.title("Test Grid")
plt.show()
