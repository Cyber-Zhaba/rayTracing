import numpy as np
from matplotlib import pyplot as plt

width = 200
height = 100
image = np.zeros((height, width, 3))
for i in range(height):
    for j in range(width):
        image[i][j] = [
            i / height,
            j / width,
            .2
        ]

plt.imsave("image.png", image)
