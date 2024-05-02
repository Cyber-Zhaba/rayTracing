import numpy as np
from matplotlib import pyplot as plt


class Ray:
    def __init__(self, a: np.array, b: np.array):
        self.A = a
        self.B = b

    def origin(self):
        return self.A

    def direction(self):
        return self.B

    def point_at_parameter(self, t: float):
        return self.A + t * self.B


def color(ray: Ray):
    unit_direction = ray.direction() / np.linalg.norm(ray.direction())
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])


nx = 200
ny = 100

lower_left_corner = np.array([-2.0, -1.0, -1.0])
horizontal = np.array([4.0, 0.0, 0.0])
vertical = np.array([0.0, 2.0, 0.0])

origin = np.array([0.0, 0.0, 0.0])

image = np.zeros((ny, nx, 3))
for j in range(ny):
    for i in range(nx):
        u = i / nx
        v = (ny - j) / ny
        r = Ray(origin, lower_left_corner + u * horizontal + v * vertical)
        col = color(r)
        image[j, i] = col

plt.imsave("image.png", image)
