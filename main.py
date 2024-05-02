from matplotlib import pyplot as plt
from tqdm import trange

from hittable import *


def color(ray: Ray, world: Hittable):
    rec = HitRecord(0, np.array([0, 0, 0]), np.array([0, 0, 0]))
    if world.hit(ray, 0, np.inf, rec):
        return 0.5 * np.array([rec.normal[0] + 1, rec.normal[1] + 1, rec.normal[2] + 1])
    else:
        unit_direction = ray.direction() / np.linalg.norm(ray.direction())
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])


def render():
    nx = 200
    ny = 100

    lower_left_corner = np.array([-2.0, -1.0, -1.0])
    horizontal = np.array([4.0, 0.0, 0.0])
    vertical = np.array([0.0, 2.0, 0.0])

    origin = np.array([0.0, 0.0, 0.0])
    image = np.zeros((ny, nx, 3))

    world = Hittable([Sphere(np.array([0, 0, -1]), 0.5), Sphere(np.array([0, -100.5, -1]), 100)])

    for j in trange(ny):
        for i in range(nx):
            u = i / nx
            v = (ny - j) / ny
            r = Ray(origin, lower_left_corner + u * horizontal + v * vertical)
            col = color(r, world)
            image[j, i] = col

    plt.imsave("image.png", image)


if __name__ == '__main__':
    render()
