import sys
from random import random

from matplotlib import pyplot as plt
from tqdm import trange

from hittable import *


class Camera:
    def __init__(self):
        self.lower_left_corner = np.array([-2.0, -1.0, -1.0])
        self.horizontal = np.array([4.0, 0.0, 0.0])
        self.vertical = np.array([0.0, 2.0, 0.0])
        self.origin = np.array([0.0, 0.0, 0.0])

    def get_ray(self, u, v):
        return Ray(self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin)


def color(ray: Ray, world: Hittable):
    rec = HitRecord(0, np.array([0, 0, 0]), np.array([0, 0, 0]))
    if world.hit(ray, 0, np.inf, rec):
        target = rec.p + rec.normal + random_in_unit_sphere()
        return 0.5 * color(Ray(rec.p, target - rec.p), world)
    else:
        unit_direction = ray.direction() / np.linalg.norm(ray.direction())
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])


def random_in_unit_sphere():
    p = 2 * np.array([random(), random(), random()]) - np.array([1, 1, 1])
    while p[0] * p[0] + p[1] * p[1] + p[2] * p[2] >= 1:
        p = 2 * np.array([random(), random(), random()]) - np.array([1, 1, 1])
    return p


def render():
    nx = 200
    ny = 100
    ns = 100

    cam = Camera()

    image = np.zeros((ny, nx, 3))

    world = Hittable([Sphere(np.array([0, 0, -1]), 0.5), Sphere(np.array([0, -100.5, -1]), 100)])

    for j in trange(ny):
        for i in range(nx):
            col = np.array([0, 0, 0])
            for s in range(ns):
                u = (i + random() / 2) / nx
                v = (ny - j + random() / 2) / ny
                r = cam.get_ray(u, v)
                p = r.point_at_parameter(2.0)
                col = col + color(r, world)
            col /= ns
            image[j, i] = col

    plt.imsave("image.png", image)


if __name__ == '__main__':
    render()
