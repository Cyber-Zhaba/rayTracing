from random import random

from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map

from hittable import *
from materials import Lambertian, Metal
from objects import Sphere


class Camera:
    def __init__(self):
        self.lower_left_corner = np.array([-2.0, -1.0, -1.0])
        self.horizontal = np.array([4.0, 0.0, 0.0])
        self.vertical = np.array([0.0, 2.0, 0.0])
        self.origin = np.array([0.0, 0.0, 0.0])

    def get_ray(self, u, v):
        return Ray(self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin)


def color(ray: Ray, world: Hittable, depth=0):
    rec = HitRecord(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), Material())
    if world.hit(ray, 0.001, np.inf, rec):
        scattered = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        attenuation = np.array([0.0, 0.0, 0.0])
        if depth < 50 and rec.material.scatter(ray, rec, attenuation, scattered):
            return attenuation * color(scattered, world, depth + 1)
        else:
            return np.array([0.0, 0.0, 0.0])
    else:
        unit_direction = ray.direction() / np.linalg.norm(ray.direction())
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])


def compute_row(args):
    j, nx, ny, ns, world, cam = args
    row = np.zeros((nx, 3))
    for i in range(nx):
        col = np.array([0.0, 0.0, 0.0])
        for s in range(ns):
            u = (i + random()) / nx
            v = (ny - j + random()) / ny
            r = cam.get_ray(u, v)
            col = col + color(r, world)
        col /= ns
        col = np.sqrt(col)
        row[i] = col
    return row


def render():
    nx = 200
    ny = 100
    ns = 100

    cam = Camera()

    image = np.zeros((ny, nx, 3))

    world = Hittable(
        [
            Sphere(np.array([0, 0, -1]), 0.5, Lambertian(np.array([0.8, 0.3, 0.3]))),
            Sphere(np.array([0, -100.5, -1]), 100, Lambertian(np.array([0.8, 0.8, 0.0]))),
            Sphere(np.array([1, 0, -1]), 0.5, Metal(np.array([0.8, 0.6, 0.2]), 1.0)),
            Sphere(np.array([-1, 0, -1]), 0.5, Metal(np.array([0.8, 0.8, 0.8]), 0.3)),
        ],
    )

    params = [(i, nx, ny, ns, world, cam) for i in range(ny)]
    results = process_map(compute_row, params, max_workers=8, chunksize=1, desc="Rendering")

    for i, row in enumerate(results):
        image[i] = row

    plt.imsave("image.png", image)


if __name__ == '__main__':
    render()
