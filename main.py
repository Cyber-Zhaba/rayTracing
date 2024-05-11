import math
from math import pi, tan
from random import random, randint

import numpy as np
from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map

from hittable import *
from materials import Lambertian, Metal, Glass, Light
from objects import Sphere, Plane, Cube, Icosahedron


def random_in_unit_disk():
    p = 2 * np.array([random(), random(), random()]) - np.array([1, 1, 0]).astype(float)
    while np.dot(p, p) >= 1.0:
        p = 2 * np.array([random(), random(), random()]) - np.array([1, 1, 0]).astype(float)
    return p


class Camera:
    def __init__(self, look_from, look_at, vup, fov, aspect, aperture, focus_dist):
        self.lens_radius = aperture / 2
        self.theta = fov * pi / 180
        self.half_height = tan(self.theta / 2)
        self.half_weight = aspect * self.half_height
        self.origin = look_from

        w = look_from - look_at
        w = w / np.linalg.norm(w)

        cross = lambda v1, v2: np.array([v1[1] * v2[2] - v1[2] * v2[1],
                                         -(v1[0] * v2[2] - v1[2] * v2[0]),
                                         v2[0] * v2[1] - v1[1] * v2[0]])

        u = cross(vup, w)
        u = u / np.linalg.norm(u)

        v = cross(w, u)

        self.lower_left_corner = self.origin - self.half_weight * u * focus_dist - self.half_height * v * focus_dist - w * focus_dist
        self.horizontal = 2 * self.half_weight * u * focus_dist
        self.vertical = 2 * self.half_height * v * focus_dist

    def get_ray(self, u, v):
        rd = self.lens_radius * random_in_unit_disk()
        offset = u * rd[0] + v * rd[1]
        return Ray(self.origin + offset,
                   self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin - offset)


def color(ray: Ray, world: Hittable, depth=0):
    rec = HitRecord(0, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), Material())
    if world.hit(ray, 0.001, np.inf, rec):
        scattered = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        attenuation = np.array([0.0, 0.0, 0.0])

        beam = rec.material.scatter(ray, rec, attenuation, scattered)

        if isinstance(rec.material, Light):
            return attenuation
        elif depth < 50 and beam:
            return attenuation * color(scattered, world, depth + 1)
        else:
            return np.array([0.0, 0.0, 0.0])
    else:
        return np.array([0, 0, 0])


def compute_row(args):
    j, nx, ny, ns, cam, world = args
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


def render(nx, ny, ns, cam, world, filename):

    image = np.zeros((ny, nx, 3))

    params = [(i, nx, ny, ns, cam, world) for i in range(ny)]
    results = process_map(compute_row, params, max_workers=15, chunksize=1, desc="Rendering")

    for i, row in enumerate(results):
        image[i] = row

    plt.imsave(f"{filename}", image)


def main():
    nx = 500
    ny = nx * 3 // 5
    ns = 100 * 5

    look_from = np.array([0, 3, 0])
    look_at = np.array([0, 3, 10])
    dist_to_focus = look_from - look_at
    dist_to_focus = np.sqrt(dist_to_focus[0] ** 2 + dist_to_focus[1] ** 2 + dist_to_focus[2] ** 2)
    aperture = 0.02

    cam = Camera(look_from, look_at, np.array([0, 1, 0]), 80, nx / ny, aperture, dist_to_focus)

    lst = [
        Plane(np.array([0, 0, 0]), np.array([0, 1, 0]), Lambertian(np.array([1, 1, 1]))),

        Cube(np.array([-8, 4, 4]), 8, Lambertian(np.array([0.9, 0.5, 0.32]))),
        Cube(np.array([8, 4, 4]), 8, Lambertian(np.array([0, 0, 1]))),
        Cube(np.array([0, 4, 12]), 8, Lambertian(np.array([1, 1, 1]))),
        Cube(np.array([0, 4, -4]), 8, Lambertian(np.array([1, 1, 1]))),
        Cube(np.array([0, 12, 4]), 8, Light(np.array([1, 1, 1]))),

        Icosahedron(np.array([1, 2, 4]), 2, Lambertian(np.array([0.56, 0, 0.62]))),
        Cube(np.array([-3, 1.5, 3]), 1.5, Metal(np.array([0.4, 0.6, 0.7]), 0.3), pi / 3),
    ]

    world = Hittable(lst)

    render(nx, ny, ns, cam, world, F"image.png")


if __name__ == '__main__':
    main()
