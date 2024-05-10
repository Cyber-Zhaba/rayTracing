from math import pi, tan
from random import random

from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map

from hittable import *
from materials import Lambertian, Metal, Glass
from objects import Sphere


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
        if depth < 50 and rec.material.scatter(ray, rec, attenuation, scattered):
            return attenuation * color(scattered, world, depth + 1)
        else:
            return np.array([0.0, 0.0, 0.0])
    else:
        unit_direction = ray.direction() / np.linalg.norm(ray.direction())
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])


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


def render():
    nx = 400
    ny = 300
    ns = 20

    look_from = np.array([8, 1.8, 3])
    look_at = np.array([0, 0, 0])
    dist_to_focus = np.array([4, 1, 0]) - look_at
    dist_to_focus = np.sqrt(dist_to_focus[0] ** 2 + dist_to_focus[1] ** 2 + dist_to_focus[2] ** 2)
    aperture = 0.02

    cam = Camera(look_from, look_at, np.array([0, 1, 0]), 50, nx / ny, aperture, dist_to_focus)

    world_list = [Sphere(np.array([0, -1000, 0]), 1000, Lambertian(np.array([0.5, 0.5, 0.5])))]

    for a in range(-11, 11):
        for b in range(-11, 11):
            mat = random()
            center = np.array([a + 0.9 * random(), 0.2, b + 0.9 * random()])
            if mat < 0.8:
                world_list.append(
                    Sphere(center, 0.2,
                           Lambertian(np.array([random() * random(), random() * random(), random() * random()])))
                )
            elif mat < 0.95:
                world_list.append(
                    Sphere(center, 0.2,
                           Metal(np.array([0.5 * (1 + random()), 0.5 * (1 + random()), 0.5 * (1 + random())])))
                )
            else:
                world_list.append(
                    Sphere(center, 0.2, Glass(1.5))
                )

    world_list.append(Sphere(np.array([0, 1, 0]), 1, Glass(1.5)))
    world_list.append(Sphere(np.array([-4, 1, 0]), 1, Lambertian(np.array([0.4, 0.2, 0.1]))))
    world_list.append(Sphere(np.array([4, 1, 0]), 1, Metal(np.array([0.7, 0.6, 0.5]), 0)))

    world = Hittable(world_list)

    image = np.zeros((ny, nx, 3))

    params = [(i, nx, ny, ns, cam, world) for i in range(ny)]
    results = process_map(compute_row, params, max_workers=15, chunksize=1, desc="Rendering")

    for i, row in enumerate(results):
        image[i] = row

    plt.imsave("image.png", image)


if __name__ == '__main__':
    render()
