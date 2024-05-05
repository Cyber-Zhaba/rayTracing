from random import random
from typing import override

import numpy as np

from hittable import Material, Ray, HitRecord


def random_in_unit_sphere():
    p = 2 * np.array([random(), random(), random()]) - np.array([1, 1, 1])
    while p[0] * p[0] + p[1] * p[1] + p[2] * p[2] >= 1:
        p = 2 * np.array([random(), random(), random()]) - np.array([1, 1, 1])
    return p


def reflect(v: np.array, n: np.array):
    return v - 2 * np.dot(v, n) * n


class Lambertian(Material):
    def __init__(self, albedo: np.array):
        self.albedo = albedo

    @override
    def scatter(self, ray: Ray, rec: HitRecord, attenuation: np.array, scattered: Ray):
        target = rec.p + rec.normal + random_in_unit_sphere()
        scattered += Ray(rec.p, target - rec.p) - scattered
        attenuation += self.albedo - attenuation
        return True


class Metal(Material):
    def __init__(self, albedo: np.array, fuzz: float = 0):
        self.fuzz = fuzz if fuzz < 1 else 1
        self.albedo = albedo

    @override
    def scatter(self, ray: Ray, rec: HitRecord, attenuation: np.array, scattered: Ray):
        reflected = reflect(ray.direction(), rec.normal)
        scattered += Ray(rec.p, reflected + self.fuzz * random_in_unit_sphere()) - scattered
        attenuation += self.albedo - attenuation
        return np.dot(scattered.direction(), rec.normal) > 0
