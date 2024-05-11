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


def refract(v: np.array, n: np.array, ni_over_nt: float, refracted: np.array):
    uv = v / np.linalg.norm(v)
    dt = np.dot(uv, n)
    discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt)

    if discriminant > 0:
        refracted += ni_over_nt * (uv - n * dt) - n * np.sqrt(discriminant) - refracted
        return True

    return False


def schlick(cosine: float, ref_idx: float):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 **= 2
    return r0 + (1 - r0) * (1 - cosine) ** 5


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


class Glass(Material):
    def __init__(self, ri: float):
        self.ref_idx = ri

    def scatter(self, ray: Ray, rec: HitRecord, attenuation: np.array, scattered: Ray):
        reflected = reflect(ray.direction(), rec.normal)
        attenuation += np.array([1.0, 1.0, 1.0]) - attenuation
        refracted = np.array([0.0, 0.0, 0.0])
        length = np.sqrt(ray.direction()[0] ** 2 + ray.direction()[1] ** 2 + ray.direction()[2] ** 2)

        if np.dot(ray.direction(), rec.normal) > 0:
            outward_normal = -rec.normal
            ni_over_nt = self.ref_idx
            cosine = (self.ref_idx * np.dot(ray.direction(), rec.normal)) / length
        else:
            outward_normal = rec.normal
            ni_over_nt = 1 / self.ref_idx
            cosine = -np.dot(ray.direction(), rec.normal) / length

        if refract(ray.direction(), outward_normal, ni_over_nt, refracted):
            reflected_prob = schlick(cosine, self.ref_idx)
        else:
            reflected_prob = 1.0

        if random() < reflected_prob:
            scattered += Ray(rec.p, reflected) - scattered
        else:
            scattered += Ray(rec.p, refracted) - scattered

        return True


class Light(Material):
    def __init__(self, albedo: np.array):
        self.albedo = albedo

    def scatter(self, ray: Ray, rec: HitRecord, attenuation: np.array, scattered: Ray):
        attenuation += self.albedo - attenuation
        return True
