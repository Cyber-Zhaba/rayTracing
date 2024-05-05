from typing import List

import numpy as np


class Object:
    def hit(self, *args, **kwargs):
        pass


class Material:
    def scatter(self, *args, **kwargs):
        pass


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

    def __sub__(self, other):
        return Ray(self.A - other.A, self.B - other.B)

    def __iadd__(self, other):
        self.A = self.A + other.A
        self.B = self.B + other.B
        return self

    def __add__(self, other):
        if isinstance(other, Ray):
            return Ray(self.A + other.A, self.B + other.B)
        return Ray(self.A + other, self.B + other)


class HitRecord:
    def __init__(self, t: float, p: np.array, normal: np.array, m: Material):
        self.t = t
        self.p = p
        self.normal = normal
        self.material = m


class Hittable:
    def __init__(self, lst: List[Object]):
        self.list: List[Object] = lst

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord):
        hit_anything = False
        closest_so_far = t_max
        for obj in self.list:
            if obj.hit(ray, t_min, closest_so_far, rec):
                hit_anything = True
                closest_so_far = rec.t
        return hit_anything
