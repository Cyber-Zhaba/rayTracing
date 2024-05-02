from typing import List

import numpy as np


class Object:
    def hit(self, *args, **kwargs):
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


class HitRecord:
    def __init__(self, t: float, p: np.array, normal: np.array):
        self.t = t
        self.p = p
        self.normal = normal


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


class Sphere(Object):
    def __init__(self, center: np.array, radius: float):
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord):
        oc = ray.origin() - self.center
        a = np.dot(ray.direction(), ray.direction())
        b = np.dot(oc, ray.direction())
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - a * c
        if discriminant > 0:
            temp = (-b - np.sqrt(discriminant)) / a
            if t_min < temp < t_max:
                rec.t = temp
                rec.p = ray.point_at_parameter(rec.t)
                rec.normal = (rec.p - self.center) / self.radius
                return True
            temp = (-b + np.sqrt(discriminant)) / a
            if t_min < temp < t_max:
                rec.t = temp
                rec.p = ray.point_at_parameter(rec.t)
                rec.normal = (rec.p - self.center) / self.radius
                return True
        return False
