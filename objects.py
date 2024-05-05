from typing import override

import numpy as np

from hittable import Object, HitRecord, Ray, Material


class Sphere(Object):
    def __init__(self, center: np.array, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    @override
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
                rec.material = self.material
                return True
            temp = (-b + np.sqrt(discriminant)) / a
            if t_min < temp < t_max:
                rec.t = temp
                rec.p = ray.point_at_parameter(rec.t)
                rec.normal = (rec.p - self.center) / self.radius
                rec.material = self.material
                return True
        return False
