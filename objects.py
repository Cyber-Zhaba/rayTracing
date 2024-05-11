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


class Plane(Object):
    def __init__(self, point: np.array, normal: np.array, material: Material):
        self.point = point
        self.normal = normal
        self.material = material

    @override
    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord):
        numerator = np.dot(self.normal, (self.point - ray.origin()))
        denominator = np.dot(self.normal, ray.direction())
        if denominator != 0:
            t = numerator / denominator
            if t_min < t < t_max:
                rec.t = t
                rec.p = ray.point_at_parameter(rec.t)
                rec.normal = self.normal
                rec.material = self.material
                return True
        return False


class Cube(Object):
    def __init__(self, center: np.array, side_length: float, material: Material, theta: float = 0.0):
        self.center = center
        self.side_length = side_length
        self.material = material
        self.theta = theta
        self.min_bound = self.center - side_length / 2
        self.max_bound = self.center + side_length / 2

    def rotate(self, point: np.array) -> np.array:
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(self.theta), 0, np.sin(self.theta)],
            [0, 1, 0],
            [-np.sin(self.theta), 0, np.cos(self.theta)]
        ])
        # Rotate the point around the center of the cube
        return np.dot(rotation_matrix, point - self.center) + self.center

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord):
        # Ray-box intersection algorithm
        ray_origin_rotated = self.rotate(ray.origin())
        ray_direction_rotated = self.rotate(ray.direction())
        inv_direction = 1.0 / ray_direction_rotated
        t0 = (self.min_bound - ray_origin_rotated) * inv_direction
        t1 = (self.max_bound - ray_origin_rotated) * inv_direction
        tmin = np.max(np.minimum(t0, t1))
        tmax = np.min(np.maximum(t0, t1))

        if tmax < tmin:
            return False

        # Check if the intersection point is within the valid range
        temp = tmin if tmin >= t_min and tmin <= t_max else tmax
        if temp < t_min or temp > t_max:
            return False

        rec.t = temp
        rec.p = ray.point_at_parameter(rec.t)

        # Determine the normal based on which face was hit
        epsilon = 0.0001  # Small value to handle floating-point imprecisions
        for i in range(3):
            if np.abs(rec.p[i] - self.min_bound[i]) < epsilon:
                rec.normal = -np.eye(3)[i]
                break
            elif np.abs(rec.p[i] - self.max_bound[i]) < epsilon:
                rec.normal = np.eye(3)[i]
                break

        rec.material = self.material
        return True


class Icosahedron(Object):
    def __init__(self, center: np.array, edge_length: float, material: Material):
        self.center = center
        self.edge_length = edge_length
        self.material = material
        self.theta = -np.arccos(1 / np.sqrt(5))  # Rotation angle
        self.vertices = self.calculate_vertices()
        self.faces = self.calculate_faces()

    def rotate(self, point: np.array) -> np.array:
        # Rotation matrix for rotation around x-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(self.theta), -np.sin(self.theta)],
            [0, np.sin(self.theta), np.cos(self.theta)]
        ])
        # Rotate the point around the center of the icosahedron
        return np.dot(rotation_matrix, point - self.center) + self.center

    def calculate_vertices(self):
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = np.array([
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1]
        ])
        vertices *= self.edge_length / np.sqrt(5)
        vertices += self.center
        # Rotate the vertices
        vertices = np.array([self.rotate(vertex) for vertex in vertices])
        return vertices

    # Rest of the class remains the same

    # Rest of the class remains the same
    def calculate_faces(self):
        faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1]
        ]
        return faces

    def hit(self, ray: Ray, t_min: float, t_max: float, rec: HitRecord):
        # Ray-triangle intersection with each face
        hit_anything = False
        closest_t = t_max

        for face_indices in self.faces:
            vertices = self.vertices[face_indices]
            v0, v1, v2 = vertices[0], vertices[1], vertices[2]

            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray.direction(), edge2)
            a = np.dot(edge1, h)

            if -1e-6 < a < 1e-6:
                continue  # Ray parallel to the triangle

            f = 1.0 / a
            s = ray.origin() - v0
            u = f * np.dot(s, h)

            if u < 0.0 or u > 1.0:
                continue

            q = np.cross(s, edge1)
            v = f * np.dot(ray.direction(), q)

            if v < 0.0 or u + v > 1.0:
                continue

            t = f * np.dot(edge2, q)

            if t > t_min and t < closest_t:
                closest_t = t
                hit_anything = True
                rec.t = t
                rec.p = ray.point_at_parameter(rec.t)
                rec.normal = np.cross(edge1, edge2)
                rec.material = self.material

        return hit_anything
