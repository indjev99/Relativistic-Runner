import numpy as np


class LabFrame:
    def __init__(self, velocity):
        self.update(velocity)

    def update(self, velocity):
        self.velocity = velocity
        self.gamma = 1 / np.sqrt(1 - np.linalg.norm(velocity) ** 2)
        self.normals = np.array([0, 1]) if np.linalg.norm(velocity) == 0 else velocity / np.linalg.norm(velocity) 
        self.transformation = np.array([[self.gamma,
                                         -self.gamma * self.velocity[0],
                                         -self.gamma * self.velocity[1]],
                                        [-self.gamma * self.velocity[0],
                                         1 + (self.gamma - 1) * self.normals[0] ** 2,
                                        (self.gamma - 1) * self.normals[0] * self.normals[1]],
                                        [-self.gamma * self.velocity[1],
                                         (self.gamma - 1) * self.normals[0] * self.normals[1],
                                         1 + (self.gamma - 1) * self.normals[1] ** 2]])

    def transform(self, event):
        return np.dot(self.transformation, event)

    def transform_polygon(self, polygon):
        return [self.transform(point) for point in polygon]

    def doppler_shift(self, wavelength, event):
        theta_velocity = np.arctan2(self.velocity[1], self.velocity[0])
        theta_position = np.arctan2(event[2], event[1])
        theta = theta_velocity - theta_position
        return wavelength * (self.gamma * (1 + np.linalg.norm(self.velocity) * np.cos(theta)))

    def get_mass(self, mass_0):
        return self.gamma * mass_0
