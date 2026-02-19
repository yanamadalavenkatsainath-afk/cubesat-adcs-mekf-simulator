import numpy as np
from utils.quaternion import rot_matrix

class SunSensor:

    def __init__(self, sigma_noise=5e-4):
        self.sigma = sigma_noise

    def measure(self, q_true, sun_inertial):

        R = rot_matrix(q_true)
        s_body = R @ sun_inertial

        noise = self.sigma * np.random.randn(3)
        s_body = s_body + noise

        # Normalize (sun sensors measure direction)
        return s_body / np.linalg.norm(s_body)
