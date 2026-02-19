import numpy as np
from utils.quaternion import rot_matrix

class Magnetometer:

    def __init__(self, sigma_noise=5e-7):
        self.sigma = sigma_noise

    def measure(self, q_true, B_I):

        R = rot_matrix(q_true)
        B_body = R @ B_I

        noise = self.sigma * np.random.randn(3)
        return B_body + noise
