import numpy as np

class Gyro:

    def __init__(self, sigma_noise=1e-3, sigma_bias=1e-5, dt=0.01):
        self.sigma_noise = sigma_noise
        self.sigma_bias = sigma_bias
        self.dt = dt
        self.bias = np.zeros(3)

    def update_bias(self):
        self.bias += self.sigma_bias * np.sqrt(self.dt) * np.random.randn(3)

    def measure(self, omega_true):
        self.update_bias()
        noise = self.sigma_noise * np.random.randn(3)
        return omega_true + self.bias + noise
