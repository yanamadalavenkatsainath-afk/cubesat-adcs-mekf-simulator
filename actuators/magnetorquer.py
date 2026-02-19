import numpy as np

class Magnetorquer:

    def __init__(self, m_max=0.2):
        self.m_max = m_max

    def compute_dipole(self, h, B_body):

        B_norm_sq = np.dot(B_body, B_body)
        if B_norm_sq < 1e-12:
            return np.zeros(3)

        k_dump = 5
        m = -k_dump * np.cross(B_body, h) / B_norm_sq

        # Saturation
        m = np.clip(m, -self.m_max, self.m_max)

        return m

    def compute_torque(self, m, B_body):
        return np.cross(m, B_body)
