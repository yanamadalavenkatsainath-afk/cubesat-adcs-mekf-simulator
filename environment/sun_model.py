import numpy as np

class SunModel:

    def __init__(self):
        # Arbitrary inertial sun direction
        self.s_I = np.array([1.0, 0.3, 0.2])
        self.s_I = self.s_I / np.linalg.norm(self.s_I)

    def get_sun_vector(self):
        return self.s_I
