import numpy as np

class MagneticField:

    def __init__(self):
        # Simplified constant inertial magnetic field (Tesla)
        self.B_I = np.array([2e-5, -1e-5, 3e-5])

    def get_field(self):
        return self.B_I
