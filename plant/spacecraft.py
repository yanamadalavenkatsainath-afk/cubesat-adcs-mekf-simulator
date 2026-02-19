import numpy as np
from utils.quaternion import normalize

class Spacecraft:

    def __init__(self, inertia):
        self.I = inertia
        self.I_inv = np.linalg.inv(inertia)

        self.q = np.array([1., 0., 0., 0.])
        self.omega = np.zeros(3)

    def omega_dot(self, omega, torque):
        return self.I_inv @ (
            torque - np.cross(omega, self.I @ omega)
        )

    def quat_dot(self, q, omega):
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return 0.5 * Omega @ q

    def step(self, torque, disturbance, dt):

        def f(q, omega):
            qd = self.quat_dot(q, omega)
            wd = self.omega_dot(omega, torque + disturbance)
            return qd, wd

        k1_q, k1_w = f(self.q, self.omega)
        k2_q, k2_w = f(self.q + 0.5*dt*k1_q,
                       self.omega + 0.5*dt*k1_w)
        k3_q, k3_w = f(self.q + 0.5*dt*k2_q,
                       self.omega + 0.5*dt*k2_w)
        k4_q, k4_w = f(self.q + dt*k3_q,
                       self.omega + dt*k3_w)

        self.q += (dt/6)*(k1_q+2*k2_q+2*k3_q+k4_q)
        self.omega += (dt/6)*(k1_w+2*k2_w+2*k3_w+k4_w)

        self.q = normalize(self.q)

        return self.q, self.omega
