import numpy as np
from utils.quaternion import quat_multiply, normalize, rot_matrix


class MEKF:

    def __init__(self, dt):
        self.dt = dt

        # Nominal state
        self.q = np.array([1., 0., 0., 0.])
        self.bias = np.zeros(3)
        
        # Covariance
        self.P = np.eye(6) * 0.1
        self.P[3:6, 3:6] = np.eye(3) * 1e-4

        self.Q = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
        self.R_mag = np.eye(3) * 1e-12
        self.R_sun = np.eye(3) * 1e-6   


    def predict(self, omega_m):

        omega = omega_m - self.bias

        wx, wy, wz = omega

        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])

        self.q += 0.5 * self.dt * Omega @ self.q
        self.q = normalize(self.q)

        F = np.zeros((6, 6))
        F[0:3, 3:6] = -np.eye(3)

        Phi = np.eye(6) + F * self.dt

        self.P = Phi @ self.P @ Phi.T + self.Q


    def update_vector(self, z_body, v_inertial, R):
        Rb = rot_matrix(self.q)
        z_pred = Rb @ v_inertial

        vx, vy, vz = z_pred
        skew = np.array([
        [0, -vz, vy],
        [vz, 0, -vx],
        [-vy, vx, 0]
        ])

        H = np.zeros((3, 6))
        H[:, 0:3] = -skew

        y = z_body - z_pred

        # Innovation gate — reject outlier measurements
        S = H @ self.P @ H.T + R
        mahal = y @ np.linalg.inv(S) @ y
        if mahal > 20.0:   # chi-squared threshold, 3-DOF
           return         # skip this update, measurement is an outlier

        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        self.P = (np.eye(6) - K @ H) @ self.P

        dtheta = dx[0:3]
        db = dx[3:6]

        dq = np.hstack([1., 0.5 * dtheta])
        self.q = quat_multiply(dq, self.q)
        self.q = normalize(self.q)
        self.bias += db
