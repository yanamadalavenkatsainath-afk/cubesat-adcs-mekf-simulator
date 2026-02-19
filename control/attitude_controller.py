import numpy as np
from utils.quaternion import quat_error

class AttitudeController:

    def __init__(self, Kp=0.3, Kd=0.08):
        self.Kp = Kp
        self.Kd = Kd

    def compute(self, q_est, omega_est, q_ref):
        q_err = quat_error(q_ref, q_est)
        torque = -self.Kp * q_err[1:] - self.Kd * omega_est
        return torque, q_err
