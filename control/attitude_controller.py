import numpy as np
from utils.quaternion import quat_error

class AttitudeController:
    """
    PD attitude controller.

    Outputs the desired corrective torque τ_des that the reaction wheel
    should store (i.e. the torque commanded TO the wheel).  Because
    spacecraft.py subtracts τ_rw_cmd from the body (Newton 3rd law), a
    positive τ_des here produces a stabilising effect:

        body_accel ∝  -τ_des  →  -(-Kp·q_err - Kd·ω)  =  +Kp·q_err + Kd·ω

    which drives q_err and ω to zero.
    """

    def __init__(self, Kp=0.3, Kd=0.08):
        self.Kp = Kp
        self.Kd = Kd

    def compute(self, q_est, omega_est, q_ref):
        q_err = quat_error(q_ref, q_est)
        # Positive torque commanded to wheel → negative reaction on body → stabilises
        torque = self.Kp * q_err[1:] + self.Kd * omega_est
        return torque, q_err