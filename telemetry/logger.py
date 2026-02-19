import pandas as pd
import numpy as np

class Logger:

    def __init__(self):
        self.data = []

    def log(self, t,
            q_true, q_est,
            omega_true,
            torque_cmd,
            wheel_momentum,
            gyro_meas):

        self.data.append([
            t,
            *q_true,
            *q_est,
            *omega_true,
            *torque_cmd,
            *wheel_momentum,
            *gyro_meas
        ])

    def save(self, filename="telemetry.csv"):

        columns = [
            "time",
            "q0_true","q1_true","q2_true","q3_true",
            "q0_est","q1_est","q2_est","q3_est",
            "wx_true","wy_true","wz_true",
            "Tx_cmd","Ty_cmd","Tz_cmd",
            "hx","hy","hz",
            "gyro_x","gyro_y","gyro_z"
        ]

        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv(filename, index=False)
        return df
