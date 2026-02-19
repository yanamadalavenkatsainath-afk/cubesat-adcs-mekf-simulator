import numpy as np
import matplotlib.pyplot as plt

from plant.spacecraft import Spacecraft
from sensors.gyro import Gyro
from sensors.magnetometer import Magnetometer
from sensors.sun_sensor import SunSensor
from environment.magnetic_field import MagneticField
from environment.sun_model import SunModel
from actuators.reaction_wheel import ReactionWheel
from actuators.magnetorquer import Magnetorquer
from control.attitude_controller import AttitudeController
from estimation.mekf import MEKF
from telemetry.logger import Logger
from utils.quaternion import quat_error

# =====================================================
# Simulation Parameters
# =====================================================
dt = 0.01
t_final = 150
time_array = np.arange(0, t_final, dt)

I = np.diag([10, 8, 5])

# =====================================================
# Initialize System Components
# =====================================================
sc = Spacecraft(I)

gyro = Gyro(dt=dt)
mag = Magnetometer()
sun_sensor = SunSensor()

mag_field = MagneticField()
sun_model = SunModel()

rw = ReactionWheel(h_max=0.5)
mtq = Magnetorquer(m_max=0.2)

controller = AttitudeController(Kp=0.3, Kd=0.8)
ekf = MEKF(dt)

logger = Logger()

q_ref = np.array([1., 0., 0., 0.])

# Initial body rates
sc.omega = np.array([0.01, -0.005, 0.008])

# =====================================================
# Simulation Loop
# =====================================================
for t in time_array:

    disturbance = np.array([1e-5, -1e-5, 5e-6])

    # -------------------------------------------------
    # Controller (uses estimated state)
    # -------------------------------------------------
    omega_est = sc.omega - ekf.bias
    torque_cmd, _ = controller.compute(ekf.q, omega_est, q_ref)
    torque_cmd = np.clip(torque_cmd, -0.5, 0.5)

    # Reaction Wheel Actuation
    torque_rw = rw.apply_torque(torque_cmd, dt)

    # Sensors BEFORE dump computation
    omega_meas = gyro.measure(sc.omega)
    B_I = mag_field.get_field()
    B_meas = mag.measure(sc.q, B_I)

    sun_I = sun_model.get_sun_vector()
    sun_meas = sun_sensor.measure(sc.q, sun_I)

    # Estimator
    ekf.predict(omega_meas)
    ekf.update_vector(B_meas, B_I, ekf.R_mag)
    ekf.update_vector(sun_meas, sun_I, ekf.R_sun)

    # Magnetorquer dumping
    m_cmd = mtq.compute_dipole(rw.h, B_meas)
    torque_mtq = mtq.compute_torque(m_cmd, B_meas)

    # TOTAL TORQUE (THIS IS IMPORTANT)
    total_torque = torque_rw + torque_mtq

    # Propagate spacecraft
    q_true, omega_true = sc.step(total_torque, disturbance, dt)


    # -------------------------------------------------
    # Log Telemetry
    # -------------------------------------------------
    logger.log(
        t,
        q_true,
        ekf.q,
        omega_true,
        torque_cmd,
        rw.h,
        omega_meas
    )

# =====================================================
# Save Data
# =====================================================
df = logger.save("telemetry.csv")

# =====================================================
# Professional Plot Formatting
# =====================================================
plt.rcParams.update({
    "font.size": 11,
    "figure.figsize": (8, 5),
    "axes.grid": True
})

# =====================================================
# 1️⃣ True vs Estimated Quaternion
# =====================================================
plt.figure()
plt.plot(df["time"], df["q1_true"], label="q1 True")
plt.plot(df["time"], df["q1_est"], '--', label="q1 Estimated")
plt.xlabel("Time [s]")
plt.ylabel("Quaternion Component")
plt.title("True vs Estimated Quaternion (q1)")
plt.legend()
plt.tight_layout()

# =====================================================
# 2️⃣ Attitude Estimation Error (Degrees)
# =====================================================
err_deg = []

for i in range(len(df)):
    qt = df.loc[i, ["q0_true","q1_true","q2_true","q3_true"]].values
    qe = df.loc[i, ["q0_est","q1_est","q2_est","q3_est"]].values
    qerr = quat_error(qt, qe)
    angle = 2 * np.linalg.norm(qerr[1:])
    err_deg.append(np.degrees(angle))

plt.figure()
plt.plot(df["time"], err_deg)
plt.xlabel("Time [s]")
plt.ylabel("Attitude Error [deg]")
plt.title("Attitude Estimation Error")
plt.tight_layout()

# =====================================================
# 3️⃣ Reaction Wheel Momentum
# =====================================================
plt.figure()
plt.plot(df["time"], df["hx"], label="h_x")
plt.plot(df["time"], df["hy"], label="h_y")
plt.plot(df["time"], df["hz"], label="h_z")
plt.xlabel("Time [s]")
plt.ylabel("Wheel Momentum [N·m·s]")
plt.title("Reaction Wheel Angular Momentum")
plt.legend()
plt.tight_layout()

# =====================================================
# 4️⃣ Commanded Control Torque
# =====================================================
plt.figure()
plt.plot(df["time"], df["Tx_cmd"], label="T_x")
plt.plot(df["time"], df["Ty_cmd"], label="T_y")
plt.plot(df["time"], df["Tz_cmd"], label="T_z")
plt.xlabel("Time [s]")
plt.ylabel("Control Torque [N·m]")
plt.title("Commanded Control Torque")
plt.legend()
plt.tight_layout()

# =====================================================
# 5️⃣ Gyroscope Noise Distribution
# =====================================================
noise = df[["gyro_x","gyro_y","gyro_z"]].values - \
        df[["wx_true","wy_true","wz_true"]].values

plt.figure()
plt.hist(noise[:, 0], bins=60, density=True)
plt.xlabel("Gyro Noise X [rad/s]")
plt.ylabel("Probability Density")
plt.title("Gyroscope Noise Distribution (X-Axis)")
plt.tight_layout()

plt.show()
