# 🛰️ Spacecraft Attitude Estimation & Control Simulation

A Python simulation of spacecraft attitude estimation using a **Multiplicative Extended Kalman Filter (MEKF)** with reaction wheel control and magnetorquer momentum dumping.

---

## 📁 Project Structure

```
spacecraft-sim/
├── main.py                  # Simulation entry point
├── plant/
│   └── spacecraft.py        # Spacecraft dynamics (RK4 integration)
├── sensors/
│   ├── gyro.py              # Gyroscope with bias random walk
│   ├── magnetometer.py      # Magnetometer with Gaussian noise
│   └── sun_sensor.py        # Sun sensor with Gaussian noise
├── actuators/
│   ├── reaction_wheel.py    # Reaction wheel with momentum saturation
│   └── magnetorquer.py      # Magnetorquer for momentum dumping
├── estimation/
│   └── mekf.py              # Multiplicative EKF (attitude + bias)
├── control/
│   └── attitude_controller.py  # PD attitude controller
├── environment/
│   ├── magnetic_field.py    # Simplified constant magnetic field model
│   └── sun_model.py         # Simplified sun vector model
├── telemetry/
│   └── logger.py            # Telemetry logging to CSV
├── utils/
│   └── quaternion.py        # Quaternion math utilities
├── telemetry.csv            # Output data (generated on run)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourname/spacecraft-sim.git
cd spacecraft-sim
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Simulation

```bash
python main.py
```

This will run a 150-second simulation and display 5 plots:
1. True vs Estimated Quaternion (q1)
2. Attitude Estimation Error [deg]
3. Reaction Wheel Angular Momentum
4. Commanded Control Torque
5. Gyroscope Noise Distribution

Telemetry is saved to `telemetry.csv`.

---

## 🧠 System Overview

| Component | Description |
|---|---|
| **Dynamics** | Euler's equations, RK4 integration |
| **Estimator** | MEKF — estimates quaternion + gyro bias |
| **Sensors** | Gyro, magnetometer, sun sensor |
| **Control** | PD controller on quaternion error |
| **Actuators** | Reaction wheels (momentum limited) + magnetorquers (desaturation) |

---

## 🔧 Key Tuning Parameters

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `Kp` | `attitude_controller.py` | 0.5 | Proportional gain — higher = faster response |
| `Kd` | `attitude_controller.py` | 0.1 | Derivative gain — higher = more damping |
| `self.Q` | `mekf.py` | `1e-7 * I` | Process noise — higher = trust sensor more |
| `self.R_mag` | `mekf.py` | `1e-12 * I` | Magnetometer noise — match sensor sigma² |
| `self.R_sun` | `mekf.py` | `1e-6 * I` | Sun sensor noise — match sensor sigma² |
| `self.P` | `mekf.py` | `1e-2 * I` | Initial covariance — higher = filter corrects faster |
| `h_max` | `reaction_wheel.py` | 0.05 N·m·s | Wheel momentum saturation limit |

---

## 📊 Expected Results (Healthy Simulation)

- ✅ Quaternion estimate tracks truth closely
- ✅ Attitude error converges toward 0° and stays below ~1°
- ✅ Gyro noise is zero-mean Gaussian
- ✅ Control torque decays as spacecraft stabilizes
- ✅ Reaction wheel momentum stays within ±h_max

---

## 📦 Dependencies

See `requirements.txt`. Core libraries: `numpy`, `matplotlib`, `pandas`.

---

# Final working parameters — add these to your README tuning table
h_max = 0.5       # N·m·s
Kp = 0.3
Kd = 0.8
sc.omega = [0.01, -0.005, 0.008]   # rad/s initial rates