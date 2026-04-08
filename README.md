# Autonomous Spacecraft Rendezvous — Relative Navigation & GNC Simulation

A flight-representative, closed-loop Guidance, Navigation and Control simulation for autonomous proximity operations between two spacecraft in LEO, implemented entirely in Python with no proprietary dependencies.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-100%20runs-orange)

---

## Results Summary

| Metric | Result |
|--------|--------|
| Attitude gate confirmed | 100 / 100 runs |
| Rendezvous triggered | 95 / 100 runs |
| Successful rendezvous (final range < 15 m) | 83 / 100 runs |
| Mean final range | 6.3 m |
| Mean ΔV | 66 mm/s |

### Option 1 — Shortened Transfer Window

| Metric | Baseline | Short Transfer |
|--------|----------|----------------|
| Rendezvous success | 83 / 100 | 89 / 100 |
| Mean final range | 6.3 m | 3.47 m |
| Mean ΔV | 66 mm/s | 138 mm/s |

---

## What This Is

Two spacecraft. Same 500 km circular orbit. The chief holds passively. The deputy starts 100 m behind it along the flight path.

In LEO, uncontrolled relative motion doesn't stay small — Coriolis and tidal gravity pull the deputy along a curved drift path. Leave it alone and it never comes back. The deputy has to close that gap autonomously, with no GPS and no ground contact. Its only sensor is a laser rangefinder returning a single scalar distance to the chief at each timestep.

From that one measurement, the system reconstructs the full 6-state relative position and velocity, plans two impulsive burns, and executes a ~46-minute transfer to bring the deputy to rest within metres of the chief.

---

## GNC Pipeline

### Navigation

- **CW-EKF**: Extended Kalman Filter formulated in the Clohessy-Wiltshire (Hill) frame. Reconstructs all 6 relative states — position and velocity — from scalar range measurements alone.
- **ROE-EKF**: Parallel filter tracking the same state through Relative Orbital Elements, absorbing secular J2 drift that the linearised CW equations ignore.

### Guidance (Burn Planning)

Once both filters converge, the planner solves a two-impulse boundary value problem using the CW state transition matrix: find two burns separated by time T such that the deputy arrives at the chief with near-zero relative velocity.

- **Burn 1** fires on trigger.
- **~46 minutes** of ballistic coast.
- **Burn 2** brakes to rest.
- **300-second hold** then final range measurement.

### The Accuracy–Fuel Trade-off

The baseline planner minimises ΔV, which biases it toward longer transfer times. Longer coast accumulates velocity estimation error — and that error drives Burn 2 misses. Two paths forward:

- **Option 1 — Shorten the transfer window**: better Burn 2 accuracy, but fuel doubles. Success improved from 83% to 89%, mean final range halved to 3.47 m, mean ΔV doubled to 138 mm/s.
- **Option 2 — Lambert's solver**: solve in inertial space directly, no linearisation, exact boundary conditions, globally optimal for the given time-of-flight. Both accuracy and fuel. That's the next build.

---

## Previous Work: ADCS Simulation

This project extends a previously validated 3U CubeSat Attitude Determination and Control System (ADCS) simulation. That simulation established the attitude estimation and control stack that gates the rendezvous planner here.

### ADCS Results Summary

| Metric | Result |
|--------|--------|
| QUEST acceptance rate | 99 / 100 runs |
| MEKF convergence rate | 95 / 100 runs |
| Steady-state pointing error | < 0.5° in 95% of runs |
| Wheel saturation events | 0 / 100 runs |
| End-to-end ADCS mission success | 95 / 100 runs |

### ADCS Pipeline

- **Environment**: SGP4 orbit propagation, IGRF-13 magnetic field, NRLMSISE-00 atmospheric density, SRP with dual-cone eclipse model, gravity gradient torque
- **Sensors**: Gyro (Allan variance noise model), magnetometer, sun sensor — all with hardware-representative noise
- **Estimator**: QUEST (3-vector: mag + sun + nadir) for initialisation, 6-state MEKF with Joseph-form covariance update for fine pointing
- **Control**: B-dot detumbling, PD attitude controller, cross-product momentum desaturation
- **FSW**: 5-mode hierarchical state machine (SAFE → DETUMBLE → SUN_ACQUISITION → FINE_POINTING → MOMENTUM_DUMP)

---

## Project Structure

```
flight sim/
│
├── main.py                    # Single-run simulation entry point (Phase 1 + Phase 2)
├── monte_carlo.py             # 100-run Monte Carlo validation
├── requirements.txt
│
├── plant/
│   └── spacecraft.py          # Rigid body dynamics (Euler equations + quaternion kinematics)
│
├── sensors/
│   ├── gyro.py                # Allan variance gyro model (ARW + BI + RRW)
│   ├── magnetometer.py        # MEMS magnetometer with Gaussian noise
│   ├── sun_sensor.py          # Coarse sun sensor array model
│   └── ranging_sensor.py      # Range + bearing sensor (σ_range=0.5m, σ_angle=0.1°, FOV=60°)
│
├── environment/
│   ├── orbit.py               # SGP4/SDP4 orbit propagation
│   ├── magnetic_field.py      # IGRF-13 geomagnetic field
│   ├── sun_model.py           # Sun vector ephemeris
│   ├── aerodynamic_drag.py    # NRLMSISE-00 drag torque
│   ├── gravity_gradient.py    # Gravity gradient torque
│   ├── solar_radiation_pressure.py  # SRP torque + eclipse
│   ├── cw_dynamics.py         # Clohessy-Wiltshire relative motion propagator
│   └── roe_dynamics.py        # J2-perturbed relative orbital elements propagator
│
├── estimation/
│   ├── quest.py               # QUEST algorithm (Wahba's problem, 3-vector)
│   ├── mekf.py                # 6-state MEKF with Joseph form (attitude)
│   ├── cw_ekf.py              # CW-frame EKF (6-state relative position + velocity)
│   └── roe_ekf.py             # ROE-EKF (J2-aware, feeds rendezvous planner)
│
├── control/
│   ├── attitude_controller.py # PD attitude controller
│   ├── rendezvous_controller.py  # Two-impulse CW BVP planner (RelNavMode FSM)
│   └── roe_controller.py      # ROE formation hold + rendezvous (ROEMode FSM)
│
├── actuators/
│   ├── reaction_wheel.py      # Reaction wheel momentum model
│   ├── magnetorquer.py        # Magnetorquer torque + desaturation law
│   └── bdot.py                # B-dot detumble controller
│
├── fsw/
│   └── mode_manager.py        # 5-mode FSW state machine (SAFE → … → MOMENTUM_DUMP)
│
└── utils/
    └── quaternion.py          # Quaternion algebra (multiply, error, rotation matrix)
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/cubesat-adcs.git
cd cubesat-adcs
pip install -r requirements.txt
```

### 2. Run single simulation (60 min mission)

```bash
python main.py
```

Produces two matplotlib figures:
- **Figure 1**: Full mission overview — relative position/velocity, filter residuals, burn events, FSW mode timeline
- **Figure 2**: CW-EKF and ROE-EKF estimation error during coast and terminal phases

### 3. Run Monte Carlo validation (100 runs)

```bash
python monte_carlo.py
```

Produces `monte_carlo_results.png` with subplots covering filter convergence, rendezvous trigger rate, final range CDF, ΔV distribution, and mode reach statistics. Monte Carlo is randomised over initial tumble rate, sensor noise, relative position jitter, and trigger timing.

---

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Initial separation | 100 m (along-track) | `main.py` |
| Orbit altitude | 500 km circular | `main.py` |
| Rangefinder noise (1σ) | 0.5 m | `sensors/rangefinder.py` |
| CW-EKF process noise | tunable | `navigation/cw_ekf.py` |
| Transfer time T | 46 min (baseline) | `guidance/burn_planner.py` |
| Rendezvous gate (final range) | < 15 m | `monte_carlo.py` |
| Attitude gate threshold | configurable | `fsw/mode_manager.py` |
| Inertia matrix | diag(0.030, 0.025, 0.010) kg·m² | `main.py` |
| B-dot gain k_bdot | 2×10⁵ A·m²·s/T | `main.py` |
| PD gains Kp / Kd | 0.0005 / 0.008 | `main.py` |
| QUEST quality threshold | 0.01 | `main.py` |
| MEKF Mahalanobis gate | 16.0 (4-sigma) | `estimation/mekf.py` |

---

## Algorithm Notes

### CW-EKF
The Clohessy-Wiltshire equations linearise relative motion about a circular reference orbit. The EKF propagates a 6-state (position + velocity in the Hill frame) using the analytic CW state transition matrix, and updates from scalar range measurements via a nonlinear observation function. The scalar range observability is limited — the filter requires a transient in the relative trajectory or initial state dispersion to fully resolve all six states.

### ROE-EKF
Tracks the same relative state parameterised as Relative Orbital Elements. The ROE formulation handles J2 perturbations naturally through mean-to-osculating element conversions, capturing the slow secular drift that accumulates over a ~46-minute coast and causes baseline Burn 2 misses.

### Two-Impulse Burn Planner
Solves the CW boundary value problem: given current state estimate and a target state of zero relative position and velocity, find impulsive ΔV₁ and ΔV₂ separated by time T. Solution uses the analytic inverse of the CW state transition matrix partitioned into position and velocity sub-blocks. Minimum-ΔV weighting biases toward longer T, which is the root cause of the 17% failure rate.

### MEKF (Attitude)
6-state error state: [δθ (3), δbias (3)]. Joseph-form covariance update for numerical stability. QUEST-assisted reinitialisation if attitude error exceeds 25°.

### Spacecraft Dynamics
Euler's equation with gyroscopic coupling:
```
I·ω̇ = τ_ext + τ_rw - ω×(I·ω) - ω×h_rw
```
The `ω×h_rw` term is essential for correct desaturation physics.

---

## Systems Engineering Documentation

Available in `/docs`:

| Document | Contents |
|----------|----------|
| `ADCS_Requirements_FlowDown.docx` | Mission → system → subsystem requirements with verification methods |
| `ADCS_FMECA.docx` | Failure modes, effects and criticality analysis (8 entries) |
| `ADCS_Technical_Analysis.docx` | Disturbance margins, power budget, verification strategy |
| `CubeSat_ADCS_Brief.docx` | Full technical project brief |

---

## References

- Clohessy & Wiltshire, *Terminal Guidance System for Satellite Rendezvous*, Journal of the Aerospace Sciences, 1960
- Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*, Springer 2014
- Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.
- IGRF-13: Alken et al., *Earth, Planets and Space*, 2021
- NRLMSISE-00: Picone et al., *Journal of Geophysical Research*, 2002
- ECSS-E-ST-60-30C: Satellite Attitude and Orbit Control System Standard

---

## License

MIT License — free to use, modify, and distribute with attribution.