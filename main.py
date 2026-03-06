import numpy as np
import matplotlib.pyplot as plt
import sys

from plant.spacecraft import Spacecraft
from sensors.gyro import Gyro
from sensors.magnetometer import Magnetometer
from sensors.sun_sensor import SunSensor
from environment.magnetic_field import MagneticField
from environment.sun_model import SunModel
from environment.orbit import OrbitPropagator
from actuators.reaction_wheel import ReactionWheel
from actuators.magnetorquer import Magnetorquer
from actuators.bdot import BDotController
from control.attitude_controller import AttitudeController
from estimation.mekf import MEKF
from estimation.quest import QUEST
from telemetry.logger import Logger
from utils.quaternion import quat_error
from environment.gravity_gradient import GravityGradient
from environment.solar_radiation_pressure import SolarRadiationPressure
from environment.aerodynamic_drag import AerodynamicDrag
from fsw.mode_manager import ModeManager, Mode

# ── Relative Navigation extension ────────────────────────────────────
from environment.cw_dynamics import CWDynamics
from sensors.ranging_sensor import RangingBearingSensor
from estimation.cw_ekf import CWEKF
from control.rendezvous_controller import RendezvousController, RelNavMode

modules_to_clear = [key for key in sys.modules.keys()
                    if 'estimation' in key or 'sensors' in key
                    or 'actuators' in key or 'plant' in key]
for mod in modules_to_clear:
    sys.modules.pop(mod, None)

# =====================================================
# ──  MASTER SWITCH  ──────────────────────────────────
# Set True to run relative navigation / formation sim
# Set False for original single-spacecraft ADCS only
RELATIVE_NAV = True

# Relative nav scenario:
#   'formation'  — hold a fixed trailing position (100 m along-track)
#   'rendezvous' — start in formation, then switch to rendezvous at t=RDV_START_T
#   'both'       — formation hold, then auto-switch to rendezvous
RELNAV_SCENARIO  = 'both'
RDV_START_T      = 1200.0     # s — time to trigger rendezvous maneuver
FORMATION_OFFSET = np.array([0.0, 100.0, 0.0])   # m — initial formation position
# ─────────────────────────────────────────────────────────────────────

# =====================================================
# Hardware Parameters — 3U CubeSat
# =====================================================
I = np.diag([0.030, 0.025, 0.010])   # kg·m²
dt_detumble  = 0.1    # 10 Hz — detumble and sun acquisition
dt_control   = 0.01   # 100 Hz — MEKF + PD fine pointing

TLE_LINE1 = "1 25544U 98067A   25001.50000000  .00006789  00000-0  12345-3 0  9999"
TLE_LINE2 = "2 25544  51.6400 208.9163 0001147  83.8771  11.2433 15.49815689432399"

# =====================================================
# Disturbance environment
# =====================================================
gg   = GravityGradient(I)
srp  = SolarRadiationPressure()
drag = AerodynamicDrag(Cd=2.2, f107=150.0, f107a=150.0, ap=4.0)

# =====================================================
# Spacecraft & Sensors — single set, whole mission
# =====================================================
sc         = Spacecraft(I)
sc.omega   = np.array([0.18, -0.14, 0.22])   # ~18 deg/s post-deployment

orbit      = OrbitPropagator(tle_line1=TLE_LINE1, tle_line2=TLE_LINE2)
mag_field  = MagneticField(epoch_year=2025.0)
sun_model  = SunModel(epoch_year=2025.0)
mag_sens   = Magnetometer()
sun_sens   = SunSensor()
gyro       = Gyro(dt=dt_control, bias_init_max_deg_s=0.05)

rw         = ReactionWheel(h_max=0.004)
mtq        = Magnetorquer(m_max=0.2)
bdot       = BDotController(k_bdot=2e5, m_max=0.2)
controller = AttitudeController(Kp=0.0005, Kd=0.008)
quest_alg  = QUEST()
in_eclipse = False
ekf        = MEKF(dt_control)
ekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0) ** 2

fsw        = ModeManager()
logger     = Logger()

q_ref      = np.array([1., 0., 0., 0.])
t_sim_max  = 3600.0   # 60 min hard cap

# =====================================================
# ── Relative Navigation Setup ────────────────────────
# =====================================================
if RELATIVE_NAV:
    # Chief orbit radius (from TLE — ISS-like ~410 km altitude)
    R_CHIEF_KM = 6371.0 + 410.0

    # CW propagator for deputy relative motion
    cw = CWDynamics(chief_orbit_radius_km=R_CHIEF_KM)

    # Initialise deputy on a Passive Safety Ellipse with along-track offset
    # PSE: 2:1 ellipse (50 m radial × 100 m along-track), trailing at 100 m
    cw.set_initial_offset(
        dr_lvlh_m  = FORMATION_OFFSET,
        dv_lvlh_ms = None   # drift-free velocity
    )

    # Ranging + bearing sensor (range noise 2m, angle noise 0.5°)
    rng_sensor = RangingBearingSensor(
        sigma_range_m    = 2.0,
        sigma_range_frac = 0.005,
        sigma_angle_rad  = np.radians(0.5),
        fov_half_deg     = 60.0,
        max_range_m      = 5000.0,
    )

    # CW-EKF for relative navigation
    cw_ekf = CWEKF(n=cw.n, dt=dt_detumble)

    # Seed EKF with true initial state + noise
    init_noise = np.array([5., 5., 5., 0.05, 0.05, 0.05])
    cw_ekf.initialise(cw.state + init_noise * np.random.randn(6))

    # Rendezvous / formation hold controller
    rel_ctrl = RendezvousController(
        n           = cw.n,
        mode        = RelNavMode.FORMATION_HOLD,
        target_lvlh = FORMATION_OFFSET,
    )

    # Relative nav telemetry
    rn_t, rn_dx, rn_dy, rn_dz        = [], [], [], []
    rn_est_dx, rn_est_dy, rn_est_dz  = [], [], []
    rn_range, rn_est_range            = [], []
    rn_std_x, rn_std_y, rn_std_z     = [], [], []
    rn_dv_total                       = []
    rn_mode                           = []
    rn_rdv_triggered                  = False

    print("  RelNav: deputy initialised at "
          f"[{FORMATION_OFFSET[0]:.0f}, {FORMATION_OFFSET[1]:.0f}, "
          f"{FORMATION_OFFSET[2]:.0f}] m LVLH")
    print(f"  RelNav: scenario = '{RELNAV_SCENARIO}', "
          f"rdv at t={RDV_START_T:.0f}s")

# =====================================================
# Telemetry storage
# =====================================================
tel_t, tel_mode                     = [], []
tel_wx, tel_wy, tel_wz, tel_rate    = [], [], [], []
tel_B, tel_rho                      = [], []
tel_T_aero, tel_T_gg, tel_T_srp    = [], [], []
tel_hx, tel_hy, tel_hz             = [], [], []
tel_err_deg                         = []

print("=" * 60)
print("  3U CubeSat ADCS — State Machine Driven Simulation")
if RELATIVE_NAV:
    print("  + Relative Navigation (Clohessy-Wiltshire)")
print("=" * 60)
print(f"  Initial tumble: {np.degrees(np.linalg.norm(sc.omega)):.1f} deg/s")
print()

# =====================================================
# Main simulation loop
# =====================================================
t = 0.0
triad_err_deg = None
mekf_seeded   = False
last_good_q      = None
last_good_t      = -999.0
gyro_bridge      = False
mekf_seed_t = 0.0

while t < t_sim_max:

    # ── Sensors ──────────────────────────────────────────────────────
    pos, vel   = orbit.step(dt_detumble)
    B_I        = mag_field.get_field(pos)
    B_meas     = mag_sens.measure(sc.q, B_I)
    sun_I      = sun_model.get_sun_vector(t_seconds=t)
    sun_meas   = sun_sens.measure(sc.q, sun_I)
    omega_meas = gyro.measure(sc.omega)

    # ── Disturbances ─────────────────────────────────────────────────
    sun_pos_km  = sun_I * 1.496e8
    T_gg        = gg.compute(pos, sc.q)
    T_srp, nu   = srp.compute(sc.q, sun_I, pos_km=pos, sun_pos_km=sun_pos_km)
    T_aero, rho = drag.compute(sc.q, pos, vel, t_seconds=t)
    disturbance = T_gg + T_srp + T_aero

    # ── QUEST — only during SUN_ACQUISITION ──────────────────────────
    if fsw.is_sun_acquiring:
        in_eclipse = (nu < 0.1)

        nadir_I = QUEST.nadir_inertial(pos)
        nadir_b = QUEST.nadir_body_from_earth_sensor(pos, sc.q)

        if in_eclipse:
            q_quest, quest_quality = quest_alg.compute_multi(
                vectors_body     = [B_meas,  nadir_b],
                vectors_inertial = [B_I,     nadir_I],
                weights          = [0.85,    0.15],
            )
        else:
            q_quest, quest_quality = quest_alg.compute_multi(
                vectors_body     = [B_meas,  sun_meas, nadir_b],
                vectors_inertial = [B_I,     sun_I,    nadir_I],
                weights          = [0.70,    0.20,     0.10],
            )

        if q_quest[0] < 0:
            q_quest = -q_quest

        quest_ok = (quest_quality > 0.01)

        if quest_ok:
            last_good_q = q_quest.copy()
            if last_good_q[0] < 0:
                last_good_q = -last_good_q
            last_good_t = t
            gyro_bridge = False
            triad_err_deg = 5.0
        elif last_good_q is not None and (t - last_good_t) < 120.0:
            omega_corr = omega_meas - ekf.bias
            wx, wy, wz = omega_corr
            Omega = np.array([
                [ 0,   -wx, -wy, -wz],
                [ wx,   0,   wz, -wy],
                [ wy,  -wz,  0,   wx],
                [ wz,   wy, -wx,  0 ]
            ])
            last_good_q += 0.5 * dt_detumble * Omega @ last_good_q
            last_good_q /= np.linalg.norm(last_good_q)
            if last_good_q[0] < 0:
                last_good_q = -last_good_q
            triad_err_deg = 5.0
            gyro_bridge = True
        else:
            triad_err_deg = 180.0

    # ── FSW mode update ───────────────────────────────────────────────
    mode = fsw.update(t, sc.omega, rw.h, triad_err_deg=triad_err_deg)

    # ── Seed MEKF ONCE on transition into FINE_POINTING ──────────────
    if mode == Mode.FINE_POINTING and not mekf_seeded:
        if last_good_q is not None:
            seed_q = last_good_q.copy()
            if seed_q[0] < 0:
                seed_q = -seed_q
            ekf.q = seed_q
            ekf.P[0:3, 0:3] = np.eye(3) * (np.radians(5.0)) ** 2
            print(f"  MEKF seeded from QUEST (5° uncertainty)")
        else:
            ekf.q = sc.q.copy()
            ekf.P[0:3, 0:3] = np.eye(3) * (np.radians(5.0)) ** 2
            print(f"  MEKF seeded from sc.q")
        mekf_seeded = True
        mekf_seed_t = t

    # ── Actuators ─────────────────────────────────────────────────────
    if mode == Mode.SAFE_MODE:
        sc.step(np.zeros(3), disturbance, dt_detumble)

    elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
        m_cmd, _     = bdot.compute(B_meas, sc.omega, B_I, dt_detumble)
        total_torque = mtq.compute_torque(m_cmd, B_meas)
        sc.step(total_torque, disturbance, dt_detumble)

    elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
        q_err_check = quat_error(sc.q, ekf.q)
        if q_err_check[0] < 0:
            q_err_check = -q_err_check
        err_check = np.degrees(2 * np.linalg.norm(q_err_check[1:]))

        if err_check > 25.0 and last_good_q is not None:
            nadir_I  = QUEST.nadir_inertial(pos)
            nadir_b  = QUEST.nadir_body_from_earth_sensor(pos, sc.q)
            q_fresh, q_qual = quest_alg.compute_multi(
                vectors_body     = [B_meas,  sun_meas, nadir_b],
                vectors_inertial = [B_I,     sun_I,    nadir_I],
                weights          = [0.70,    0.20,     0.10],
            )
            if q_fresh[0] < 0:
                q_fresh = -q_fresh
            ekf.q = q_fresh.copy()

        n_inner = int(dt_detumble / dt_control)
        for _ in range(n_inner):
            omega_meas_inner = gyro.measure(sc.omega)
            ekf.predict(omega_meas_inner)
            ekf.update_vector(B_meas, B_I, ekf.R_mag)
            ekf.update_vector(sun_meas, sun_I, ekf.R_sun)

            omega_est     = sc.omega - ekf.bias
            torque_cmd, _ = controller.compute(ekf.q, omega_est, q_ref)
            torque_rw     = rw.apply_torque(torque_cmd, dt_control)

            m_cmd = mtq.compute_dipole(rw.h, B_meas)
            if mode == Mode.MOMENTUM_DUMP:
                m_cmd = np.clip(m_cmd * 5.0, -mtq.m_max, mtq.m_max)
            torque_mtq = mtq.compute_torque(m_cmd, B_meas)

            sc.step(torque_mtq + torque_rw, disturbance, dt_control,
                    tau_rw=np.zeros(3), h_rw=rw.h.copy())

        q_err_vec = quat_error(sc.q, ekf.q)
        if q_err_vec[0] < 0:
            q_err_vec = -q_err_vec
        err_deg = np.degrees(2 * np.linalg.norm(q_err_vec[1:]))
        tel_err_deg.append((t, err_deg))

        if int(t) % 200 == 0:
            bias_deg = np.degrees(np.linalg.norm(ekf.bias))
            P_att    = np.sqrt(np.diag(ekf.P[0:3, 0:3]))
            P_bias   = np.sqrt(np.diag(ekf.P[3:6, 3:6]))
            print(f"  t={t:.0f} err={err_deg:.2f}° "
                  f"bias={bias_deg:.4f}°/s "
                  f"P_att={np.degrees(P_att.mean()):.3f}° "
                  f"P_bias={np.degrees(P_bias.mean()):.4f}°/s")

    # ── RELATIVE NAVIGATION ───────────────────────────────────────────
    if RELATIVE_NAV:
        # ── Scenario mode switching ───────────────────────────────────
        if (RELNAV_SCENARIO in ('rendezvous', 'both')
                and not rn_rdv_triggered
                and t >= RDV_START_T):
            rel_ctrl.set_mode(RelNavMode.RENDEZVOUS, t=t)
            rn_rdv_triggered = True

        # ── CW-EKF predict ────────────────────────────────────────────
        accel_cmd, impulse_dv = rel_ctrl.compute(cw_ekf.x, t)
        cw_ekf.predict(accel_cmd)

        # ── Apply impulse to true deputy (if burn due) ────────────────
        if impulse_dv is not None:
            cw.apply_impulse(impulse_dv)
            total_dv_log = cw.total_dv_ms
            print(f"  RelNav [{t:.0f}s] Burn: Δv="
                  f"[{impulse_dv[0]:.3f}, {impulse_dv[1]:.3f}, "
                  f"{impulse_dv[2]:.3f}] m/s  "
                  f"ΣΔv={total_dv_log:.3f} m/s")

        # ── Propagate true deputy dynamics ────────────────────────────
        true_state = cw.step(dt_detumble, accel_cmd)

        # ── Sensor measurement ────────────────────────────────────────
        # Sensor pointing: along-track (+y in LVLH) toward trailing deputy
        z_meas, R_meas = rng_sensor.measure(
            dr_lvlh              = true_state[0:3],
            sensor_pointing_lvlh = np.array([0., 1., 0.])
        )

        # ── EKF update if measurement available ──────────────────────
        if z_meas is not None:
            cw_ekf.update(z_meas, R_meas)

        # ── Log relative nav telemetry ────────────────────────────────
        rn_t.append(t)
        rn_dx.append(true_state[0])
        rn_dy.append(true_state[1])
        rn_dz.append(true_state[2])
        rn_est_dx.append(cw_ekf.x[0])
        rn_est_dy.append(cw_ekf.x[1])
        rn_est_dz.append(cw_ekf.x[2])
        rn_range.append(cw.range_m)
        rn_est_range.append(np.linalg.norm(cw_ekf.position))
        rn_std_x.append(cw_ekf.position_std[0])
        rn_std_y.append(cw_ekf.position_std[1])
        rn_std_z.append(cw_ekf.position_std[2])
        rn_dv_total.append(cw.total_dv_ms)
        rn_mode.append(rel_ctrl.mode.value)

    # ── Chief telemetry ────────────────────────────────────────────────
    tel_t.append(t)
    tel_mode.append(mode.value)
    tel_wx.append(np.degrees(sc.omega[0]))
    tel_wy.append(np.degrees(sc.omega[1]))
    tel_wz.append(np.degrees(sc.omega[2]))
    tel_rate.append(np.degrees(np.linalg.norm(sc.omega)))
    tel_B.append(np.linalg.norm(B_I) * 1e9)
    tel_rho.append(rho)
    tel_T_aero.append(np.linalg.norm(T_aero) * 1e9)
    tel_T_gg.append(np.linalg.norm(T_gg) * 1e9)
    tel_T_srp.append(np.linalg.norm(T_srp) * 1e9)
    tel_hx.append(rw.h[0] * 1000)
    tel_hy.append(rw.h[1] * 1000)
    tel_hz.append(rw.h[2] * 1000)

    t += dt_detumble

print(f"t={t:.0f} P_cross={ekf.P[0:3, 3:6]}")

# =====================================================
# Summary
# =====================================================
print()
print("=" * 60)
print("  Simulation complete")
print("=" * 60)
print(f"  Total sim time : {t:.1f}s ({t/60:.1f} min)")
print(f"  Mode history:")
for t_trans, m in fsw.mode_history:
    print(f"    t={t_trans:7.1f}s  →  {m.name}")
print()
print(f"  Final disturbance means:")
print(f"    |T_aero| : {np.mean(tel_T_aero):.3f} nN·m")
print(f"    |T_gg|   : {np.mean(tel_T_gg):.3f} nN·m")
print(f"    |T_srp|  : {np.mean(tel_T_srp):.3f} nN·m")
if tel_err_deg:
    errs = [e for _, e in tel_err_deg]
    print(f"  Estimation error (fine pointing): "
          f"mean={np.mean(errs):.3f}°, max={np.max(errs):.3f}°")
if RELATIVE_NAV:
    print(f"  Relative Nav:")
    print(f"    Final range     : {rn_range[-1]:.2f} m")
    print(f"    Total delta-V   : {cw.total_dv_ms:.4f} m/s")
    est_err = np.sqrt((np.array(rn_dx)-np.array(rn_est_dx))**2
                    + (np.array(rn_dy)-np.array(rn_est_dy))**2
                    + (np.array(rn_dz)-np.array(rn_est_dz))**2)
    print(f"    CW-EKF pos err  : mean={np.mean(est_err):.2f} m, "
          f"max={np.max(est_err):.2f} m")

# =====================================================
# Plots — Chief ADCS
# =====================================================
plt.rcParams.update({"font.size": 11, "axes.grid": True,
                      "grid.alpha": 0.35, "lines.linewidth": 1.2})

MODE_COLORS = {
    Mode.SAFE_MODE.value:       ('red',       'SAFE'),
    Mode.DETUMBLE.value:        ('royalblue', 'DETUMBLE'),
    Mode.SUN_ACQUISITION.value: ('orange',    'SUN ACQ'),
    Mode.FINE_POINTING.value:   ('green',     'FINE POINT'),
    Mode.MOMENTUM_DUMP.value:   ('purple',    'MTM DUMP'),
}

def add_mode_bands(ax, t_arr, mode_arr):
    t_arr    = np.array(t_arr)
    mode_arr = np.array(mode_arr)
    changes  = np.where(np.diff(mode_arr))[0]
    segments = np.concatenate([[0], changes + 1, [len(mode_arr)]])
    for i in range(len(segments) - 1):
        s, e = segments[i], segments[i+1]
        m    = mode_arr[s]
        col, _ = MODE_COLORS.get(m, ('gray', '?'))
        ax.axvspan(t_arr[s], t_arr[e-1], alpha=0.08, color=col, linewidth=0)

fig1, axes1 = plt.subplots(2, 3, figsize=(18, 9))
fig1.suptitle("3U CubeSat ADCS — Full Mission (Chief)",
              fontsize=13, fontweight='bold')

t_arr = np.array(tel_t)

ax = axes1[0, 0]
ax.plot(t_arr, tel_wx, label="ωx", color='royalblue', alpha=0.8)
ax.plot(t_arr, tel_wy, label="ωy", color='darkorange', alpha=0.8)
ax.plot(t_arr, tel_wz, label="ωz", color='green', alpha=0.8)
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Angular Rate [deg/s]")
ax.set_title("Angular Rate Components"); ax.legend(fontsize=8)

ax = axes1[0, 1]
ax.plot(t_arr, tel_rate, color='purple')
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Total Rate [deg/s]")
ax.set_title("Total Angular Rate")

ax = axes1[0, 2]
ax.semilogy(t_arr, tel_T_aero, label="Aero drag",    color='saddlebrown')
ax.semilogy(t_arr, tel_T_gg,   label="Gravity grad", color='steelblue')
ax.semilogy(t_arr, tel_T_srp,  label="SRP",          color='goldenrod')
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("|Torque| [nN·m]")
ax.set_title("Disturbance Torques"); ax.legend(fontsize=8)

ax = axes1[1, 0]
ax.plot(t_arr, tel_hx, label="h_x", color='royalblue')
ax.plot(t_arr, tel_hy, label="h_y", color='darkorange')
ax.plot(t_arr, tel_hz, label="h_z", color='green')
ax.axhline( 4.0, color='red', linestyle=':', linewidth=1, label="±h_max")
ax.axhline(-4.0, color='red', linestyle=':', linewidth=1)
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Momentum [mN·m·s]")
ax.set_title("Reaction Wheel Momentum"); ax.legend(fontsize=8)

ax = axes1[1, 1]
ax.semilogy(t_arr, tel_rho, color='saddlebrown')
add_mode_bands(ax, tel_t, tel_mode)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Density [kg/m³]")
ax.set_title("Atmospheric Density (NRLMSISE-00)")

ax = axes1[1, 2]
mode_arr = np.array(tel_mode)
ax.step(t_arr, mode_arr, color='black', linewidth=1.5)
for val, (col, label) in MODE_COLORS.items():
    mask = mode_arr == val
    if mask.any():
        ax.fill_between(t_arr, val-0.4, val+0.4,
                        where=mask, alpha=0.4, color=col, label=label)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Mode")
ax.set_yticks([m.value for m in Mode])
ax.set_yticklabels([m.name for m in Mode], fontsize=7)
ax.set_title("FSW Mode Timeline"); ax.legend(fontsize=7, loc='right')

fig1.tight_layout()

if tel_err_deg:
    t_err  = [x[0] for x in tel_err_deg]
    e_err  = [x[1] for x in tel_err_deg]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_err, e_err, color='crimson', label="Attitude estimation error")
    ax2.axhline(0.5, color='gray', linestyle=':', label="0.5° ref")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Error [deg]")
    ax2.set_title("MEKF Attitude Estimation Error — Fine Pointing Phase")
    ax2.legend(fontsize=9)
    fig2.tight_layout()

# =====================================================
# Plots — Relative Navigation
# =====================================================
if RELATIVE_NAV and rn_t:
    rn_t_arr     = np.array(rn_t)
    rn_dx_arr    = np.array(rn_dx)
    rn_dy_arr    = np.array(rn_dy)
    rn_dz_arr    = np.array(rn_dz)
    rn_edx_arr   = np.array(rn_est_dx)
    rn_edy_arr   = np.array(rn_est_dy)
    rn_edz_arr   = np.array(rn_est_dz)
    rn_std_x_arr = np.array(rn_std_x)
    rn_std_y_arr = np.array(rn_std_y)
    rn_std_z_arr = np.array(rn_std_z)
    pos_err_arr  = np.sqrt((rn_dx_arr - rn_edx_arr)**2
                         + (rn_dy_arr - rn_edy_arr)**2
                         + (rn_dz_arr - rn_edz_arr)**2)

    RELNAV_COLORS = {
        RelNavMode.FORMATION_HOLD.value:  ('steelblue', 'FORM HOLD'),
        RelNavMode.RENDEZVOUS.value:      ('crimson',   'RENDEZVOUS'),
        RelNavMode.STATION_KEEPING.value: ('green',     'STN KEEP'),
        RelNavMode.COASTING.value:        ('gray',      'COASTING'),
    }

    def add_rn_bands(ax):
        rn_mode_arr = np.array(rn_mode)
        changes  = np.where(np.diff(rn_mode_arr))[0]
        segments = np.concatenate([[0], changes + 1, [len(rn_mode_arr)]])
        for i in range(len(segments) - 1):
            s, e = segments[i], segments[i+1]
            m    = rn_mode_arr[s]
            col, lbl = RELNAV_COLORS.get(m, ('gray', '?'))
            ax.axvspan(rn_t_arr[s], rn_t_arr[e-1],
                       alpha=0.10, color=col, linewidth=0)

    fig3, axes3 = plt.subplots(2, 3, figsize=(18, 9))
    fig3.suptitle("Relative Navigation — CW Dynamics + EKF",
                  fontsize=13, fontweight='bold')

    # ── Relative position components ─────────────────────────────────
    ax = axes3[0, 0]
    ax.plot(rn_t_arr, rn_dx_arr,  color='royalblue',  label="δx true (radial)")
    ax.plot(rn_t_arr, rn_edx_arr, color='royalblue',
            linestyle='--', alpha=0.6, label="δx est")
    ax.fill_between(rn_t_arr,
                    rn_edx_arr - 3*rn_std_x_arr,
                    rn_edx_arr + 3*rn_std_x_arr,
                    alpha=0.15, color='royalblue', label="3σ")
    add_rn_bands(ax)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("δx [m]")
    ax.set_title("Radial (δx)"); ax.legend(fontsize=8)

    ax = axes3[0, 1]
    ax.plot(rn_t_arr, rn_dy_arr,  color='darkorange', label="δy true (along-track)")
    ax.plot(rn_t_arr, rn_edy_arr, color='darkorange',
            linestyle='--', alpha=0.6, label="δy est")
    ax.fill_between(rn_t_arr,
                    rn_edy_arr - 3*rn_std_y_arr,
                    rn_edy_arr + 3*rn_std_y_arr,
                    alpha=0.15, color='darkorange', label="3σ")
    add_rn_bands(ax)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("δy [m]")
    ax.set_title("Along-Track (δy)"); ax.legend(fontsize=8)

    ax = axes3[0, 2]
    ax.plot(rn_t_arr, rn_dz_arr,  color='green', label="δz true (cross-track)")
    ax.plot(rn_t_arr, rn_edz_arr, color='green',
            linestyle='--', alpha=0.6, label="δz est")
    ax.fill_between(rn_t_arr,
                    rn_edz_arr - 3*rn_std_z_arr,
                    rn_edz_arr + 3*rn_std_z_arr,
                    alpha=0.15, color='green', label="3σ")
    add_rn_bands(ax)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("δz [m]")
    ax.set_title("Cross-Track (δz)"); ax.legend(fontsize=8)

    # ── Range ─────────────────────────────────────────────────────────
    ax = axes3[1, 0]
    ax.plot(rn_t_arr, np.array(rn_range),     color='purple', label="True range")
    ax.plot(rn_t_arr, np.array(rn_est_range), color='purple',
            linestyle='--', alpha=0.6, label="EKF range est")
    if RDV_START_T < t_sim_max:
        ax.axvline(RDV_START_T, color='crimson', linestyle=':',
                   linewidth=1.5, label=f"Rendezvous starts")
    add_rn_bands(ax)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Range [m]")
    ax.set_title("Inter-Spacecraft Range"); ax.legend(fontsize=8)

    # ── EKF estimation error ──────────────────────────────────────────
    ax = axes3[1, 1]
    ax.plot(rn_t_arr, pos_err_arr, color='crimson', label="|pos error|")
    ax.axhline(10.0, color='gray', linestyle=':', label="10 m ref")
    add_rn_bands(ax)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position Error [m]")
    ax.set_title("CW-EKF Position Estimation Error"); ax.legend(fontsize=8)

    # ── Relative trajectory (δx vs δy — LVLH plane) ──────────────────
    ax = axes3[1, 2]
    sc_traj = ax.scatter(rn_dy_arr, rn_dx_arr,
                         c=rn_t_arr, cmap='viridis', s=3, zorder=3)
    ax.plot(0, 0, 'k*', markersize=14, label="Chief", zorder=5)
    ax.plot(rn_dy_arr[0], rn_dx_arr[0], 'go', markersize=8,
            label="Deputy start", zorder=4)
    ax.plot(rn_dy_arr[-1], rn_dx_arr[-1], 'rs', markersize=8,
            label="Deputy end", zorder=4)
    plt.colorbar(sc_traj, ax=ax, label="Time [s]")
    ax.set_xlabel("δy — Along-Track [m]")
    ax.set_ylabel("δx — Radial [m]")
    ax.set_title("LVLH Relative Trajectory"); ax.legend(fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')

    # ── Legend for mode bands ─────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, alpha=0.4, label=l)
                      for _, (c, l) in RELNAV_COLORS.items()]
    fig3.legend(handles=legend_patches, loc='lower center',
                ncol=4, fontsize=9, title="RelNav Mode")

    fig3.tight_layout(rect=[0, 0.04, 1, 1])

plt.show()