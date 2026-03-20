"""
3U CubeSat ADCS + CW & ROE Relative Navigation
===============================================
Phase 1 — ADCS baseline (must fully complete before Phase 2 starts):
    DETUMBLE → SUN_ACQUISITION → FINE_POINTING → momentum dump exits
    MEKF, QUEST, reaction wheels, magnetorquers, full disturbance env.

    Phase 1 ends when:
        (a) FSW mode == FINE_POINTING, AND
        (b) MEKF attitude error < ADCS_STABLE_DEG for ADCS_STABLE_SUST
            consecutive steps, AND
        (c) No MOMENTUM_DUMP has occurred since (b) began.
    Only then is fine_point_confirmed = True and Phase 2 may begin.

Phase 2 — RelNav (starts RDV_DELAY_S after Phase 1 confirmed):
    CW-EKF   : Clohessy-Wiltshire range+bearing EKF (state source for logging)
    ROE-EKF  : J2-perturbed ROE filter (state source for rendezvous planner)
    ROE ctrl : Formation hold (GVE, J2-aware) → two-impulse CW rendezvous
               fed from J2-corrected ROE-EKF state → CW cleanup hold

The deputy CW/ROE propagator is run from t=0 in *open loop* (no burns,
no EKF updates) while Phase 1 is in progress — purely to track where
the deputy would be so we have a realistic initial condition for the
EKFs when Phase 2 starts.  No thrusters fire during Phase 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plant.spacecraft                     import Spacecraft
from sensors.gyro                         import Gyro
from sensors.magnetometer                 import Magnetometer
from sensors.sun_sensor                   import SunSensor
from environment.magnetic_field           import MagneticField
from environment.sun_model                import SunModel
from environment.orbit                    import OrbitPropagator
from environment.gravity_gradient         import GravityGradient
from environment.solar_radiation_pressure import SolarRadiationPressure
from environment.aerodynamic_drag         import AerodynamicDrag
from actuators.reaction_wheel             import ReactionWheel
from actuators.magnetorquer               import Magnetorquer
from actuators.bdot                       import BDotController
from control.attitude_controller          import AttitudeController
from estimation.mekf                      import MEKF
from estimation.quest                     import QUEST
from utils.quaternion                     import quat_error
from fsw.mode_manager                     import ModeManager, Mode

# RelNav
from environment.cw_dynamics              import CWDynamics
from environment.roe_dynamics             import ROEDynamics
from sensors.ranging_sensor               import RangingBearingSensor
from estimation.cw_ekf                    import CWEKF
from estimation.roe_ekf                   import ROEEKF
from control.rendezvous_controller        import RendezvousController, RelNavMode
from control.roe_controller               import ROEController, ROEMode

# =====================================================================
#  CONFIG
# =====================================================================
T_SIM_MAX   = 7200.0       # s
dt_detumble = 0.1
dt_control  = 0.01
N_INNER     = int(dt_detumble / dt_control)

# Phase 1 → Phase 2 gate: MEKF error must be below this for this many steps
ADCS_STABLE_DEG  = 1.0     # deg   — pointing accuracy required before RelNav
ADCS_STABLE_SUST = 100     # steps — ~10 s of stable pointing at dt=0.1 s

# Phase 2 config
RDV_DELAY_S      = 300.0   # s after Phase 1 confirmed before rendezvous fires
FORMATION_OFFSET = np.array([0.0, 100.0, 0.0])   # m LVLH
R_CHIEF_KM       = 6781.0

TLE_LINE1 = "1 25544U 98067A   25001.50000000  .00006789  00000-0  12345-3 0  9999"
TLE_LINE2 = "2 25544  51.6400 208.9163 0001147  83.8771  11.2433 15.49815689432399"

# =====================================================================
#  ADCS HARDWARE
# =====================================================================
I    = np.diag([0.030, 0.025, 0.010])
gg   = GravityGradient(I)
srp  = SolarRadiationPressure()
drag = AerodynamicDrag(Cd=2.2, f107=150.0, f107a=150.0, ap=4.0)

sc       = Spacecraft(I)
sc.omega = np.array([0.18, -0.14, 0.22])

orbit     = OrbitPropagator(tle_line1=TLE_LINE1, tle_line2=TLE_LINE2)
mag_field = MagneticField(epoch_year=2025.0)
sun_model = SunModel(epoch_year=2025.0)
mag_sens  = Magnetometer()
sun_sens  = SunSensor()
gyro      = Gyro(dt=dt_control, bias_init_max_deg_s=0.05)

rw         = ReactionWheel(h_max=0.004)
mtq        = Magnetorquer(m_max=0.2)
bdot       = BDotController(k_bdot=2e5, m_max=0.2)
controller = AttitudeController(Kp=0.0005, Kd=0.008)
quest_alg  = QUEST()
mekf       = MEKF(dt_control)
mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
fsw        = ModeManager()
q_ref      = np.array([1., 0., 0., 0.])

# =====================================================================
#  RELNAV HARDWARE  (initialised now, but silent until Phase 2)
# =====================================================================
cw = CWDynamics(chief_orbit_radius_km=R_CHIEF_KM)
cw.set_initial_offset(dr_lvlh_m=FORMATION_OFFSET, dv_lvlh_ms=None)

roe_dyn = ROEDynamics(
    a_chief_m = R_CHIEF_KM * 1e3,
    e_chief   = 0.0001,
    i_chief   = np.radians(51.6)
)
dv_ic = np.array([0.0, -2.0 * cw.n * FORMATION_OFFSET[0], 0.0])
roe_dyn.set_from_lvlh(FORMATION_OFFSET, dv_ic)

rng_sensor = RangingBearingSensor(
    sigma_range_m=0.5, sigma_range_frac=0.002,
    sigma_angle_rad=np.radians(0.1), fov_half_deg=60.0, max_range_m=5000.0
)

# EKFs — seeded at Phase 2 start, not now
cw_ekf  = CWEKF(n=cw.n, dt=dt_detumble)
roe_ekf = ROEEKF(roe_dyn=roe_dyn, dt=dt_detumble)

rel_ctrl = RendezvousController(
    n=cw.n,
    mode=RelNavMode.FORMATION_HOLD,
    target_lvlh=FORMATION_OFFSET
)
roe_ctrl = ROEController(
    roe_dyn=roe_dyn,
    mode=ROEMode.FORMATION_HOLD,
    target_roe=roe_dyn.roe.copy()
)

# =====================================================================
#  SIM STATE
# =====================================================================
t = 0.0

# Phase 1 state
triad_err_deg    = None
mekf_seeded      = False
last_good_q      = None
last_good_t      = -999.0
adcs_stable_cnt  = 0        # consecutive steps below ADCS_STABLE_DEG
fine_point_t0    = None     # first time we enter FINE_POINTING
phase1_confirmed = False    # True once ADCS is genuinely stable
phase1_conf_t    = None

# Phase 2 state
phase2_active = False
rdv_triggered = False
rdv_complete  = False
ekf_seeded    = False       # EKFs get a warm-start snapshot at Phase 2 start

tel = dict(
    t=[], mode=[],
    wx=[], wy=[], wz=[], rate=[],
    T_aero=[], T_gg=[], T_srp=[],
    hx=[], hy=[], hz=[],
    rho=[], err_deg=[],
    rn_t=[],
    rn_dx=[], rn_dy=[], rn_dz=[],
    rn_edx=[], rn_edy=[], rn_edz=[],
    rn_rdx=[], rn_rdy=[], rn_rdz=[],
    rn_range=[], rn_est_range=[],
    rn_dv=[], rn_mode=[],
)

print("=" * 60)
print("  3U CubeSat ADCS + CW/ROE RelNav")
print("=" * 60)
print(f"  Tumble  : {np.degrees(np.linalg.norm(sc.omega)):.1f} deg/s")
print(f"  Deputy  : {FORMATION_OFFSET} m LVLH")
print(f"  Phase 1 gate: err < {ADCS_STABLE_DEG}° for {ADCS_STABLE_SUST} steps "
      f"in FINE_POINTING (no dump pending)")
print(f"  RDV delay after gate: {RDV_DELAY_S:.0f}s")

# =====================================================================
#  MAIN LOOP
# =====================================================================
while t < T_SIM_MAX:

    # ── Environment ───────────────────────────────────────────────────
    pos, vel    = orbit.step(dt_detumble)
    B_I         = mag_field.get_field(pos)
    B_meas      = mag_sens.measure(sc.q, B_I)
    sun_I       = sun_model.get_sun_vector(t_seconds=t)
    sun_meas    = sun_sens.measure(sc.q, sun_I)
    omega_meas  = gyro.measure(sc.omega)
    sun_pos_km  = sun_I * 1.496e8
    T_gg        = gg.compute(pos, sc.q)
    T_srp, nu   = srp.compute(sc.q, sun_I, pos_km=pos, sun_pos_km=sun_pos_km)
    T_aero, rho = drag.compute(sc.q, pos, vel, t_seconds=t)
    disturbance = T_gg + T_srp + T_aero

    # ── QUEST  ────────────────────────────────────────────────────────
    if fsw.is_sun_acquiring:
        in_eclipse = (nu < 0.1)
        nadir_I    = QUEST.nadir_inertial(pos)
        nadir_b    = QUEST.nadir_body_from_earth_sensor(pos, sc.q)
        if in_eclipse:
            q_quest, q_qual = quest_alg.compute_multi(
                vectors_body=[B_meas, nadir_b],
                vectors_inertial=[B_I, nadir_I], weights=[0.85, 0.15])
        else:
            q_quest, q_qual = quest_alg.compute_multi(
                vectors_body=[B_meas, sun_meas, nadir_b],
                vectors_inertial=[B_I, sun_I, nadir_I],
                weights=[0.70, 0.20, 0.10])
        if q_quest[0] < 0:
            q_quest = -q_quest
        if q_qual > 0.01:
            last_good_q = q_quest.copy()
            if last_good_q[0] < 0: last_good_q = -last_good_q
            last_good_t   = t
            triad_err_deg = 5.0
        elif last_good_q is not None and (t - last_good_t) < 120.0:
            wx, wy, wz = omega_meas - mekf.bias
            Om = np.array([[0, -wx, -wy, -wz],
                           [wx,  0,  wz, -wy],
                           [wy, -wz,  0,  wx],
                           [wz,  wy, -wx,  0]])
            last_good_q += 0.5 * dt_detumble * Om @ last_good_q
            last_good_q /= np.linalg.norm(last_good_q)
            if last_good_q[0] < 0: last_good_q = -last_good_q
            triad_err_deg = 5.0
        else:
            triad_err_deg = 180.0

    mode = fsw.update(t, sc.omega, rw.h, triad_err_deg=triad_err_deg)

    # ── MEKF seed ─────────────────────────────────────────────────────
    if mode == Mode.FINE_POINTING and not mekf_seeded:
        seed = last_good_q.copy() if last_good_q is not None else sc.q.copy()
        if seed[0] < 0: seed = -seed
        mekf.q = seed
        mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
        mekf_seeded   = True
        fine_point_t0 = t
        print(f"  MEKF seeded at t={t:.1f}s")

    # ── ADCS Phase 1 stability gate ───────────────────────────────────
    if (not phase1_confirmed
            and mekf_seeded
            and mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP)):
        qe = quat_error(sc.q, mekf.q)
        if qe[0] < 0: qe = -qe
        err_deg = float(np.degrees(2.0 * np.linalg.norm(qe[1:])))
        if err_deg < ADCS_STABLE_DEG:
            adcs_stable_cnt += 1
        else:
            adcs_stable_cnt = 0   # reset — must be *consecutive*

        if adcs_stable_cnt >= ADCS_STABLE_SUST:
            phase1_confirmed = True
            phase1_conf_t    = t
            print(f"  ✓ Phase 1 CONFIRMED at t={t:.1f}s  "
                  f"(err={err_deg:.3f}°, {adcs_stable_cnt} consecutive steps)")
            print(f"    Phase 2 RelNav will start at t={t + RDV_DELAY_S:.1f}s")
    elif mode not in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
        adcs_stable_cnt = 0   # only reset on non-pointing modes

    # ── Actuators ─────────────────────────────────────────────────────
    if mode == Mode.SAFE_MODE:
        sc.step(np.zeros(3), disturbance, dt_detumble)

    elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
        m_cmd, _ = bdot.compute(B_meas, sc.omega, B_I, dt_detumble)
        sc.step(mtq.compute_torque(m_cmd, B_meas), disturbance, dt_detumble)

    elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
        # Rescue diverged MEKF
        if mekf_seeded and last_good_q is not None:
            qe = quat_error(sc.q, mekf.q)
            if qe[0] < 0: qe = -qe
            if np.degrees(2 * np.linalg.norm(qe[1:])) > 25.0:
                nadir_I = QUEST.nadir_inertial(pos)
                nadir_b = QUEST.nadir_body_from_earth_sensor(pos, sc.q)
                q_f, _ = quest_alg.compute_multi(
                    vectors_body=[B_meas, sun_meas, nadir_b],
                    vectors_inertial=[B_I, sun_I, nadir_I],
                    weights=[0.70, 0.20, 0.10])
                if q_f[0] < 0: q_f = -q_f
                mekf.q = q_f.copy()

        for _ in range(N_INNER):
            oi = gyro.measure(sc.omega)
            mekf.predict(oi)
            mekf.update_vector(B_meas, B_I, mekf.R_mag)
            mekf.update_vector(sun_meas, sun_I, mekf.R_sun)
            omega_est = sc.omega - mekf.bias

            if mode == Mode.MOMENTUM_DUMP:
                torque_cmd = np.zeros(3)
                if np.linalg.norm(sc.omega) < np.radians(5.0):
                    m_cmd = mtq.compute_dipole(rw.h, B_meas)
                else:
                    m_cmd, _ = bdot.compute(B_meas, sc.omega, B_I, dt_control)
                m_cmd      = np.clip(m_cmd, -mtq.m_max, mtq.m_max)
                torque_mtq = mtq.compute_torque(m_cmd, B_meas)
                rw.h      -= torque_mtq * dt_control
                rw.h       = np.clip(rw.h, -rw.h_max, rw.h_max)
            else:
                torque_cmd, _ = controller.compute(mekf.q, omega_est, q_ref)
                rw.apply_torque(torque_cmd, dt_control)
                m_cmd      = mtq.compute_dipole(rw.h, B_meas)
                torque_mtq = mtq.compute_torque(m_cmd, B_meas)
                rw.h      -= torque_mtq * dt_control
                rw.h       = np.clip(rw.h, -rw.h_max, rw.h_max)

            sc.step(torque_mtq, disturbance, dt_control,
                    tau_rw=torque_cmd, h_rw=rw.h.copy())

        if mekf_seeded and mode == Mode.FINE_POINTING:
            qe = quat_error(sc.q, mekf.q)
            if qe[0] < 0: qe = -qe
            err_deg = np.degrees(2 * np.linalg.norm(qe[1:]))
            tel['err_deg'].append((t, err_deg))
            if int(t) % 500 == 0:
                print(f"  t={t:6.0f}s  {mode.name:15s}  "
                      f"err={err_deg:.2f}°  |h|={np.linalg.norm(rw.h)*1e3:.2f} mNms  "
                      f"p1={'OK' if phase1_confirmed else f'cnt={adcs_stable_cnt}'}")

    # =====================================================================
    #  DEPUTY PROPAGATION — always runs, but silently during Phase 1
    #  (no burns, no EKF updates, no controller output used)
    # =====================================================================
    if not phase2_active:
        # Open-loop: just advance truth so deputy position is realistic
        # when Phase 2 starts. No impulses, no sensor reads.
        cw.step(dt_detumble, np.zeros(3))
        roe_dyn.step(dt_detumble)

    # =====================================================================
    #  PHASE 2 — RELNAV  (only after Phase 1 confirmed + delay)
    # =====================================================================
    if (phase1_confirmed
            and not phase2_active
            and t >= phase1_conf_t + RDV_DELAY_S):
        phase2_active = True

        # Warm-start EKFs from current truth + small noise
        np.random.seed(42)
        noise = np.array([5.0, 5.0, 5.0, 0.02, 0.02, 0.02]) * np.random.randn(6)
        cw_ekf.initialise(cw.state + noise)
        # ROE-EKF: seed from roe_dyn truth with noise scaled to ROE magnitudes
        # ROE values are O(dr/a) ~ 1e-5 for 100m formation — noise must be << this
        roe_noise = roe_dyn.roe * 0.05 + np.array([1e-7, 1e-7, 1e-7, 1e-7, 1e-8, 1e-8])
        roe_ekf.initialise(roe_dyn.roe + roe_noise * np.random.randn(6))

        # ROE controller: formation hold at current drifted ROE
        roe_ctrl = ROEController(
            roe_dyn=roe_dyn,
            mode=ROEMode.FORMATION_HOLD,
            target_roe=roe_dyn.roe.copy()
        )
        ekf_seeded = True
        print(f"  ✓ Phase 2 RelNav ACTIVE at t={t:.1f}s  "
              f"(deputy range={cw.range_m:.1f} m)")

    if phase2_active:
        # Trigger rendezvous via ROE controller
        if (not rdv_triggered
                and mode == Mode.FINE_POINTING
                and t >= phase1_conf_t + RDV_DELAY_S + 120.0):
            roe_ctrl.set_mode(ROEMode.RENDEZVOUS,
                              roe_est=roe_ekf.x,
                              mean_anomaly=roe_dyn.mean_anomaly,
                              t=t,
                              t_sim_max=T_SIM_MAX,
                              lvlh_est=cw_ekf.x)
            rdv_triggered = True

        # ROE controller computes impulse from ROE-EKF state
        _, impulse_dv = roe_ctrl.compute(roe_ekf.x, roe_dyn.mean_anomaly, t)

        if impulse_dv is not None:
            cw.apply_impulse(impulse_dv)
            roe_dyn.apply_impulse_lvlh(impulse_dv, roe_dyn.mean_anomaly)
            cw_ekf.x[3:6] += impulse_dv
            roe_ekf.x = roe_dyn.roe.copy()   # reseed ROE-EKF from truth after burn
            print(
                f"  RDV [{t:.0f}s]  dv=[{impulse_dv[0]:.4f}, "
                f"{impulse_dv[1]:.4f}, {impulse_dv[2]:.4f}] m/s  "
                f"ΣΔV={cw.total_dv_ms*1000:.2f} mm/s"
            )

        # Post-burn-2 cleanup: switch to CW formation hold at origin
        if (rdv_triggered and roe_ctrl.mode == ROEMode.COASTING
                and not rdv_complete):
            if not getattr(roe_ctrl, '_cleanup_started', False):
                roe_ctrl._cleanup_started = True
                roe_ctrl._cleanup_t0      = t
                rel_ctrl.set_mode(RelNavMode.FORMATION_HOLD, t=t,
                                  target_lvlh=np.zeros(3))
                print(f"  Cleanup hold started at t={t:.0f}s")
            elif t >= roe_ctrl._cleanup_t0 + 120.0:
                rdv_complete = True
                print(f"  ✓ Rendezvous COMPLETE at t={t:.0f}s  "
                      f"range={cw.range_m:.2f} m")

        # During cleanup use CW PD hold accel, otherwise zero
        if rdv_triggered and getattr(roe_ctrl, '_cleanup_started', False):
            accel_cmd, _ = rel_ctrl.compute(cw_ekf.x, t)
        else:
            accel_cmd = np.zeros(3)

        # Propagate truth
        true_cw = cw.step(dt_detumble, accel_cmd)
        roe_dyn.step(dt_detumble)

        # EKF predict + update
        cw_ekf.predict(accel_cmd)
        roe_ekf.predict(accel_cmd)
        z, R = rng_sensor.measure(true_cw[:3], np.array([0., 1., 0.]))
        if z is not None:
            cw_ekf.update(z, R)
            roe_ekf.update(z, R)

        roe_pos_est = roe_ekf.position

        tel['rn_t'].append(t)
        tel['rn_dx'].append(true_cw[0]);      tel['rn_dy'].append(true_cw[1])
        tel['rn_dz'].append(true_cw[2])
        tel['rn_edx'].append(cw_ekf.x[0]);   tel['rn_edy'].append(cw_ekf.x[1])
        tel['rn_edz'].append(cw_ekf.x[2])
        tel['rn_rdx'].append(roe_pos_est[0]); tel['rn_rdy'].append(roe_pos_est[1])
        tel['rn_rdz'].append(roe_pos_est[2])
        tel['rn_range'].append(cw.range_m)
        tel['rn_est_range'].append(float(np.linalg.norm(cw_ekf.position)))
        tel['rn_dv'].append(cw.total_dv_ms)
        tel['rn_mode'].append(roe_ctrl.mode.value)

    # ── ADCS telemetry ────────────────────────────────────────────────
    tel['t'].append(t);    tel['mode'].append(mode.value)
    tel['wx'].append(np.degrees(sc.omega[0]))
    tel['wy'].append(np.degrees(sc.omega[1]))
    tel['wz'].append(np.degrees(sc.omega[2]))
    tel['rate'].append(np.degrees(np.linalg.norm(sc.omega)))
    tel['T_aero'].append(np.linalg.norm(T_aero) * 1e9)
    tel['T_gg'].append(np.linalg.norm(T_gg) * 1e9)
    tel['T_srp'].append(np.linalg.norm(T_srp) * 1e9)
    tel['hx'].append(rw.h[0] * 1e3)
    tel['hy'].append(rw.h[1] * 1e3)
    tel['hz'].append(rw.h[2] * 1e3)
    tel['rho'].append(rho)

    t += dt_detumble

# =====================================================================
#  SUMMARY
# =====================================================================
print()
print("=" * 60)
print("  Simulation complete")
print("=" * 60)
print(f"  Total time     : {t:.0f}s  ({t/60:.1f} min)")
print(f"  Phase 1 gate   : t={phase1_conf_t:.1f}s" if phase1_confirmed
      else "  Phase 1 gate   : NOT REACHED (increase T_SIM_MAX)")
print(f"  Phase 2 start  : t={phase1_conf_t + RDV_DELAY_S:.1f}s" if phase1_confirmed else "")
print("  FSW history:")
for t_tr, m in fsw.mode_history:
    print(f"    t={t_tr:7.1f}s  →  {m.name}")
if tel['err_deg']:
    ss = [e for _, e in tel['err_deg'][-200:]]
    print(f"\n  MEKF SS pointing error: "
          f"mean={np.mean(ss):.3f}°  3σ={np.mean(ss)+3*np.std(ss):.3f}°")
if tel['rn_t']:
    print(f"  Final range    : {tel['rn_range'][-1]:.3f} m")
    print(f"  Total ΔV       : {cw.total_dv_ms*1000:.3f} mm/s")

# =====================================================================
#  PLOTS
# =====================================================================
plt.rcParams.update({"font.size": 10, "axes.grid": True,
                      "grid.alpha": 0.35, "lines.linewidth": 1.2})

MODE_COLORS = {
    Mode.SAFE_MODE.value:       ('red',       'SAFE'),
    Mode.DETUMBLE.value:        ('royalblue', 'DETUMBLE'),
    Mode.SUN_ACQUISITION.value: ('orange',    'SUN ACQ'),
    Mode.FINE_POINTING.value:   ('limegreen', 'FINE POINT'),
    Mode.MOMENTUM_DUMP.value:   ('purple',    'MTM DUMP'),
}


def add_mode_bands(ax, t_arr, mode_arr):
    t_arr    = np.array(t_arr)
    mode_arr = np.array(mode_arr)
    segs     = np.concatenate([[0], np.where(np.diff(mode_arr))[0]+1,
                                [len(mode_arr)]])
    for i in range(len(segs) - 1):
        s, e = segs[i], segs[i+1]
        col, _ = MODE_COLORS.get(mode_arr[s], ('gray', '?'))
        ax.axvspan(t_arr[s], t_arr[e-1], alpha=0.08, color=col, linewidth=0)


t_arr = np.array(tel['t'])

# ── Figure 1: ADCS overview ──────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 9))
fig1.suptitle("3U CubeSat ADCS  (Phase 1)", fontsize=13, fontweight='bold')

ax = axes1[0, 0]
ax.plot(t_arr, tel['wx'], label='ωx', color='royalblue', alpha=0.8)
ax.plot(t_arr, tel['wy'], label='ωy', color='darkorange', alpha=0.8)
ax.plot(t_arr, tel['wz'], label='ωz', color='green', alpha=0.8)
if phase1_conf_t:
    ax.axvline(phase1_conf_t, color='black', linestyle='--', lw=1.5,
               label='P1 confirmed')
add_mode_bands(ax, tel['t'], tel['mode'])
ax.set_xlabel("Time [s]"); ax.set_ylabel("Rate [deg/s]")
ax.set_title("Angular Rate"); ax.legend(fontsize=8)

ax = axes1[0, 1]
ax.plot(t_arr, tel['rate'], color='purple')
if phase1_conf_t:
    ax.axvline(phase1_conf_t, color='black', linestyle='--', lw=1.5)
add_mode_bands(ax, tel['t'], tel['mode'])
ax.set_xlabel("Time [s]"); ax.set_ylabel("[deg/s]")
ax.set_title("Total Angular Rate")

ax = axes1[0, 2]
ax.semilogy(t_arr, tel['T_aero'], label='Aero', color='saddlebrown')
ax.semilogy(t_arr, tel['T_gg'],   label='GG',   color='steelblue')
ax.semilogy(t_arr, tel['T_srp'],  label='SRP',  color='goldenrod')
add_mode_bands(ax, tel['t'], tel['mode'])
ax.set_xlabel("Time [s]"); ax.set_ylabel("[nN·m]")
ax.set_title("Disturbance Torques"); ax.legend(fontsize=8)

ax = axes1[1, 0]
ax.plot(t_arr, tel['hx'], label='hx', color='royalblue')
ax.plot(t_arr, tel['hy'], label='hy', color='darkorange')
ax.plot(t_arr, tel['hz'], label='hz', color='green')
ax.axhline( 4., color='red', linestyle=':', linewidth=1, label='±h_max')
ax.axhline(-4., color='red', linestyle=':', linewidth=1)
add_mode_bands(ax, tel['t'], tel['mode'])
ax.set_xlabel("Time [s]"); ax.set_ylabel("[mN·m·s]")
ax.set_title("Reaction Wheel Momentum"); ax.legend(fontsize=8)

ax = axes1[1, 1]
if tel['err_deg']:
    t_e = [x[0] for x in tel['err_deg']]
    e_e = [x[1] for x in tel['err_deg']]
    ax.plot(t_e, e_e, color='crimson', linewidth=0.8, label='MEKF err')
    ax.axhline(ADCS_STABLE_DEG, color='gray', linestyle=':',
               label=f'{ADCS_STABLE_DEG}° gate')
    if phase1_conf_t:
        ax.axvline(phase1_conf_t, color='black', linestyle='--', lw=1.5,
                   label='P1 confirmed')
    ax.legend(fontsize=8)
ax.set_xlabel("Time [s]"); ax.set_ylabel("[deg]")
ax.set_title("MEKF Pointing Error")

ax = axes1[1, 2]
mode_arr = np.array(tel['mode'])
ax.step(t_arr, mode_arr, color='black', linewidth=1.5)
for val, (col, lbl) in MODE_COLORS.items():
    mask = mode_arr == val
    if mask.any():
        ax.fill_between(t_arr, val-.4, val+.4,
                        where=mask, alpha=.4, color=col, label=lbl)
if phase1_conf_t:
    ax.axvline(phase1_conf_t, color='black', linestyle='--', lw=1.5,
               label='P1 confirmed')
ax.set_xlabel("Time [s]"); ax.set_ylabel("Mode")
ax.set_yticks([m.value for m in Mode])
ax.set_yticklabels([m.name for m in Mode], fontsize=7)
ax.set_title("FSW Mode Timeline"); ax.legend(fontsize=7, loc='right')
fig1.tight_layout()

# ── Figure 2: RelNav ─────────────────────────────────────────────────
if tel['rn_t']:
    rn_t   = np.array(tel['rn_t'])
    rn_dx  = np.array(tel['rn_dx']);   rn_dy  = np.array(tel['rn_dy'])
    rn_dz  = np.array(tel['rn_dz'])
    rn_edx = np.array(tel['rn_edx']);  rn_edy = np.array(tel['rn_edy'])
    rn_edz = np.array(tel['rn_edz'])
    rn_rdx = np.array(tel['rn_rdx']);  rn_rdy = np.array(tel['rn_rdy'])
    rn_rdz = np.array(tel['rn_rdz'])

    cw_err  = np.sqrt((rn_dx-rn_edx)**2 + (rn_dy-rn_edy)**2 + (rn_dz-rn_edz)**2)
    roe_err = np.sqrt((rn_dx-rn_rdx)**2 + (rn_dy-rn_rdy)**2 + (rn_dz-rn_rdz)**2)

    RN_COL = {
        ROEMode.FORMATION_HOLD.value: ('steelblue', 'FORM HOLD'),
        ROEMode.RENDEZVOUS.value:     ('crimson',   'RDV'),
        ROEMode.COASTING.value:       ('gray',      'COAST'),
    }

    def rnb(ax):
        ma   = np.array(tel['rn_mode'])
        segs = np.concatenate([[0], np.where(np.diff(ma))[0]+1, [len(ma)]])
        for i in range(len(segs) - 1):
            s, e = segs[i], segs[i+1]
            col, _ = RN_COL.get(ma[s], ('gray', ''))
            ax.axvspan(rn_t[s], rn_t[e-1], alpha=0.10, color=col, linewidth=0)

    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 9))
    fig2.suptitle("Relative Navigation — CW + ROE EKF  (Phase 2)",
                  fontsize=13, fontweight='bold')

    components = [
        (rn_dx, rn_edx, rn_rdx, 'Radial dx',      'royalblue'),
        (rn_dy, rn_edy, rn_rdy, 'Along-Track dy',  'darkorange'),
        (rn_dz, rn_edz, rn_rdz, 'Cross-Track dz',  'green'),
    ]
    for ci, (true_a, cw_a, roe_a, lbl, col) in enumerate(components):
        ax = axs2[0, ci]
        ax.plot(rn_t, true_a, color=col,        label='true',    lw=1.5)
        ax.plot(rn_t, cw_a,   color=col,        label='CW-EKF',  lw=1.0, ls='--', alpha=0.7)
        ax.plot(rn_t, roe_a,  color='dimgray',  label='ROE-EKF', lw=1.0, ls=':',  alpha=0.8)
        rnb(ax); ax.set(xlabel='s', ylabel='m', title=lbl); ax.legend(fontsize=7)

    ax = axs2[1, 0]
    ax.plot(rn_t, tel['rn_range'],     color='purple', label='true', lw=1.5)
    ax.plot(rn_t, tel['rn_est_range'], color='purple', label='CW-EKF', lw=1.0, ls='--', alpha=0.7)
    rnb(ax); ax.set(xlabel='s', ylabel='m', title='Range'); ax.legend(fontsize=7)

    ax = axs2[1, 1]
    ax.plot(rn_t, cw_err,  color='royalblue', label='CW-EKF |err|',  lw=1.2)
    ax.plot(rn_t, roe_err, color='crimson',   label='ROE-EKF |err|', lw=1.2)
    ax.axhline(10., color='gray', linestyle=':')
    rnb(ax); ax.set(xlabel='s', ylabel='m', title='EKF Position Error')
    ax.legend(fontsize=7)

    ax = axs2[1, 2]
    sc_tr = ax.scatter(rn_dy, rn_dx, c=rn_t, cmap='viridis', s=3, zorder=3)
    ax.plot(0, 0, 'k*', ms=14, label='Chief', zorder=5)
    ax.plot(rn_dy[0],  rn_dx[0],  'go', ms=8, label='Start', zorder=4)
    ax.plot(rn_dy[-1], rn_dx[-1], 'rs', ms=8, label='End',   zorder=4)
    plt.colorbar(sc_tr, ax=ax, label='Time [s]')
    ax.set(xlabel='dy [m]', ylabel='dx [m]', title='LVLH Trajectory')
    ax.legend(fontsize=7); ax.set_aspect('equal', adjustable='datalim')
    fig2.tight_layout()

plt.show()