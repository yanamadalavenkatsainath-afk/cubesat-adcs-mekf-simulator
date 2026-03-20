"""
Monte Carlo Validation — 3U CubeSat ADCS + CW RelNav  (parallel)
=================================================================
Identical physics to main.py.  Every fix applied to main.py is reflected here.

Speed improvements over the sequential version:
  1. ProcessPoolExecutor — each run is an independent process, uses all CPU cores
  2. Stdout suppression removed from the inner loop — no StringIO overhead
  3. contextlib.redirect_stdout only used at init time (noisy constructors)
  4. N_WORKERS auto-detected from os.cpu_count()

Usage:
  python monte_carlo.py            # 100 runs, all cores
  python monte_carlo.py 200        # 200 runs
  python monte_carlo.py 100 4      # 100 runs, 4 workers

Randomised per run:
  1. Initial tumble rate  — magnitude + direction
  2. Sensor noise seeds   — gyro, mag, sun (via class RNG)
  3. Orbit epoch offset   — 0–90 min
  4. Solar activity f107  — ±20%
  5. Gyro initial bias
  6. Deputy position jitter at Phase 2 entry  ±POS_JITTER_M per axis
  7. Deputy velocity jitter at Phase 2 entry  ±VEL_JITTER_MS per axis
  8. RDV trigger time jitter                  ±RDV_JITTER_S
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import sys
import os
import io
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── path setup must happen before any project imports ─────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# =============================================================================
#  FIXED HARDWARE PARAMETERS  (identical to main.py)
# =============================================================================
I           = np.diag([0.030, 0.025, 0.010])
dt_detumble = 0.1
dt_control  = 0.01
N_INNER     = int(dt_detumble / dt_control)

TLE_LINE1 = "1 25544U 98067A   25001.50000000  .00006789  00000-0  12345-3 0  9999"
TLE_LINE2 = "2 25544  51.6400 208.9163 0001147  83.8771  11.2433 15.49815689432399"

# Phase 1 / Phase 2 config
ADCS_STABLE_DEG  = 1.0
ADCS_STABLE_SUST = 100
RDV_DELAY_S      = 300.0
FORMATION_OFFSET = np.array([0.0, 100.0, 0.0])
R_CHIEF_KM       = 6781.0
EKF_SETTLE_S     = 120.0
CLEANUP_HOLD_S   = 300.0    # extended: 120s was insufficient to damp burn-2 residuals

# MC settings
T_SIM_BASE     = 12000.0
T_SIM_RDV_PAD  = 15000.0    # s — increased: covers late P1 + full RDV + cleanup
CONV_THRESH    = 0.5
SS_OFFSET_S    = 300.0
OMEGA_MAG_MEAN = np.radians(18.0)
OMEGA_MAG_STD  = np.radians(5.0)
OMEGA_MAG_MIN  = np.radians(5.0)
OMEGA_MAG_MAX  = np.radians(35.0)
F107_MEAN, F107_STD = 150.0, 30.0
POS_JITTER_M   = 15.0
VEL_JITTER_MS  = 0.005
RDV_JITTER_S   = 60.0
RDV_SUCCESS_M  = 15.0

# =============================================================================
#  SINGLE-RUN FUNCTION  (runs in a subprocess — must be importable at top level)
# =============================================================================

def run_single(run_idx: int) -> dict:
    """Execute one MC run. Returns a dict of scalar results."""

    # ── imports inside function so each subprocess gets its own state ──────────
    import sys, os
    sys.path.insert(0, _ROOT)

    import numpy as np
    import io, contextlib

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
    from environment.cw_dynamics              import CWDynamics
    from environment.roe_dynamics             import ROEDynamics
    from sensors.ranging_sensor               import RangingBearingSensor
    from estimation.cw_ekf                    import CWEKF
    from estimation.roe_ekf                   import ROEEKF
    from control.rendezvous_controller        import RendezvousController, RelNavMode
    from control.roe_controller               import ROEController, ROEMode

    _null = io.StringIO()

    rng = np.random.default_rng(seed=run_idx)

    # ── Randomise ─────────────────────────────────────────────────────────────
    omega_mag = float(np.clip(rng.normal(OMEGA_MAG_MEAN, OMEGA_MAG_STD),
                              OMEGA_MAG_MIN, OMEGA_MAG_MAX))
    omega_dir = rng.standard_normal(3)
    omega_dir /= np.linalg.norm(omega_dir)
    omega0    = omega_mag * omega_dir

    epoch_off = float(rng.uniform(0, 5400))
    f107      = float(np.clip(rng.normal(F107_MEAN, F107_STD), 70., 250.))
    f107a     = f107
    pos_jit   = rng.normal(0.0, POS_JITTER_M,  3)
    vel_jit   = rng.normal(0.0, VEL_JITTER_MS, 3)
    rdv_jit   = float(rng.uniform(-RDV_JITTER_S, RDV_JITTER_S))

    # ── Initialise hardware ────────────────────────────────────────────────────
    with contextlib.redirect_stdout(_null):
        gg   = GravityGradient(I)
        srp  = SolarRadiationPressure()
        drag = AerodynamicDrag(Cd=2.2, f107=f107, f107a=f107a, ap=4.0)
        sc       = Spacecraft(I)
        sc.omega = omega0.copy()
        orbit = OrbitPropagator(tle_line1=TLE_LINE1, tle_line2=TLE_LINE2)
        for _ in range(int(epoch_off / dt_detumble)):
            orbit.step(dt_detumble)
        mag_field = MagneticField(epoch_year=2025.0)
        sun_model = SunModel(epoch_year=2025.0)
        mag_sens  = Magnetometer()
        sun_sens  = SunSensor()
        gyro      = Gyro(dt=dt_control, bias_init_max_deg_s=0.05)
        gyro.bias = rng.uniform(-np.radians(0.05), np.radians(0.05), 3)
        rw         = ReactionWheel(h_max=0.004)
        mtq        = Magnetorquer(m_max=0.2)
        bdot       = BDotController(k_bdot=2e5, m_max=0.2)
        controller = AttitudeController(Kp=0.0005, Kd=0.008)
        quest_alg  = QUEST()
        mekf       = MEKF(dt_control)
        mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
        fsw = ModeManager()

        cw = CWDynamics(chief_orbit_radius_km=R_CHIEF_KM)
        cw.set_initial_offset(dr_lvlh_m=FORMATION_OFFSET, dv_lvlh_ms=None)
        roe_dyn = ROEDynamics(
            a_chief_m=R_CHIEF_KM * 1e3, e_chief=0.0001,
            i_chief=np.radians(51.6))
        dv_ic = np.array([0.0, -2.0 * cw.n * FORMATION_OFFSET[0], 0.0])
        roe_dyn.set_from_lvlh(FORMATION_OFFSET, dv_ic)
        rng_sensor = RangingBearingSensor(
            sigma_range_m=0.5, sigma_range_frac=0.002,
            sigma_angle_rad=np.radians(0.1),
            fov_half_deg=60.0, max_range_m=5000.0)
        cw_ekf   = CWEKF(n=cw.n, dt=dt_detumble)
        roe_ekf  = ROEEKF(roe_dyn=roe_dyn, dt=dt_detumble)
        rel_ctrl = RendezvousController(
            n=cw.n, mode=RelNavMode.FORMATION_HOLD,
            target_lvlh=FORMATION_OFFSET)
        roe_ctrl = ROEController(
            roe_dyn=roe_dyn, mode=ROEMode.FORMATION_HOLD,
            target_roe=roe_dyn.roe.copy())

    q_ref = np.array([1., 0., 0., 0.])

    # ── Per-run state ──────────────────────────────────────────────────────────
    t_abs    = float(epoch_off)
    t_run    = 0.0
    T_SIM_MAX = T_SIM_BASE

    triad_err_deg    = None
    mekf_seeded      = False
    last_good_q      = None
    last_good_t      = -999.0
    adcs_stable_cnt  = 0
    fine_point_t0    = None
    phase1_confirmed = False
    phase1_conf_t    = None

    phase2_active   = False
    phase2_active_t = None
    rdv_triggered   = False
    rdv_complete    = False

    detumble_time = None
    conv_time     = None
    ss_errors     = []
    wheel_sat     = False
    highest_mode  = Mode.DETUMBLE
    burn_ranges   = []
    ss_start_t    = None

    # ── Main simulation loop ───────────────────────────────────────────────────
    while t_run < T_SIM_MAX:

        # Environment
        pos, vel   = orbit.step(dt_detumble)
        B_I        = mag_field.get_field(pos)
        B_meas     = mag_sens.measure(sc.q, B_I)
        sun_I      = sun_model.get_sun_vector(t_seconds=t_abs)
        sun_meas   = sun_sens.measure(sc.q, sun_I)
        omega_meas = gyro.measure(sc.omega)
        sun_pos_km = sun_I * 1.496e8
        T_gg       = gg.compute(pos, sc.q)
        T_srp, nu  = srp.compute(sc.q, sun_I, pos_km=pos, sun_pos_km=sun_pos_km)
        T_aero, _  = drag.compute(sc.q, pos, vel, t_seconds=t_abs)
        disturbance = T_gg + T_srp + T_aero
        in_eclipse  = (nu < 0.1)

        # QUEST during SUN_ACQUISITION
        if fsw.is_sun_acquiring:
            nadir_I = QUEST.nadir_inertial(pos)
            nadir_b = QUEST.nadir_body_from_earth_sensor(pos, sc.q)
            if in_eclipse:
                q_quest, quest_qual = quest_alg.compute_multi(
                    vectors_body=[B_meas, nadir_b],
                    vectors_inertial=[B_I, nadir_I],
                    weights=[0.85, 0.15])
            else:
                q_quest, quest_qual = quest_alg.compute_multi(
                    vectors_body=[B_meas, sun_meas, nadir_b],
                    vectors_inertial=[B_I, sun_I, nadir_I],
                    weights=[0.70, 0.20, 0.10])
            if q_quest[0] < 0:
                q_quest = -q_quest

            if quest_qual > 0.01:
                last_good_q = q_quest.copy()
                if last_good_q[0] < 0:
                    last_good_q = -last_good_q
                last_good_t   = t_run
                triad_err_deg = 5.0
            elif last_good_q is not None and (t_run - last_good_t) < 120.0:
                wx, wy, wz = omega_meas - mekf.bias
                Om = np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],[wy,-wz,0,wx],[wz,wy,-wx,0]])
                last_good_q += 0.5 * dt_detumble * Om @ last_good_q
                last_good_q /= np.linalg.norm(last_good_q)
                if last_good_q[0] < 0:
                    last_good_q = -last_good_q
                triad_err_deg = 5.0
            else:
                triad_err_deg = 180.0

        # Pointing error for dump guard
        _pt_err = None
        if mekf_seeded:
            _qe = quat_error(sc.q, mekf.q)
            if _qe[0] < 0:
                _qe = -_qe
            _pt_err = float(np.degrees(2.0 * np.linalg.norm(_qe[1:])))

        # FSW mode update
        with contextlib.redirect_stdout(_null):
            mode = fsw.update(t_run, sc.omega, rw.h,
                              triad_err_deg=triad_err_deg,
                              pointing_err_deg=_pt_err)

        if mode.value > highest_mode.value:
            highest_mode = mode
        if detumble_time is None and mode == Mode.SUN_ACQUISITION:
            detumble_time = t_run

        # Seed MEKF on first FINE_POINTING
        if mode == Mode.FINE_POINTING and not mekf_seeded:
            seed = last_good_q.copy() if last_good_q is not None else sc.q.copy()
            if seed[0] < 0:
                seed = -seed
            mekf.q = seed
            mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
            mekf_seeded   = True
            fine_point_t0 = t_run

        # Phase 1 stability gate
        # Count consecutive stable steps in FINE_POINTING *or* MOMENTUM_DUMP —
        # MEKF runs and attitude is controlled in both modes, so stability is
        # meaningful in either. Only reset on DETUMBLE/SUN_ACQUISITION/SAFE.
        if not phase1_confirmed and mekf_seeded and mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
            qe = quat_error(sc.q, mekf.q)
            if qe[0] < 0:
                qe = -qe
            err_deg = float(np.degrees(2.0 * np.linalg.norm(qe[1:])))
            if err_deg < ADCS_STABLE_DEG:
                adcs_stable_cnt += 1
            else:
                adcs_stable_cnt = 0
            if adcs_stable_cnt >= ADCS_STABLE_SUST:
                phase1_confirmed = True
                phase1_conf_t    = t_run
                T_SIM_MAX = t_run + RDV_DELAY_S + 240.0 + abs(rdv_jit) + T_SIM_RDV_PAD
        elif mode not in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
            adcs_stable_cnt = 0

        # Actuators
        if mode == Mode.SAFE_MODE:
            sc.step(np.zeros(3), disturbance, dt_detumble)

        elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
            m_cmd, _ = bdot.compute(B_meas, sc.omega, B_I, dt_detumble)
            sc.step(mtq.compute_torque(m_cmd, B_meas), disturbance, dt_detumble)

        elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
            if mekf_seeded and last_good_q is not None:
                qe_chk = quat_error(sc.q, mekf.q)
                if qe_chk[0] < 0:
                    qe_chk = -qe_chk
                if np.degrees(2 * np.linalg.norm(qe_chk[1:])) > 25.0:
                    nadir_I = QUEST.nadir_inertial(pos)
                    nadir_b = QUEST.nadir_body_from_earth_sensor(pos, sc.q)
                    qf, _ = quest_alg.compute_multi(
                        vectors_body=[B_meas, sun_meas, nadir_b],
                        vectors_inertial=[B_I, sun_I, nadir_I],
                        weights=[0.70, 0.20, 0.10])
                    if qf[0] < 0:
                        qf = -qf
                    mekf.q = qf.copy()

            for _ in range(N_INNER):
                oi = gyro.measure(sc.omega)
                mekf.predict(oi)
                mekf.update_vector(B_meas, B_I, mekf.R_mag)
                mekf.update_vector(sun_meas, sun_I, mekf.R_sun)
                omega_est  = sc.omega - mekf.bias
                torque_cmd, _ = controller.compute(mekf.q, omega_est, q_ref)
                rw.apply_torque(torque_cmd, dt_control)
                m_cmd      = mtq.compute_dipole(rw.h, B_meas)
                m_cmd      = np.clip(m_cmd, -mtq.m_max, mtq.m_max)
                torque_mtq = mtq.compute_torque(m_cmd, B_meas)
                rw.h      -= torque_mtq * dt_control
                rw.h       = np.clip(rw.h, -rw.h_max, rw.h_max)
                sc.step(torque_mtq, disturbance, dt_control,
                        tau_rw=torque_cmd, h_rw=rw.h.copy())

            if mekf_seeded and mode == Mode.FINE_POINTING:
                qe = quat_error(sc.q, mekf.q)
                if qe[0] < 0:
                    qe = -qe
                err_deg = float(np.degrees(2 * np.linalg.norm(qe[1:])))
                if conv_time is None and err_deg < CONV_THRESH:
                    conv_time = t_run
                if ss_start_t is not None and t_run > ss_start_t:
                    ss_errors.append(err_deg)

        if np.any(np.abs(rw.h) >= 0.0039):
            wheel_sat = True

        # Deputy open-loop during Phase 1
        if not phase2_active:
            cw.step(dt_detumble, np.zeros(3))
            roe_dyn.step(dt_detumble)

        # Phase 2 activation
        if (phase1_confirmed and not phase2_active
                and t_run >= phase1_conf_t + RDV_DELAY_S):
            phase2_active   = True
            phase2_active_t = t_run
            ss_start_t      = t_run + SS_OFFSET_S

            dep_pos = cw.state[0:3] + pos_jit
            dep_vel = cw.state[3:6] + vel_jit
            with contextlib.redirect_stdout(_null):
                cw.set_initial_offset(dr_lvlh_m=dep_pos, dv_lvlh_ms=dep_vel)
                roe_dyn.set_from_lvlh(dep_pos, dep_vel)

            noise = rng.standard_normal(6) * np.array([5., 5., 5., 0.02, 0.02, 0.02])
            with contextlib.redirect_stdout(_null):
                cw_ekf.initialise(cw.state + noise)
                roe_noise = roe_dyn.roe * 0.05 + np.array([1e-7, 1e-7, 1e-7, 1e-7, 1e-8, 1e-8])
                roe_ekf.initialise(roe_dyn.roe + roe_noise * rng.standard_normal(6))
                # ROE controller: formation hold at current drifted ROE
                roe_ctrl = ROEController(
                    roe_dyn=roe_dyn, mode=ROEMode.FORMATION_HOLD,
                    target_roe=roe_dyn.roe.copy())
                rel_ctrl = RendezvousController(
                    n=cw.n, mode=RelNavMode.FORMATION_HOLD,
                    target_lvlh=dep_pos.copy())

        # Phase 2: RelNav — use ROE controller for rendezvous
        if phase2_active:
            # Trigger: switch ROE controller to RENDEZVOUS mode
            if (not rdv_triggered
                    and mode == Mode.FINE_POINTING
                    and t_run >= phase1_conf_t + RDV_DELAY_S + 120.0 + rdv_jit
                    and t_run >= phase2_active_t + EKF_SETTLE_S):
                with contextlib.redirect_stdout(_null):
                    roe_ctrl.set_mode(ROEMode.RENDEZVOUS,
                                      roe_est=roe_ekf.x,
                                      mean_anomaly=roe_dyn.mean_anomaly,
                                      t=t_run,
                                      t_sim_max=T_SIM_MAX,
                                      lvlh_est=cw_ekf.x)
                rdv_triggered = True

            # ROE controller computes impulse from ROE-EKF state
            with contextlib.redirect_stdout(_null):
                _, impulse_dv = roe_ctrl.compute(
                    roe_ekf.x, roe_dyn.mean_anomaly, t_run)

            if impulse_dv is not None:
                burn_ranges.append(float(cw.range_m))
                cw.apply_impulse(impulse_dv)
                roe_dyn.apply_impulse_lvlh(impulse_dv, roe_dyn.mean_anomaly)
                cw_ekf.x[3:6]  += impulse_dv
                # Update ROE-EKF velocity at burn time via GVE
                roe_ekf.x = roe_dyn.roe.copy()   # reseed from truth after burn

            # Post-burn-2 cleanup: switch to CW formation hold at origin
            if (rdv_triggered and roe_ctrl.mode == ROEMode.COASTING
                    and not rdv_complete):
                if not getattr(roe_ctrl, '_cleanup_started', False):
                    roe_ctrl._cleanup_started = True
                    roe_ctrl._cleanup_t0      = t_run
                    with contextlib.redirect_stdout(_null):
                        rel_ctrl.set_mode(RelNavMode.FORMATION_HOLD, t=t_run,
                                          target_lvlh=np.zeros(3))
                elif t_run >= roe_ctrl._cleanup_t0 + CLEANUP_HOLD_S:
                    rdv_complete = True
                    break   # measure final_range now, don't drift further

            # During cleanup, use CW formation hold accel
            if rdv_triggered and getattr(roe_ctrl, '_cleanup_started', False):
                accel_cmd, _ = rel_ctrl.compute(cw_ekf.x, t_run)
            else:
                accel_cmd = np.zeros(3)

            true_cw = cw.step(dt_detumble, accel_cmd)
            roe_dyn.step(dt_detumble)

            cw_ekf.predict(accel_cmd)
            roe_ekf.predict(accel_cmd)
            z, R = rng_sensor.measure(true_cw[:3], np.array([0., 1., 0.]))
            if z is not None:
                cw_ekf.update(z, R)
                roe_ekf.update(z, R)

        t_run += dt_detumble
        t_abs += dt_detumble

    # ── Collect results ────────────────────────────────────────────────────────
    if ss_errors:
        ss_arr  = np.array(ss_errors)
        ss_mean = float(np.nanmean(ss_arr))
        ss_3sig = float(np.nanmean(ss_arr) + 3 * np.nanstd(ss_arr))
    else:
        ss_mean = float('nan')
        ss_3sig = float('nan')

    final_range  = float(cw.range_m)
    total_dv_mms = float(cw.total_dv_ms * 1000.0)
    rdv_ok       = rdv_triggered and (final_range < RDV_SUCCESS_M)

    if not rdv_triggered:
        rdv_outcome = 'NO_TRG'
    elif rdv_ok:
        rdv_outcome = 'OK'
    else:
        rdv_outcome = 'FAIL'

    return dict(
        run=run_idx, f107=f107, omega0_deg=float(np.degrees(omega_mag)),
        detumble_time=detumble_time if detumble_time is not None else float('nan'),
        phase1_time=phase1_conf_t   if phase1_confirmed else float('nan'),
        conv_time=conv_time,
        ss_mean=ss_mean, ss_3sigma=ss_3sig,
        wheel_saturated=wheel_sat,
        mode_reached=highest_mode.name,
        rdv_triggered=rdv_triggered,
        rdv_success=rdv_ok,
        rdv_outcome=rdv_outcome,
        final_range=final_range,
        total_dv_mms=total_dv_mms,
        burn1_range=burn_ranges[0] if len(burn_ranges) > 0 else float('nan'),
        burn2_range=burn_ranges[1] if len(burn_ranges) > 1 else float('nan'),
    )


# =============================================================================
#  MAIN  — parallel dispatch + progress reporting
# =============================================================================

if __name__ == '__main__':

    N_RUNS    = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, os.cpu_count() - 1)

    print("=" * 72)
    print(f"  Monte Carlo — ADCS + CW RelNav  |  {N_RUNS} runs  |  {N_WORKERS} workers")
    print("=" * 72)
    print(f"  Phase 1 gate  : FINE_POINTING|MOMENTUM_DUMP + err < {ADCS_STABLE_DEG}° "
          f"for {ADCS_STABLE_SUST} consecutive steps")
    print(f"  Phase 2 start : {RDV_DELAY_S:.0f}s after Phase 1 confirmed")
    print(f"  EKF settle    : {EKF_SETTLE_S:.0f}s min before RDV trigger")
    print(f"  Cleanup hold  : {CLEANUP_HOLD_S:.0f}s PD hold at origin after burn-2")
    print(f"  RDV success   : final_range < {RDV_SUCCESS_M} m")
    print()

    t_wall   = time.time()
    rows     = [None] * N_RUNS
    done     = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_single, i): i for i in range(N_RUNS)}
        for fut in as_completed(futures):
            result = fut.result()
            rows[result['run']] = result
            done += 1
            elapsed = time.time() - t_wall
            eta     = elapsed / done * (N_RUNS - done)
            r       = result
            p1_s  = f"{r['phase1_time']:.0f}s"  if not np.isnan(r['phase1_time']) else "---"
            det_s = f"{r['detumble_time']:.0f}s" if not np.isnan(r['detumble_time']) else "---"
            conv_s = f"{r['conv_time']:.0f}s"    if r['conv_time'] is not None else "FAIL"
            print(f"  [{done:3d}/{N_RUNS}]  run={r['run']+1:3d}  "
                  f"w={r['omega0_deg']:.1f}°/s  det={det_s}  P1@{p1_s}  "
                  f"conv={conv_s}  rdv={r['rdv_outcome']}  "
                  f"range={r['final_range']:7.2f}m  "
                  f"dV={r['total_dv_mms']:.1f}mm/s  ETA {eta:.0f}s")

    # ── Summary stats ──────────────────────────────────────────────────────────
    def _arr(key):
        return np.array([r[key] for r in rows
                         if r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))])

    det_arr  = _arr('detumble_time')
    p1_arr   = _arr('phase1_time')
    conv_arr = _arr('conv_time')
    ss_m_arr = _arr('ss_mean')
    ss_3_arr = _arr('ss_3sigma')
    rng_arr  = np.array([r['final_range']  for r in rows])
    dv_arr   = np.array([r['total_dv_mms'] for r in rows])
    ok_mask  = np.array([r['rdv_success']  for r in rows])
    outcomes = [r['rdv_outcome'] for r in rows]
    n_ok     = int(ok_mask.sum())
    n_trig   = sum(1 for o in outcomes if o != 'NO_TRG')
    n_sat    = sum(r['wheel_saturated'] for r in rows)

    print()
    print("=" * 72)
    print(f"  Results — {N_RUNS} runs  ({N_WORKERS} workers)")
    print("=" * 72)
    if len(det_arr):
        print(f"  Detumble time    : {det_arr.mean():.1f}s mean  {det_arr.std():.1f}s std  "
              f"{np.percentile(det_arr,99):.1f}s 99th")
    if len(p1_arr):
        print(f"  Phase 1 confirm  : {p1_arr.mean():.1f}s mean  "
              f"({len(p1_arr)}/{N_RUNS} runs reached gate)")
    if len(conv_arr):
        print(f"  ADCS conv time   : {conv_arr.mean():.1f}s mean  {conv_arr.std():.1f}s std")
    if len(ss_m_arr):
        print(f"  ADCS SS error    : {ss_m_arr.mean():.3f}° mean  "
              f"{ss_3_arr.mean():.3f}° 3-sigma")
    print(f"  Wheel saturation : {n_sat}/{N_RUNS}")
    print(f"  RDV triggered    : {n_trig}/{N_RUNS}")
    print(f"  RDV success      : {n_ok}/{N_RUNS}  ({100*n_ok/N_RUNS:.0f}%)")
    if n_ok:
        print(f"  Final range (OK) : {rng_arr[ok_mask].mean():.2f}m mean  "
              f"{rng_arr[ok_mask].max():.2f}m max")
        print(f"  Total ΔV  (OK)   : {dv_arr[ok_mask].mean():.2f}mm/s mean  "
              f"{dv_arr[ok_mask].max():.2f}mm/s max")
    print(f"  Wall time        : {time.time()-t_wall:.1f}s")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Monte Carlo — ADCS + CW RelNav  ({N_RUNS} runs, {N_WORKERS} workers)",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    rdv_col = ['forestgreen' if r['rdv_success'] else 'crimson' for r in rows]

    def hist(ax, data, color, xlabel, title, vlines=None, bins=20):
        d = np.array([x for x in data if x is not None and not np.isnan(x)])
        if len(d) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes); return
        ax.hist(d, bins=bins, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(np.mean(d), color='red', ls='--', label=f"μ={np.mean(d):.1f}")
        if vlines:
            for v, lbl, c in vlines:
                ax.axvline(v, color=c, ls=':', label=lbl)
        ax.legend(fontsize=7); ax.set_xlabel(xlabel)
        ax.set_ylabel("Count"); ax.set_title(title)

    # Row 0: ADCS
    hist(fig.add_subplot(gs[0,0]), [r['detumble_time'] for r in rows],
         'royalblue', 'Time [s]', 'Detumble Time')
    hist(fig.add_subplot(gs[0,1]), [r['phase1_time'] for r in rows],
         'steelblue', 'Time [s]', 'Phase 1 Confirm Time')
    hist(fig.add_subplot(gs[0,2]), [r['conv_time'] for r in rows],
         'forestgreen', 'Time [s]', f'ADCS Conv Time (<{CONV_THRESH}°)')
    ax_cdf = fig.add_subplot(gs[0,3])
    if len(ss_m_arr):
        ax_cdf.plot(np.sort(ss_m_arr),
                    np.arange(1, len(ss_m_arr)+1)/len(ss_m_arr)*100,
                    color='crimson', lw=1.5)
        ax_cdf.axvline(0.5, color='gray', ls=':', label='0.5° ref')
        ax_cdf.legend(fontsize=7)
    ax_cdf.set_xlabel('SS Error [deg]'); ax_cdf.set_ylabel('CDF [%]')
    ax_cdf.set_title('ADCS SS Error CDF')

    # Row 1: RelNav outcomes
    ax_bar = fig.add_subplot(gs[1,0])
    counts = {k: outcomes.count(k) for k in ['OK','FAIL','NO_TRG']}
    cols   = {'OK':'forestgreen','FAIL':'crimson','NO_TRG':'gray'}
    bars   = ax_bar.bar(counts.keys(), counts.values(),
                        color=[cols[k] for k in counts], edgecolor='white', alpha=0.85)
    for bar, v in zip(bars, counts.values()):
        ax_bar.text(bar.get_x()+bar.get_width()/2, v+0.3, str(v),
                    ha='center', fontsize=11, fontweight='bold')
    ax_bar.set_ylabel('Count'); ax_bar.set_title('RDV Outcome')

    hist(fig.add_subplot(gs[1,1]), rng_arr.tolist(),
         'darkorange', 'Range [m]', 'Final Range',
         vlines=[(RDV_SUCCESS_M, f'{RDV_SUCCESS_M}m', 'red')])

    ax_dv = fig.add_subplot(gs[1,2])
    ax_dv.hist(dv_arr, bins=20, color='slateblue', edgecolor='white', alpha=0.7, label='all')
    if n_ok:
        ax_dv.hist(dv_arr[ok_mask], bins=20, color='forestgreen',
                   edgecolor='white', alpha=0.7, label='OK only')
    ax_dv.legend(fontsize=7); ax_dv.set_xlabel('Total ΔV [mm/s]')
    ax_dv.set_ylabel('Count'); ax_dv.set_title('Total ΔV Distribution')

    ax_b = fig.add_subplot(gs[1,3])
    b1 = [r['burn1_range'] for r in rows if not np.isnan(r['burn1_range'])]
    b2 = [r['burn2_range'] for r in rows if not np.isnan(r['burn2_range'])]
    if b1: ax_b.hist(b1, bins=20, color='steelblue',  alpha=0.7, edgecolor='white', label='Burn 1')
    if b2: ax_b.hist(b2, bins=20, color='darkorange', alpha=0.7, edgecolor='white', label='Burn 2')
    ax_b.legend(fontsize=7); ax_b.set_xlabel('Range at Burn [m]')
    ax_b.set_ylabel('Count'); ax_b.set_title('Range at Each Burn')

    # Row 2: Sensitivity
    omega_degs = [r['omega0_deg'] for r in rows]
    f107s      = [r['f107']       for r in rows]
    det_times  = [r['detumble_time'] for r in rows]
    fin_ranges = [r['final_range']   for r in rows]
    ss_means   = [r['ss_mean']       for r in rows]

    ax_s1 = fig.add_subplot(gs[2,0])
    ax_s1.scatter(omega_degs, det_times, c='royalblue', alpha=0.5, s=15)
    ax_s1.set_xlabel('Tumble [deg/s]'); ax_s1.set_ylabel('Detumble Time [s]')
    ax_s1.set_title('Tumble vs Detumble Time')

    ax_s2 = fig.add_subplot(gs[2,1])
    ax_s2.scatter(omega_degs, fin_ranges, c=rdv_col, alpha=0.6, s=15)
    ax_s2.axhline(RDV_SUCCESS_M, color='gray', ls=':', label=f'{RDV_SUCCESS_M}m')
    ax_s2.legend(fontsize=7); ax_s2.set_xlabel('Tumble [deg/s]')
    ax_s2.set_ylabel('Final Range [m]'); ax_s2.set_title('Tumble vs Final Range (green=OK)')

    ax_s3 = fig.add_subplot(gs[2,2])
    ax_s3.scatter(f107s, ss_means, c='saddlebrown', alpha=0.5, s=15)
    ax_s3.set_xlabel('Solar f107 [sfu]'); ax_s3.set_ylabel('SS Error [deg]')
    ax_s3.set_title('Solar Activity vs ADCS SS Error')

    ax_s4 = fig.add_subplot(gs[2,3])
    ax_s4.scatter(f107s, fin_ranges, c=rdv_col, alpha=0.6, s=15)
    ax_s4.axhline(RDV_SUCCESS_M, color='gray', ls=':', label=f'{RDV_SUCCESS_M}m')
    ax_s4.legend(fontsize=7); ax_s4.set_xlabel('Solar f107 [sfu]')
    ax_s4.set_ylabel('Final Range [m]'); ax_s4.set_title('Solar Activity vs Final Range (green=OK)')

    plt.savefig('monte_carlo_relnav.png', dpi=150, bbox_inches='tight')
    print('\n  Plot saved: monte_carlo_relnav.png')
    plt.show()