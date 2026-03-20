"""
Live 3D Orbital Visualiser  —  Dual Panel
==========================================
Run from sim root:   python3 visualiser.py

LEFT panel  — Global ECI orbit view (~8500 km scale)
              Earth sphere, orbit ring, chief gold trail
              Both dots visible on the orbit arc

RIGHT panel — Close-up LVLH view centred on chief (~300 m window)
              Chief always at centre, deputy orbits around it,
              orange separation line, range label
              THIS is where the 100 m gap is actually visible

HUD (centre top) — T+, mode, range, DV, MEKF, Phase-1 gate, RDV

Close window -> post-sim Figure 2 (LVLH spatial) + Figure 3 (3D views)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # swap to 'Qt5Agg' if needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import sys, os, io, contextlib

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
from environment.cw_dynamics              import CWDynamics
from actuators.reaction_wheel             import ReactionWheel
from actuators.magnetorquer               import Magnetorquer
from actuators.bdot                       import BDotController
from control.attitude_controller          import AttitudeController
from control.rendezvous_controller        import RendezvousController, RelNavMode
from estimation.mekf                      import MEKF
from estimation.quest                     import QUEST
from estimation.cw_ekf                    import CWEKF
from utils.quaternion                     import quat_error
from fsw.mode_manager                     import ModeManager, Mode
from sensors.ranging_sensor               import RangingBearingSensor

# ─────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────
T_SIM_MAX           = 7200.0
dt_outer            = 0.1
dt_inner            = 0.01
N_INNER             = int(dt_outer / dt_inner)

TLE_LINE1 = "1 25544U 98067A   25001.50000000  .00006789  00000-0  12345-3 0  9999"
TLE_LINE2 = "2 25544  51.6400 208.9163 0001147  83.8771  11.2433 15.49815689432399"
R_CHIEF_KM   = 6371.0 + 410.0
I_CHIEF_DEG  = 51.6
OFFSET_LVLH  = np.array([0., 100., 0.])   # m

ADCS_STABLE_DEG  = 1.0
ADCS_STABLE_SUST = 100
RDV_DELAY_S      = 300.0

I_sc = np.diag([0.030, 0.025, 0.010])

TRAIL_LEN           = 600    # samples per trail buffer
ANIM_INTERVAL       = 50     # ms per frame
SIM_STEPS_PER_FRAME = 50

# tight-view half-width — scales with current range so deputy stays visible
TIGHT_MIN_KM  = 0.15   # 150 m minimum
TIGHT_SCALE   = 1.8    # multiplier on current range

# ─────────────────────────────────────────────────────────────────────
#  Initialise simulation
# ─────────────────────────────────────────────────────────────────────
print("Initialising simulation...")

with contextlib.redirect_stdout(io.StringIO()):
    gg        = GravityGradient(I_sc)
    srp       = SolarRadiationPressure()
    drag      = AerodynamicDrag(Cd=2.2, f107=150., f107a=150., ap=4.0)
    sc        = Spacecraft(I_sc)
    sc.omega  = np.radians(np.array([5., 8., 12.]))
    orbit     = OrbitPropagator(tle_line1=TLE_LINE1, tle_line2=TLE_LINE2)
    mag_field = MagneticField(epoch_year=2025.0)
    sun_model = SunModel(epoch_year=2025.0)
    mag_sens  = Magnetometer()
    sun_sens  = SunSensor()
    gyro      = Gyro(dt=dt_inner, bias_init_max_deg_s=0.05)
    rw        = ReactionWheel(h_max=0.004)
    mtq       = Magnetorquer(m_max=0.2)
    bdot      = BDotController(k_bdot=2e5, m_max=0.2)
    ctrl      = AttitudeController(Kp=0.0005, Kd=0.008)
    quest_alg = QUEST()
    mekf      = MEKF(dt_inner)
    mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.0)**2
    fsw       = ModeManager()

    cw = CWDynamics(chief_orbit_radius_km=R_CHIEF_KM)
    cw.set_initial_offset(dr_lvlh_m=OFFSET_LVLH, dv_lvlh_ms=None)

    rng_sensor = RangingBearingSensor(
        sigma_range_m=0.5, sigma_range_frac=0.002,
        sigma_angle_rad=np.radians(0.1), fov_half_deg=60.0, max_range_m=5000.0)

    cw_ekf   = CWEKF(n=cw.n, dt=dt_outer)
    rel_ctrl = RendezvousController(n=cw.n, mode=RelNavMode.FORMATION_HOLD,
                                    target_lvlh=OFFSET_LVLH)

# ─────────────────────────────────────────────────────────────────────
#  Sim state
# ─────────────────────────────────────────────────────────────────────
S = dict(
    t=0.0,
    q_ref=np.array([1., 0., 0., 0.]),
    last_good_q=None, last_good_t=-999.,
    mekf_seeded=False,
    adcs_stable_cnt=0,
    phase1_confirmed=False, phase1_conf_t=None,
    phase2_active=False, rdv_triggered=False,
    mode_name='DETUMBLE',
    adcs_err_deg=0., total_dv_ms=0.,
    running=True,
    # ECI km
    chief_pos_km=np.array([R_CHIEF_KM, 0., 0.]),
    deputy_pos_km=np.array([R_CHIEF_KM, 0.0001, 0.]),
    # LVLH relative position in km (for right panel)
    dr_km=np.array([0., 0.1, 0.]),
    chief_trail_eci=[],    # ECI km — for left panel
    deputy_trail_eci=[],   # ECI km — for left panel
    chief_trail_rel=[],    # always [0,0,0] — for right panel
    deputy_trail_rel=[],   # LVLH km — for right panel
)

TEL = dict(t=[], dx=[], dy=[], dz=[], sep=[],
           burns=[], chief_eci=[], deputy_eci=[])

# ─────────────────────────────────────────────────────────────────────
#  Simulation step
# ─────────────────────────────────────────────────────────────────────
def sim_step():
    if S['t'] >= T_SIM_MAX:
        S['running'] = False
        return

    with contextlib.redirect_stdout(io.StringIO()):
        pos_km, vel_km = orbit.step(dt_outer)
    pos_km = np.array(pos_km)
    vel_km = np.array(vel_km)

    with contextlib.redirect_stdout(io.StringIO()):
        B_I        = mag_field.get_field(pos_km)
        B_meas     = mag_sens.measure(sc.q, B_I)
        sun_I      = sun_model.get_sun_vector(t_seconds=S['t'])
        sun_meas   = sun_sens.measure(sc.q, sun_I)
        omega_meas = gyro.measure(sc.omega)
        sun_pos_km = sun_I * 1.496e8
        T_gg       = gg.compute(pos_km, sc.q)
        T_srp, nu  = srp.compute(sc.q, sun_I, pos_km=pos_km,
                                  sun_pos_km=sun_pos_km)
        T_aero, _  = drag.compute(sc.q, pos_km, vel_km, t_seconds=S['t'])
    dist = T_gg + T_srp + T_aero

    if fsw.is_sun_acquiring:
        in_ecl  = (nu < 0.1)
        nadir_I = QUEST.nadir_inertial(pos_km)
        nadir_b = QUEST.nadir_body_from_earth_sensor(pos_km, sc.q)
        vb = [B_meas, nadir_b]  if in_ecl else [B_meas, sun_meas, nadir_b]
        vi = [B_I,    nadir_I]  if in_ecl else [B_I,    sun_I,    nadir_I]
        wt = [0.85,   0.15]     if in_ecl else [0.70,   0.20,     0.10]
        q_q, q_qual = quest_alg.compute_multi(
            vectors_body=vb, vectors_inertial=vi, weights=wt)
        if q_q[0] < 0: q_q = -q_q
        if q_qual > 0.01:
            S['last_good_q'] = q_q.copy()
            if S['last_good_q'][0] < 0: S['last_good_q'] = -S['last_good_q']
            S['last_good_t'] = S['t']
        elif S['last_good_q'] is not None and S['t'] - S['last_good_t'] < 120.0:
            wx, wy, wz = omega_meas - mekf.bias
            Om = np.array([[ 0,-wx,-wy,-wz],[wx, 0, wz,-wy],
                           [wy,-wz,  0, wx],[wz, wy,-wx, 0]])
            S['last_good_q'] += 0.5 * dt_outer * Om @ S['last_good_q']
            S['last_good_q'] /= np.linalg.norm(S['last_good_q'])
            if S['last_good_q'][0] < 0: S['last_good_q'] = -S['last_good_q']

    triad = 5.0 if S['last_good_q'] is not None else 180.0
    mode  = fsw.update(S['t'], sc.omega, rw.h, triad_err_deg=triad)
    S['mode_name'] = mode.name

    if mode == Mode.FINE_POINTING and not S['mekf_seeded']:
        seed = S['last_good_q'].copy() if S['last_good_q'] is not None else sc.q.copy()
        if seed[0] < 0: seed = -seed
        mekf.q = seed
        mekf.P[0:3, 0:3] = np.eye(3) * np.radians(5.)**2
        S['mekf_seeded'] = True

    if not S['phase1_confirmed'] and S['mekf_seeded'] and mode == Mode.FINE_POINTING:
        qe = quat_error(sc.q, mekf.q)
        if qe[0] < 0: qe = -qe
        err = float(np.degrees(2.0 * np.linalg.norm(qe[1:])))
        S['adcs_err_deg'] = err
        S['adcs_stable_cnt'] = S['adcs_stable_cnt'] + 1 if err < ADCS_STABLE_DEG else 0
        if S['adcs_stable_cnt'] >= ADCS_STABLE_SUST:
            S['phase1_confirmed'] = True
            S['phase1_conf_t']    = S['t']
            print(f"  Phase 1 confirmed at t={S['t']:.1f}s")
    elif mode != Mode.FINE_POINTING:
        S['adcs_stable_cnt'] = 0

    with contextlib.redirect_stdout(io.StringIO()):
        if mode == Mode.SAFE_MODE:
            sc.step(np.zeros(3), dist, dt_outer)
        elif mode in (Mode.DETUMBLE, Mode.SUN_ACQUISITION):
            m_cmd, _ = bdot.compute(B_meas, sc.omega, B_I, dt_outer)
            sc.step(mtq.compute_torque(m_cmd, B_meas), dist, dt_outer)
        elif mode in (Mode.FINE_POINTING, Mode.MOMENTUM_DUMP):
            if S['mekf_seeded'] and S['last_good_q'] is not None:
                qe = quat_error(sc.q, mekf.q)
                if qe[0] < 0: qe = -qe
                if np.degrees(2 * np.linalg.norm(qe[1:])) > 25.0:
                    nadir_I = QUEST.nadir_inertial(pos_km)
                    nadir_b = QUEST.nadir_body_from_earth_sensor(pos_km, sc.q)
                    qf, _ = quest_alg.compute_multi(
                        vectors_body=[B_meas, sun_meas, nadir_b],
                        vectors_inertial=[B_I, sun_I, nadir_I],
                        weights=[0.70, 0.20, 0.10])
                    if qf[0] < 0: qf = -qf
                    mekf.q = qf.copy()
            for _ in range(N_INNER):
                oi = gyro.measure(sc.omega)
                mekf.predict(oi)
                mekf.update_vector(B_meas, B_I, mekf.R_mag)
                mekf.update_vector(sun_meas, sun_I, mekf.R_sun)
                omega_est = sc.omega - mekf.bias
                if mode == Mode.MOMENTUM_DUMP:
                    torque_cmd = np.zeros(3)
                    rw.apply_torque(torque_cmd, dt_inner)
                    m_cmd = np.clip(mtq.compute_dipole(rw.h, B_meas)*5.,
                                    -mtq.m_max, mtq.m_max)
                else:
                    torque_cmd, _ = ctrl.compute(mekf.q, omega_est, S['q_ref'])
                    rw.apply_torque(torque_cmd, dt_inner)
                    m_cmd = mtq.compute_dipole(rw.h, B_meas)
                torque_mtq = mtq.compute_torque(m_cmd, B_meas)
                sc.step(torque_mtq, dist, dt_inner,
                        tau_rw=torque_cmd, h_rw=rw.h.copy())
            if S['mekf_seeded'] and mode == Mode.FINE_POINTING:
                qe = quat_error(sc.q, mekf.q)
                if qe[0] < 0: qe = -qe
                S['adcs_err_deg'] = np.degrees(2 * np.linalg.norm(qe[1:]))

    # Deputy open-loop Phase 1
    if not S['phase2_active']:
        cw.step(dt_outer, np.zeros(3))

    # Phase 2 activation
    if (S['phase1_confirmed'] and not S['phase2_active']
            and S['t'] >= S['phase1_conf_t'] + RDV_DELAY_S):
        S['phase2_active'] = True
        np.random.seed(42)
        cw_ekf.initialise(
            cw.state + np.array([5.,5.,5.,0.02,0.02,0.02])*np.random.randn(6))
        print(f"  Phase 2 active at t={S['t']:.1f}s  range={cw.range_m:.1f}m")

    if S['phase2_active']:
        if (not S['rdv_triggered'] and mode == Mode.FINE_POINTING
                and S['t'] >= S['phase1_conf_t'] + RDV_DELAY_S + 120.0):
            rel_ctrl.set_mode(RelNavMode.RENDEZVOUS, t=S['t'])
            S['rdv_triggered'] = True
            print(f"  Rendezvous triggered at t={S['t']:.1f}s")

        accel_cmd, impulse_dv = rel_ctrl.compute(cw_ekf.x, S['t'])
        if impulse_dv is not None:
            TEL['burns'].append((S['t'], float(cw.state[0]),
                                 float(cw.state[1]), float(cw.state[2])))
            cw.apply_impulse(impulse_dv)
            cw_ekf.x[3:6] += impulse_dv
            S['total_dv_ms'] += float(np.linalg.norm(impulse_dv))
            print(f"  dv [{S['t']:.0f}s] [{impulse_dv[0]:.4f},"
                  f"{impulse_dv[1]:.4f},{impulse_dv[2]:.4f}] m/s")

        true_cw = cw.step(dt_outer, accel_cmd)
        cw_ekf.predict(accel_cmd)
        z, R = rng_sensor.measure(true_cw[:3], np.array([0.,1.,0.]))
        if z is not None:
            cw_ekf.update(z, R)

    # ECI positions
    pos_m  = pos_km * 1e3
    vel_m  = vel_km * 1e3
    dr_m   = cw.state[:3]
    dep_m  = CWDynamics.lvlh_to_eci(dr_m, pos_m, vel_m)
    dep_km = dep_m * 1e-3
    dr_km  = dr_m * 1e-3   # LVLH relative [km] — for right panel

    S['chief_pos_km']  = pos_km
    S['deputy_pos_km'] = dep_km
    S['dr_km']         = dr_km

    # Left panel trails (ECI)
    S['chief_trail_eci'].append(pos_km.copy())
    S['deputy_trail_eci'].append(dep_km.copy())
    if len(S['chief_trail_eci'])  > TRAIL_LEN: S['chief_trail_eci'].pop(0)
    if len(S['deputy_trail_eci']) > TRAIL_LEN: S['deputy_trail_eci'].pop(0)

    # Right panel trails (LVLH, relative to chief)
    # Chief is always [0,0,0]; deputy is dr_km
    S['chief_trail_rel'].append(np.zeros(3))
    S['deputy_trail_rel'].append(dr_km.copy())
    if len(S['chief_trail_rel'])  > TRAIL_LEN: S['chief_trail_rel'].pop(0)
    if len(S['deputy_trail_rel']) > TRAIL_LEN: S['deputy_trail_rel'].pop(0)

    # Telemetry
    TEL['t'].append(S['t'])
    TEL['dx'].append(dr_m[0]);  TEL['dy'].append(dr_m[1])
    TEL['dz'].append(dr_m[2])
    TEL['sep'].append(float(np.linalg.norm(dr_m)))
    TEL['chief_eci'].append(pos_km.copy())
    TEL['deputy_eci'].append(dep_km.copy())

    S['t'] += dt_outer


# ─────────────────────────────────────────────────────────────────────
#  Build Figure 1 — dual-panel live animation
# ─────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig1 = plt.figure(figsize=(18, 9), facecolor='#040810')
fig1.suptitle('3U CubeSat Formation — Live Orbital Simulation',
              color='white', fontsize=13, fontweight='bold',
              fontfamily='monospace', y=0.98)

# ── LEFT: global ECI orbit ────────────────────────────────────────────
ax_orb = fig1.add_subplot(1, 2, 1, projection='3d', facecolor='#040810')
ax_orb.set_title('Global Orbit  (ECI, km scale)',
                  color='#aaaaaa', fontsize=9, fontfamily='monospace', pad=4)

u_e = np.linspace(0, 2*np.pi, 40)
v_e = np.linspace(0,   np.pi, 40)
R_E = 6371.0
xe = R_E * np.outer(np.cos(u_e), np.sin(v_e))
ye = R_E * np.outer(np.sin(u_e), np.sin(v_e))
ze = R_E * np.outer(np.ones_like(u_e), np.cos(v_e))
ax_orb.plot_surface(xe, ye, ze, color='#1a4a7c', alpha=0.35,
                    linewidth=0, antialiased=True)

th  = np.linspace(0, 2*np.pi, 300)
inc = np.radians(I_CHIEF_DEG)
rx  = R_CHIEF_KM * np.cos(th)
ry  = R_CHIEF_KM * np.sin(th) * np.cos(inc)
rz  = R_CHIEF_KM * np.sin(th) * np.sin(inc)
ax_orb.plot(rx, ry, rz, '--', color='#ffffff40', lw=0.9)

orb_chief_trail,  = ax_orb.plot([], [], [], '-', color='#FFD700', lw=1.5, alpha=0.9)
orb_deputy_trail, = ax_orb.plot([], [], [], '-', color='#00DCFF', lw=1.0, alpha=0.7)
orb_chief_dot,    = ax_orb.plot([], [], [], 'o', color='#FFD700', ms=10,
                                 markeredgecolor='white', markeredgewidth=1.0)
orb_deputy_dot,   = ax_orb.plot([], [], [], 'o', color='#00DCFF', ms=10,
                                 markeredgecolor='white', markeredgewidth=1.0)

lim_orb = R_CHIEF_KM * 1.25
ax_orb.set_xlim(-lim_orb, lim_orb)
ax_orb.set_ylim(-lim_orb, lim_orb)
ax_orb.set_zlim(-lim_orb, lim_orb)
ax_orb.set_xlabel('X [km]', color='#555', fontsize=7, labelpad=0)
ax_orb.set_ylabel('Y [km]', color='#555', fontsize=7, labelpad=0)
ax_orb.set_zlabel('Z [km]', color='#555', fontsize=7, labelpad=0)
ax_orb.tick_params(colors='#333', labelsize=5)
for pane in [ax_orb.xaxis.pane, ax_orb.yaxis.pane, ax_orb.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#111')
ax_orb.grid(True, color='#111', lw=0.3)
ax_orb.set_box_aspect([1, 1, 1])
ax_orb.view_init(elev=35, azim=120)
ax_orb.legend(handles=[
    Line2D([0],[0], marker='o', color='#FFD700', ms=7, lw=1.2,
           label='Chief', markeredgecolor='white'),
    Line2D([0],[0], marker='o', color='#00DCFF', ms=7, lw=0,
           label='Deputy (offset ~100m)', markeredgecolor='white'),
], loc='upper right', facecolor='#040810cc', edgecolor='#333',
   labelcolor='white', fontsize=8)

# ── RIGHT: tight LVLH close-up  (chief at centre, deputy ~100 m away) ─
ax_rel = fig1.add_subplot(1, 2, 2, projection='3d', facecolor='#040810')
ax_rel.set_title('Close-Up  (LVLH frame — metres scale)',
                  color='#aaaaaa', fontsize=9, fontfamily='monospace', pad=4)

# Chief is a large fixed star at origin
ax_rel.plot([0], [0], [0], '*', color='#FFD700', ms=18,
            markeredgecolor='white', markeredgewidth=1.0, zorder=10,
            label='Chief (fixed origin)')

rel_deputy_trail, = ax_rel.plot([], [], [], '-', color='#00DCFF', lw=1.8, alpha=0.9)
rel_deputy_dot,   = ax_rel.plot([], [], [], 'o', color='#00DCFF', ms=14,
                                 markeredgecolor='white', markeredgewidth=1.5)
rel_sep_line,     = ax_rel.plot([], [], [], '-', color='#FF6B35', lw=2.5, alpha=0.95)
rel_sep_lbl       = ax_rel.text2D(0.02, 0.06, '',
                                   transform=ax_rel.transAxes,
                                   color='#FF6B35', fontsize=11,
                                   fontfamily='monospace', fontweight='bold')

# Axes in km, sized to show 300 m either side of chief
# (auto-rescales in update once range changes significantly)
_T = 0.30   # initial half-width km
ax_rel.set_xlim(-_T, _T); ax_rel.set_ylim(-_T, _T); ax_rel.set_zlim(-_T, _T)
ax_rel.set_xlabel('Radial dx [km]',       color='#666', fontsize=7, labelpad=0)
ax_rel.set_ylabel('Along-track dy [km]',  color='#666', fontsize=7, labelpad=0)
ax_rel.set_zlabel('Cross-track dz [km]',  color='#666', fontsize=7, labelpad=0)
ax_rel.tick_params(colors='#444', labelsize=6)
for pane in [ax_rel.xaxis.pane, ax_rel.yaxis.pane, ax_rel.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#1a1a1a')
ax_rel.grid(True, color='#1a1a1a', lw=0.5)
ax_rel.set_box_aspect([1, 1, 1])
ax_rel.view_init(elev=25, azim=45)
ax_rel.legend(handles=[
    Line2D([0],[0], marker='*', color='#FFD700', ms=10, lw=0,
           label='Chief', markeredgecolor='white'),
    Line2D([0],[0], marker='o', color='#00DCFF', ms=8, lw=1.5,
           label='Deputy', markeredgecolor='white'),
    Line2D([0],[0], color='#FF6B35', lw=2.0, label='Separation'),
], loc='upper right', facecolor='#040810cc', edgecolor='#444',
   labelcolor='white', fontsize=9)

# HUD — centre top
hud = fig1.text(0.50, 0.955, '', color='#00FF88',
                fontsize=9, fontfamily='monospace',
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#00000099',
                          edgecolor='#00FF8844', linewidth=0.8))

MODE_COLOURS = {
    'DETUMBLE':        '#FF4444',
    'SUN_ACQUISITION': '#FF9900',
    'FINE_POINTING':   '#00FF88',
    'MOMENTUM_DUMP':   '#AA44FF',
    'SAFE_MODE':       '#FF0000',
}

# ─────────────────────────────────────────────────────────────────────
#  Animation update
# ─────────────────────────────────────────────────────────────────────
def update(frame):
    if not S['running']:
        return
    for _ in range(SIM_STEPS_PER_FRAME):
        sim_step()
        if not S['running']:
            break
    if len(S['chief_trail_eci']) < 2:
        return

    # ── LEFT panel ────────────────────────────────────────────────────
    ct = np.array(S['chief_trail_eci'])
    dt = np.array(S['deputy_trail_eci'])
    cp = S['chief_pos_km']
    dp = S['deputy_pos_km']

    orb_chief_trail.set_data(ct[:, 0], ct[:, 1])
    orb_chief_trail.set_3d_properties(ct[:, 2])
    orb_deputy_trail.set_data(dt[:, 0], dt[:, 1])
    orb_deputy_trail.set_3d_properties(dt[:, 2])
    orb_chief_dot.set_data([cp[0]], [cp[1]])
    orb_chief_dot.set_3d_properties([cp[2]])
    orb_deputy_dot.set_data([dp[0]], [dp[1]])
    orb_deputy_dot.set_3d_properties([dp[2]])
    ax_orb.view_init(elev=35, azim=120 + S['t'] * 0.003)

    # ── RIGHT panel ───────────────────────────────────────────────────
    drt = np.array(S['deputy_trail_rel'])   # LVLH km trail
    dr  = S['dr_km']                        # current LVLH km

    # Chief is fixed at 0,0,0 — plot deputy trail relative
    # Note: LVLH axes: dx=radial, dy=along-track, dz=cross-track
    # Map to plot as  X=dx, Y=dy, Z=dz
    rel_deputy_trail.set_data(drt[:, 0], drt[:, 1])
    rel_deputy_trail.set_3d_properties(drt[:, 2])
    rel_deputy_dot.set_data([dr[0]], [dr[1]])
    rel_deputy_dot.set_3d_properties([dr[2]])
    # Separation line from chief (0,0,0) to deputy
    rel_sep_line.set_data([0, dr[0]], [0, dr[1]])
    rel_sep_line.set_3d_properties([0, dr[2]])

    range_m = cw.range_m
    rel_sep_lbl.set_text(f'Range: {range_m:.1f} m')

    # Auto-scale right panel around current range
    hw = max(TIGHT_MIN_KM, range_m * 1e-3 * TIGHT_SCALE)
    ax_rel.set_xlim(-hw, hw); ax_rel.set_ylim(-hw, hw); ax_rel.set_zlim(-hw, hw)
    ax_rel.view_init(elev=25, azim=45 + S['t'] * 0.006)

    # ── HUD ───────────────────────────────────────────────────────────
    mc = MODE_COLOURS.get(S['mode_name'], '#FFFFFF')
    p1 = (f"OK t={S['phase1_conf_t']:.0f}s" if S['phase1_confirmed']
          else f"cnt={S['adcs_stable_cnt']}/{ADCS_STABLE_SUST}")
    hud.set_text(
        f"  T+{S['t']/60:5.1f} min  [{S['t']:.0f}/{T_SIM_MAX:.0f}s]  |  "
        f"MODE: {S['mode_name']}  |  "
        f"RANGE: {range_m:.2f} m  |  "
        f"DV: {S['total_dv_ms']*1000:.2f} mm/s  |  "
        f"MEKF: {S['adcs_err_deg']:.2f} deg  |  "
        f"P1: {p1}  |  "
        f"RDV: {'ACTIVE' if S['rdv_triggered'] else 'WAITING'}"
    )
    hud.set_color(mc)


print("Starting live 3D visualisation...")
print("  LEFT  = global ECI orbit   (both dots on the orbit arc)")
print("  RIGHT = close-up LVLH view (chief=star, deputy=cyan, orange=separation)")
print("  Close window to show post-sim spatial figures")
print()

ani = animation.FuncAnimation(
    fig1, update,
    interval=ANIM_INTERVAL,
    blit=False,
    cache_frame_data=False)

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────────────────────────────
#  Post-sim figures
# ─────────────────────────────────────────────────────────────────────
if len(TEL['t']) < 2:
    print("No telemetry — exiting.")
    raise SystemExit

plt.rcParams.update({"font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.35, "lines.linewidth": 1.2})
plt.style.use('default')

t_arr  = np.array(TEL['t'])
dx_arr = np.array(TEL['dx'])
dy_arr = np.array(TEL['dy'])
dz_arr = np.array(TEL['dz'])
sep    = np.array(TEL['sep'])
c_eci  = np.array(TEL['chief_eci'])
d_eci  = np.array(TEL['deputy_eci'])
burns  = TEL['burns']

print(f"\n  Post-sim summary")
print(f"  Total sim time   : {t_arr[-1]:.0f}s  ({t_arr[-1]/60:.1f} min)")
print(f"  Final separation : {sep[-1]:.3f} m")
print(f"  Total DV         : {S['total_dv_ms']*1000:.3f} mm/s")
print(f"  Burns recorded   : {len(burns)}")

# ── Figure 2: LVLH spatial overview ──────────────────────────────────
fig2 = plt.figure(figsize=(18, 11))
fig2.suptitle("Spacecraft Spatial Relationship — LVLH Frame  (post-sim)",
              fontsize=13, fontweight='bold')

# 2a: top-down — dy (along-track) vs dx (radial), chief at origin
ax2a = fig2.add_subplot(2, 3, (1, 2))
sc2a = ax2a.scatter(dy_arr, dx_arr, c=t_arr, cmap='plasma', s=5, zorder=4)
plt.colorbar(sc2a, ax=ax2a, label='Time [s]')
ax2a.plot(0, 0, 'k*', ms=20, zorder=8, label='Chief (LVLH origin)')
ax2a.plot(dy_arr[0],  dx_arr[0],  'go', ms=11, zorder=7,
          label=f'Deputy start  ({dy_arr[0]:.1f}, {dx_arr[0]:.1f}) m')
ax2a.plot(dy_arr[-1], dx_arr[-1], 'rs', ms=11, zorder=7,
          label=f'Deputy end  ({dy_arr[-1]:.1f}, {dx_arr[-1]:.1f}) m')
N_sl = max(1, len(t_arr) // 30)
for i in range(0, len(t_arr), N_sl):
    a = 0.06 + 0.45 * (i / len(t_arr))
    ax2a.plot([0, dy_arr[i]], [0, dx_arr[i]],
              color='slategray', lw=0.7, alpha=a, zorder=1)
ax2a.plot([0, dy_arr[-1]], [0, dx_arr[-1]], 'r--', lw=2.2, alpha=0.9, zorder=6,
          label=f'Final separation = {sep[-1]:.2f} m')
for bt, bdx, bdy, bdz in burns:
    ax2a.plot(bdy, bdx, 'r^', ms=12, zorder=9)
    ax2a.annotate(f'dv\n{bt:.0f}s', xy=(bdy, bdx), xytext=(bdy+3, bdx+3),
                  fontsize=7, color='darkred', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='darkred', lw=0.7))
ax2a.set(xlabel='Along-Track dy [m]', ylabel='Radial dx [m]',
         title='Top-Down LVLH: Chief (star) & Deputy Path — Separation Lines')
ax2a.legend(fontsize=8); ax2a.set_aspect('equal', adjustable='datalim')
ax2a.grid(True, alpha=0.3)

# 2b: separation over time
ax2b = fig2.add_subplot(2, 3, 3)
ax2b.fill_between(t_arr, sep, alpha=0.15, color='purple')
ax2b.plot(t_arr, sep, color='purple', lw=1.8, label='|separation|')
ax2b.axhline(sep[-1], color='red',   ls='--', lw=1.2,
             label=f'Final: {sep[-1]:.2f} m')
ax2b.axhline(sep[0],  color='green', ls=':',  lw=1.2,
             label=f'Initial: {sep[0]:.2f} m')
for bt, bdx, bdy, bdz in burns:
    idx = np.argmin(np.abs(t_arr - bt))
    ax2b.axvline(bt, color='darkred', lw=1.0, alpha=0.7, ls=':')
    ax2b.annotate('dv', xy=(bt, sep[idx]),
                  fontsize=7, color='darkred', fontweight='bold')
ax2b.set(xlabel='Time [s]', ylabel='Distance [m]',
         title='Chief–Deputy Separation Over Time')
ax2b.legend(fontsize=8); ax2b.grid(True, alpha=0.3)

# 2c: side view (dy vs dz)
ax2c = fig2.add_subplot(2, 3, 4)
ax2c.scatter(dy_arr, dz_arr, c=t_arr, cmap='plasma', s=5, zorder=4)
ax2c.plot(0, 0, 'k*', ms=16, zorder=8, label='Chief')
ax2c.plot(dy_arr[0],  dz_arr[0],  'go', ms=9, zorder=7, label='Start')
ax2c.plot(dy_arr[-1], dz_arr[-1], 'rs', ms=9, zorder=7, label='End')
for i in range(0, len(t_arr), max(1, len(t_arr)//30)):
    ax2c.plot([0, dy_arr[i]], [0, dz_arr[i]],
              color='slategray', lw=0.6, alpha=0.15, zorder=1)
ax2c.plot([0, dy_arr[-1]], [0, dz_arr[-1]], 'r--', lw=1.8, alpha=0.8, zorder=6)
for bt, bdx, bdy, bdz in burns:
    ax2c.plot(bdy, bdz, 'r^', ms=10, zorder=9)
ax2c.set(xlabel='Along-Track dy [m]', ylabel='Cross-Track dz [m]',
         title='Side View (Along-Track vs Cross-Track)')
ax2c.legend(fontsize=8); ax2c.set_aspect('equal', adjustable='datalim')
ax2c.grid(True, alpha=0.3)

# 2d: front view (dx vs dz)
ax2d = fig2.add_subplot(2, 3, 5)
ax2d.scatter(dx_arr, dz_arr, c=t_arr, cmap='plasma', s=5, zorder=4)
ax2d.plot(0, 0, 'k*', ms=16, zorder=8, label='Chief')
ax2d.plot(dx_arr[0],  dz_arr[0],  'go', ms=9, zorder=7, label='Start')
ax2d.plot(dx_arr[-1], dz_arr[-1], 'rs', ms=9, zorder=7, label='End')
for i in range(0, len(t_arr), max(1, len(t_arr)//30)):
    ax2d.plot([0, dx_arr[i]], [0, dz_arr[i]],
              color='slategray', lw=0.6, alpha=0.15, zorder=1)
ax2d.plot([0, dx_arr[-1]], [0, dz_arr[-1]], 'r--', lw=1.8, alpha=0.8, zorder=6)
for bt, bdx, bdy, bdz in burns:
    ax2d.plot(bdx, bdz, 'r^', ms=10, zorder=9)
ax2d.set(xlabel='Radial dx [m]', ylabel='Cross-Track dz [m]',
         title='Front View (Radial vs Cross-Track)')
ax2d.legend(fontsize=8); ax2d.set_aspect('equal', adjustable='datalim')
ax2d.grid(True, alpha=0.3)

# 2e: per-axis separation
ax2e = fig2.add_subplot(2, 3, 6)
ax2e.plot(t_arr, np.abs(dx_arr), color='royalblue',  lw=1.3, label='|dx| radial')
ax2e.plot(t_arr, np.abs(dy_arr), color='darkorange', lw=1.3, label='|dy| along-track')
ax2e.plot(t_arr, np.abs(dz_arr), color='green',      lw=1.3, label='|dz| cross-track')
ax2e.plot(t_arr, sep,            color='purple',     lw=2.0, ls='--', label='total |sep|')
for bt, *_ in burns:
    ax2e.axvline(bt, color='darkred', lw=1.0, alpha=0.6, ls=':')
    ax2e.annotate('dv', xy=(bt, max(sep)*0.92),
                  fontsize=7, color='darkred', fontweight='bold')
ax2e.set(xlabel='Time [s]', ylabel='Distance [m]',
         title='Per-Axis Separation Components')
ax2e.legend(fontsize=8); ax2e.grid(True, alpha=0.3)

fig2.tight_layout()

# ── Figure 3: 3D views ────────────────────────────────────────────────
fig3 = plt.figure(figsize=(16, 7))
fig3.suptitle("3D Trajectory Views", fontsize=13, fontweight='bold')

# 3a: 3D LVLH
ax3a = fig3.add_subplot(1, 2, 1, projection='3d')
norm3d = plt.Normalize(t_arr.min(), t_arr.max())
cmap3d = plt.cm.plasma
sk3 = max(1, len(t_arr) // 2000)
for i in range(0, len(t_arr)-1, sk3):
    ax3a.plot(dy_arr[i:i+2], dx_arr[i:i+2], dz_arr[i:i+2],
              color=cmap3d(norm3d(t_arr[i])), lw=1.2, alpha=0.8)
ax3a.scatter([0],[0],[0], color='black', s=150, marker='*', zorder=10,
             label='Chief')
ax3a.scatter([dy_arr[0]], [dx_arr[0]], [dz_arr[0]],
             color='lime', s=80, zorder=9, label='Start')
ax3a.scatter([dy_arr[-1]], [dx_arr[-1]], [dz_arr[-1]],
             color='red', s=80, zorder=9, marker='s',
             label=f'End (sep={sep[-1]:.1f}m)')
ax3a.plot([0, dy_arr[-1]], [0, dx_arr[-1]], [0, dz_arr[-1]],
          'r--', lw=1.8, alpha=0.8, label='Final sep line')
for i in range(0, len(t_arr), max(1, len(t_arr)//20)):
    ax3a.plot([0, dy_arr[i]], [0, dx_arr[i]], [0, dz_arr[i]],
              color='gray', lw=0.5, alpha=0.18)
for bt, bdx, bdy, bdz in burns:
    ax3a.scatter([bdy],[bdx],[bdz], color='darkred', s=80, marker='^', zorder=9)
ax3a.set(xlabel='dy [m]', ylabel='dx [m]', zlabel='dz [m]',
         title='3D LVLH Trajectory\n(Chief at origin, colour = time)')
ax3a.legend(fontsize=8)

# 3b: ECI orbital paths
ax3b = fig3.add_subplot(1, 2, 2, projection='3d')
ax3b.plot_surface(xe, ye, ze, color='steelblue', alpha=0.12, linewidth=0)
ax3b.plot(rx, ry, rz, '--', color='gray', lw=0.8, alpha=0.4)
sk_e = max(1, len(t_arr) // 800)
ax3b.plot(c_eci[::sk_e,0], c_eci[::sk_e,1], c_eci[::sk_e,2],
          color='gold', lw=1.4, alpha=0.9, label='Chief')
ax3b.plot(d_eci[::sk_e,0], d_eci[::sk_e,1], d_eci[::sk_e,2],
          color='cyan', lw=1.4, alpha=0.9, label='Deputy')
ax3b.scatter([c_eci[-1,0]],[c_eci[-1,1]],[c_eci[-1,2]],
             color='gold', s=80, marker='o')
ax3b.scatter([d_eci[-1,0]],[d_eci[-1,1]],[d_eci[-1,2]],
             color='cyan', s=80, marker='o')
ax3b.plot([c_eci[-1,0],d_eci[-1,0]],[c_eci[-1,1],d_eci[-1,1]],
          [c_eci[-1,2],d_eci[-1,2]],
          'r-', lw=2.0, alpha=0.9, label=f'Final sep={sep[-1]:.1f}m')
ax3b.set(xlabel='X [km]', ylabel='Y [km]', zlabel='Z [km]',
         title='ECI Orbits\n(Gold=Chief, Cyan=Deputy)')
ax3b.legend(fontsize=8)
lim3b = R_CHIEF_KM * 1.1
ax3b.set_xlim(-lim3b, lim3b); ax3b.set_ylim(-lim3b, lim3b)
ax3b.set_zlim(-lim3b, lim3b); ax3b.set_box_aspect([1, 1, 1])

fig3.tight_layout()
plt.show()