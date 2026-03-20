"""
ROE Controller — J2-Compensated Formation Hold + CW Rendezvous Handoff
=======================================================================
Modes
-----
FORMATION_HOLD : Impulsive correction burns when position drift > deadband.
                 Uses GVE pseudo-inverse. J2-aware: no steady-state drift.
RENDEZVOUS     : Delegates to the proven CW two-impulse STM planner.
                 Scans transfer times, picks minimum-fuel solution.
COASTING       : No thrust.

Architecture rationale
----------------------
Formation hold over multi-orbit timescales needs J2-aware ROE dynamics.
Terminal rendezvous (< 1 orbit) is well-served by CW two-impulse planner
since timescales are short enough that J2 error is negligible.
This hybrid approach matches what operational missions (PRISMA, TanDEM-X) do.

Reference
---------
D'Amico & Montenbruck (2006), JGCD 29(3).
Gaias & D'Amico (2015), JGCD 38(6).
Clohessy & Wiltshire (1960) for terminal phase.
"""

import numpy as np
from enum import Enum, auto
from environment.roe_dynamics import ROEDynamics


class ROEMode(Enum):
    FORMATION_HOLD = auto()
    RENDEZVOUS     = auto()
    COASTING       = auto()


class ROEController:
    """
    J2-compensated relative motion controller.

    Parameters
    ----------
    roe_dyn    : ROEDynamics instance (chief orbit parameters)
    mode       : initial ROEMode
    target_roe : target ROE for formation hold [6-vector, dimensionless]
    """

    def __init__(self,
                 roe_dyn:    ROEDynamics,
                 mode:       ROEMode = ROEMode.FORMATION_HOLD,
                 target_roe: np.ndarray = None):

        self.dyn  = roe_dyn
        self.mode = mode
        self.a    = roe_dyn.a
        self.n    = roe_dyn.n
        self.T    = roe_dyn.T

        self.target_roe = np.zeros(6) if target_roe is None else np.array(target_roe)

        # Thruster cap per burn [m/s]
        self.dv_max = 0.5

        # Formation hold: only fire if implied position error > this [m]
        self.pos_deadband_m = 20.0

        # Formation hold: minimum time between burns [s] (avoid chatter)
        self._last_burn_t   = -999.0
        self._burn_interval = self.T * 0.5   # at most once per half-orbit

        # Rendezvous state
        self._rdv_burn1_applied = False
        self._rdv_burn2_applied = False
        self._rdv_t_start       = None
        self._rdv_T             = None
        self._rdv_dv1           = None
        self._rdv_dv2           = None

        self.mode_history = []

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def compute(self,
                roe_est:      np.ndarray,
                mean_anomaly: float,
                t:            float = 0.0
                ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Returns (accel_cmd [m/s²], impulse_dv [m/s] or None).
        Controller is purely impulsive — accel_cmd is always zeros.
        """
        if self.mode == ROEMode.FORMATION_HOLD:
            return np.zeros(3), self._formation_hold(roe_est, mean_anomaly, t)
        elif self.mode == ROEMode.RENDEZVOUS:
            return np.zeros(3), self._rendezvous(roe_est, mean_anomaly, t)
        else:
            return np.zeros(3), None

    def set_mode(self,
                 mode:         ROEMode,
                 roe_est:      np.ndarray = None,
                 mean_anomaly: float      = 0.0,
                 t:            float      = 0.0,
                 t_sim_max:    float      = None,
                 lvlh_est:     np.ndarray = None):
        """Switch mode. For RENDEZVOUS, plans the burn sequence immediately.
        lvlh_est: optional CW-EKF state [m, m/s] — used as fallback LVLH IC
                  if ROE-based conversion gives a poor initial condition.
        """
        print(f"  ROECtrl [{t:.1f}s]: {self.mode.name} → {mode.name}")
        self.mode = mode
        self.mode_history.append((t, mode))

        if mode == ROEMode.RENDEZVOUS:
            self._rdv_burn1_applied = False
            self._rdv_burn2_applied = False
            self._rdv_t_start       = t
            self._rdv_T             = None
            self._rdv_dv1           = None
            self._rdv_dv2           = None
            if roe_est is not None:
                self._plan_rendezvous(roe_est, t, t_sim_max, lvlh_est=lvlh_est)

    # ─────────────────────────────────────────────────────────────────
    # Formation Hold — J2-compensated impulsive correction
    # ─────────────────────────────────────────────────────────────────

    def _formation_hold(self,
                        roe_est:      np.ndarray,
                        mean_anomaly: float,
                        t:            float) -> np.ndarray | None:
        """
        Correct ROE error when it implies position error > deadband.

        Uses the GVE pseudo-inverse to find a minimum-norm corrective burn.
        Burns are naturally J2-compensated because the error is computed
        in ROE space (where J2 drift is explicitly modelled).
        Rate-limiting prevents chatter: max one burn per half-orbit.
        """
        if t - self._last_burn_t < self._burn_interval:
            return None

        roe_err = roe_est - self.target_roe
        pos_err = np.linalg.norm(self._roe_to_pos(roe_err))

        if pos_err < self.pos_deadband_m:
            return None

        a, n = self.a, self.n
        u    = mean_anomaly
        su, cu   = np.sin(u), np.cos(u)
        inv_na   = 1.0 / (n * a)

        # GVE mapping Γ: columns are partial ΔROE per unit [dvr, dvt, dvn]
        # Rows: [δa, δλ, δe_x, δe_y, δi_x, δi_y]
        G = np.zeros((6, 3))
        G[0, 1] =  2.0 * inv_na          # δa  / dvt
        G[1, 0] = -2.0 * inv_na          # δλ  / dvr  (leading-order)
        G[2, 0] =  su  * inv_na          # δex / dvr
        G[2, 1] =  2.0 * cu * inv_na     # δex / dvt
        G[3, 0] = -cu  * inv_na          # δey / dvr
        G[3, 1] =  2.0 * su * inv_na     # δey / dvt
        G[4, 2] =  cu  * inv_na          # δix / dvn
        G[5, 2] =  su  * inv_na          # δiy / dvn

        # Minimum-norm corrective burn: dv = -G+ * roe_err
        try:
            dv = -np.linalg.lstsq(G, roe_err, rcond=None)[0]
        except Exception:
            return None

        dv = np.clip(dv, -self.dv_max, self.dv_max)
        if np.linalg.norm(dv) < 1e-7:
            return None

        self._last_burn_t = t
        return dv

    # ─────────────────────────────────────────────────────────────────
    # Rendezvous — CW STM two-impulse planner (terminal phase)
    # ─────────────────────────────────────────────────────────────────

    def _plan_rendezvous(self,
                         roe_est:   np.ndarray,
                         t:         float,
                         t_sim_max: float = None,
                         lvlh_est:  np.ndarray = None):
        """
        Plan rendezvous using CW two-impulse STM solver.

        Uses lvlh_est (CW-EKF state) as primary LVLH initial condition —
        it has proper position-scale noise (~5m) and is already converged.
        Falls back to ROE→LVLH conversion if lvlh_est not provided.
        """
        # Use CW-EKF LVLH state directly if available — it has better-scaled
        # initial noise than converting through the ROE-EKF state
        if lvlh_est is not None:
            lvlh0 = lvlh_est.copy()
        else:
            lvlh0 = self._roe_to_lvlh_est(roe_est)

        n   = self.n
        T   = self.T
        budget = (t_sim_max - t) if t_sim_max else T

        best_T, best_dv1, best_dv2, best_total = None, None, None, np.inf

        DV_CAP = 0.10   # m/s per burn — reject degenerate BVP solutions

        # Scan transfer times 0.25–0.55 T_orb (23–51min).
        # Shorter coast accumulates less velocity-error-driven position drift,
        # improving burn-2 accuracy at the cost of slightly higher dV.
        # Avoid CW STM singularities at nT = k*pi.
        T_frac_min = 0.25
        T_frac_max = min(0.55, budget / T - 0.05)
        if T_frac_max < T_frac_min:
            T_frac_max = min(0.95, budget / T - 0.05)  # fallback if budget is tight

        for frac in np.linspace(T_frac_min, T_frac_max, 80):
            T_try  = frac * T
            nt_try = n * T_try
            k_near = round(nt_try / np.pi)
            if abs(nt_try - k_near * np.pi) < 0.15:
                continue

            dv1_try, dv2_try = self._cw_two_impulse(lvlh0, T_try)
            if dv1_try is None:
                continue

            if np.linalg.norm(dv1_try) > DV_CAP:
                continue
            if np.linalg.norm(dv2_try) > DV_CAP:
                continue

            total = np.linalg.norm(dv1_try) + np.linalg.norm(dv2_try)
            if total < best_total:
                best_total = total
                best_T, best_dv1, best_dv2 = T_try, dv1_try, dv2_try

        if best_T is None:
            print("  ROECtrl WARNING: No valid rendezvous solution found — reverting to FORMATION_HOLD")
            self.mode = ROEMode.FORMATION_HOLD
            return

        self._rdv_T   = best_T
        self._rdv_dv1 = best_dv1
        self._rdv_dv2 = best_dv2

        print(f"  ROE Rendezvous planned (CW STM): T={best_T/60:.1f} min, "
              f"|Δv1|={np.linalg.norm(best_dv1):.3f} m/s, "
              f"|Δv2|={np.linalg.norm(best_dv2):.3f} m/s, "
              f"ΣΔv={best_total:.3f} m/s")

    def _cw_two_impulse(self,
                        state0:         np.ndarray,
                        transfer_time:  float
                        ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        CW STM two-impulse boundary value problem.
        Solve for dv1 (at t=0) and dv2 (at t=T) such that deputy reaches origin.
        """
        n  = self.n
        T  = transfer_time
        nt = n * T
        s  = np.sin(nt)
        c  = np.cos(nt)

        Phi_rr = np.array([[4-3*c,    0, 0],
                            [6*(s-nt), 1, 0],
                            [0,        0, c]])
        Phi_rv = np.array([[s/n,          2*(1-c)/n,    0  ],
                            [-2*(1-c)/n,  (4*s-3*nt)/n, 0  ],
                            [0,           0,             s/n]])
        Phi_vr = np.array([[3*n*s,     0, 0   ],
                            [6*n*(c-1), 0, 0   ],
                            [0,         0, -n*s]])
        Phi_vv = np.array([[c,    2*s,   0],
                            [-2*s, 4*c-3, 0],
                            [0,    0,     c]])

        r0, v0 = state0[0:3], state0[3:6]

        cond = np.linalg.cond(Phi_rv)
        if cond > 1e8:
            return None, None
        try:
            rhs = Phi_rr @ r0 + Phi_rv @ v0
            dv1 = -np.linalg.solve(Phi_rv, rhs)
        except np.linalg.LinAlgError:
            return None, None

        v0p = v0 + dv1
        v_T = Phi_vr @ r0 + Phi_vv @ v0p
        dv2 = -v_T

        return dv1, dv2

    def _rendezvous(self,
                    roe_est:      np.ndarray,
                    mean_anomaly: float,
                    t:            float) -> np.ndarray | None:
        """Execute the two-impulse sequence."""
        if self._rdv_T is None:
            return None

        dt_rdv = t - self._rdv_t_start

        # Burn 1 — immediate
        if not self._rdv_burn1_applied:
            self._rdv_burn1_applied = True
            return self._rdv_dv1.copy()

        # Burn 2 — at transfer time
        if not self._rdv_burn2_applied and dt_rdv >= self._rdv_T - 0.5:
            self._rdv_burn2_applied = True
            self.mode = ROEMode.COASTING
            return self._rdv_dv2.copy()

        return None

    # ─────────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────────

    def _roe_to_pos(self, roe: np.ndarray) -> np.ndarray:
        """Quick ROE → LVLH position for deadband check."""
        a = self.a
        u = self.dyn.mean_anomaly
        su, cu = np.sin(u), np.cos(u)
        da, dlam, dex, dey, dix, diy = roe
        return np.array([
            a*(dex*cu + dey*su) - 0.5*a*da,
            a*dlam + 2*a*(-dex*su + dey*cu),
            a*(dix*su - diy*cu)
        ])

    def _roe_to_lvlh_est(self, roe: np.ndarray) -> np.ndarray:
        """
        Convert a ROE state vector to full LVLH [pos, vel] using current
        mean anomaly from the dynamics object.
        Used by the rendezvous planner to convert EKF estimate → LVLH IC.
        """
        a, n = self.a, self.n
        u = self.dyn.mean_anomaly
        su, cu = np.sin(u), np.cos(u)
        da, dlam, dex, dey, dix, diy = roe

        dx  = a*(dex*cu + dey*su) - 0.5*a*da
        dy  = a*dlam + 2.0*a*(-dex*su + dey*cu)
        dz  = a*(dix*su - diy*cu)
        dvx = a*n*(-dex*su + dey*cu)
        dvy = 2.0*a*n*(-dex*cu - dey*su) - 1.5*a*n*da
        dvz = a*n*(dix*cu + diy*su)
        return np.array([dx, dy, dz, dvx, dvy, dvz])