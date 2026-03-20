"""
ROE-EKF: Extended Kalman Filter for ROE-Based Relative Navigation
==================================================================
Uses J2-perturbed ROE dynamics as the process model and
range + bearing (azimuth/elevation) as the measurement model.

State vector (6 quasi-nonsingular ROE):
    x = [δa, δλ, δe_x, δe_y, δi_x, δi_y]  (dimensionless)

The state transition matrix F = I + Ȧ·dt where Ȧ is the Jacobian of
the ROE secular drift dynamics (analytic — constant for J2).

Measurement model:
    z = h(x) = [range, azimuth, elevation]
    
    h is nonlinear: first convert ROE → LVLH via GVE mapping,
    then compute range/bearing from Cartesian.
    Jacobian H = ∂h/∂x is computed via chain rule:
        H = H_lvlh · ∂(LVLH)/∂(ROE)

Key advantage over CW-EKF:
    The process model captures J2 secular drift analytically.
    Over multiple orbits, CW-EKF accumulates systematic errors
    (position error grows ~proportional to time) while ROE-EKF 
    remains consistent because the model matches reality.

Reference:
    Gaias & D'Amico (2015), JGCD 38(6).
    Montenbruck et al. (2012), JGCD 35(5).
"""

import numpy as np
from environment.roe_dynamics import ROEDynamics


class ROEEKF:
    """
    EKF for relative navigation using ROE process model.

    Parameters
    ----------
    roe_dyn : ROEDynamics instance (provides a, n, drift coefficients)
    dt      : propagation timestep [s]
    """

    def __init__(self, roe_dyn: ROEDynamics, dt: float):
        self.dyn = roe_dyn
        self.dt  = dt
        self.a   = roe_dyn.a
        self.n   = roe_dyn.n

        # State estimate: [δa, δλ, δe_x, δe_y, δi_x, δi_y]
        self.x = np.zeros(6)

        # Mean anomaly tracked independently (same as dyn.mean_anomaly)
        self.mean_anomaly = roe_dyn.mean_anomaly

        # Initial covariance — 50m position uncertainty mapped to ROE
        # ROE position uncertainty ~ σ_pos / a
        sig_pos = 50.0 / self.a
        sig_vel = 0.5  / (self.a * self.n)
        self.P = np.diag([sig_pos**2, sig_pos**2,
                          sig_vel**2, sig_vel**2,
                          sig_pos**2, sig_pos**2])

        # Process noise — accounts for unmodelled forces (drag, SRP, J3...)
        # Tuned to ~1e-8 m/s² unmodelled acceleration
        sigma_u = 1e-8 / (self.a * self.n**2)
        self.Q = np.eye(6) * (sigma_u * dt)**2
        # δa and δλ get slightly more noise (along-track drag uncertainty)
        self.Q[0, 0] *= 10.0
        self.Q[1, 1] *= 10.0

        # Build constant state transition matrix (J2 drift is linear in ROE)
        self._build_stm()

    # ─────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────

    def initialise(self, roe0: np.ndarray, P0: np.ndarray = None):
        """
        Seed filter with initial ROE estimate.

        Parameters
        ----------
        roe0 : initial ROE state [δa, δλ, δe_x, δe_y, δi_x, δi_y]
        P0   : initial covariance (6×6). Default: 50m/0.5m/s diagonal.
        """
        self.x = roe0.copy()
        if P0 is not None:
            self.P = P0.copy()

    # ─────────────────────────────────────────────────────────────────
    # State transition matrix — J2 secular dynamics (constant!)
    # ─────────────────────────────────────────────────────────────────

    def _build_stm(self):
        """
        Analytical state transition matrix for ROE J2 secular dynamics.

        Ȧ = ∂f/∂x  (constant — J2 drift is linear in ROE):
        
        δȧ   = 0
        δλ̇   = -3/2 n δa
        δė_x = +κ δe_y
        δė_y = -κ δe_x
        δi̇_x = 0
        δi̇_y = -RAAN_dot · δi_x
        """
        dt = self.dt
        n  = self.n
        k  = self.dyn._kappa_ei
        kr = self.dyn.RAAN_dot

        # F = I + A*dt  (first-order, valid for small dt)
        # Exact for the linear/secular terms
        F = np.eye(6)

        # δλ coupled to δa
        F[1, 0] = -1.5 * n * dt

        # e-vector rotation (precession)
        # Exact rotation: [[cos, sin], [-sin, cos]] * dt → for small dt:
        ct = np.cos(k * dt)
        st = np.sin(k * dt)
        F[2, 2] =  ct
        F[2, 3] =  st
        F[3, 2] = -st
        F[3, 3] =  ct

        # i-vector RAAN drift
        F[5, 4] = -kr * dt

        self.Phi = F

    # ─────────────────────────────────────────────────────────────────
    # Predict step
    # ─────────────────────────────────────────────────────────────────

    def predict(self, accel_lvlh: np.ndarray = None):
        """
        Propagate ROE state and covariance by one timestep.

        Parameters
        ----------
        accel_lvlh : control acceleration in LVLH [m/s²]. Default: zeros.
                     Applied via GVE mapping to update ROE.
        """
        # Propagate state
        self.x = self.Phi @ self.x

        # Add continuous thrust effect if any (small — rebuild each step if needed)
        if accel_lvlh is not None and np.any(accel_lvlh != 0):
            # Convert continuous acceleration to equivalent impulse over dt
            dv_equiv = accel_lvlh * self.dt
            u = self.mean_anomaly
            su, cu = np.sin(u), np.cos(u)
            inv_na = 1.0 / (self.n * self.a)
            dvr, dvt, dvn = dv_equiv
            d_da   = 2.0 * inv_na * dvt
            d_dlam = -2.0 * inv_na * dvr
            d_dex  = inv_na * (dvr * su + 2.0 * dvt * cu)
            d_dey  = inv_na * (-dvr * cu + 2.0 * dvt * su)
            d_dix  = inv_na * dvn * cu
            d_diy  = inv_na * dvn * su
            self.x += np.array([d_da, d_dlam, d_dex, d_dey, d_dix, d_diy])

        # Propagate covariance
        self.P = self.Phi @ self.P @ self.Phi.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

        # Advance mean anomaly
        self.mean_anomaly = (self.mean_anomaly + self.n * self.dt) % (2*np.pi)

    # ─────────────────────────────────────────────────────────────────
    # Update step — range + bearing measurement
    # ─────────────────────────────────────────────────────────────────

    def update(self,
               z:      np.ndarray,
               R_meas: np.ndarray,
               gate_k: float = 5.0) -> bool:
        """
        EKF measurement update with range + bearing.

        Measurement model: ROE → LVLH (GVE) → [range, az, el]
        Jacobian H = H_bearing · J_roe2lvlh  (chain rule)

        Parameters
        ----------
        z      : [range_m, az_rad, el_rad]
        R_meas : 3×3 measurement noise covariance
        gate_k : Mahalanobis gate sigma. Default: 5σ.

        Returns
        -------
        accepted : True if measurement passed gate
        """
        # Predicted LVLH from current ROE estimate
        lvlh_pred = self._roe_to_lvlh(self.x)
        dr        = lvlh_pred[0:3]
        r_mag     = np.linalg.norm(dr)

        if r_mag < 1.0:
            return False

        # Predicted measurement
        z_pred = self._h(dr)

        # Innovation
        innov    = z - z_pred
        innov[1] = self._wrap(innov[1])
        innov[2] = self._wrap(innov[2])

        # Jacobian H = H_cart · J_roe2lvlh
        H_cart = self._H_bearing(dr)          # 3×3 (w.r.t. position)
        J      = self._jacobian_roe2pos()      # 3×6 (position w.r.t. ROE)
        H      = H_cart @ J                    # 3×6

        # Innovation covariance
        S = H @ self.P @ H.T + R_meas

        # Mahalanobis gate
        try:
            S_inv = np.linalg.inv(S)
            mahal = float(innov @ S_inv @ innov)
        except np.linalg.LinAlgError:
            return False

        if mahal > gate_k**2:
            return False

        # Kalman gain + update
        K  = self.P @ H.T @ S_inv
        self.x = self.x + K @ innov

        # Joseph-form covariance update
        IKH    = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return True

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    def _roe_to_lvlh(self, roe: np.ndarray) -> np.ndarray:
        """Convert ROE state vector to LVLH [m, m/s] using GVE linearisation."""
        a, n = self.a, self.n
        da, dlam, dex, dey, dix, diy = roe
        u = self.mean_anomaly
        su, cu = np.sin(u), np.cos(u)

        dx  = a * (dex * cu + dey * su) - 0.5 * a * da
        dy  = a * dlam + 2.0*a*(-dex * su + dey * cu)
        dz  = a * (dix * su - diy * cu)
        dvx = a * n * (-dex * su + dey * cu)
        dvy = 2.0*a*n*(-dex * cu - dey * su) - 1.5*a*n*da
        dvz = a * n * (dix * cu + diy * su)
        return np.array([dx, dy, dz, dvx, dvy, dvz])

    def _jacobian_roe2pos(self) -> np.ndarray:
        """
        3×6 Jacobian of LVLH position w.r.t. ROE state at current mean anomaly.
        ∂[δx, δy, δz] / ∂[δa, δλ, δe_x, δe_y, δi_x, δi_y]
        """
        a = self.a
        u = self.mean_anomaly
        su, cu = np.sin(u), np.cos(u)

        J = np.zeros((3, 6))
        # δx row: -a/2 δa + a(cu δe_x + su δe_y)
        J[0, 0] = -0.5 * a          # ∂δx/∂δa
        J[0, 2] = a * cu             # ∂δx/∂δe_x
        J[0, 3] = a * su             # ∂δx/∂δe_y
        # δy row: a δλ + 2a(-su δe_x + cu δe_y)
        J[1, 1] = a                  # ∂δy/∂δλ
        J[1, 2] = -2.0 * a * su     # ∂δy/∂δe_x
        J[1, 3] =  2.0 * a * cu     # ∂δy/∂δe_y
        # δz row: a(su δi_x - cu δi_y)
        J[2, 4] = a * su             # ∂δz/∂δi_x
        J[2, 5] = -a * cu            # ∂δz/∂δi_y
        return J

    def _h(self, dr: np.ndarray) -> np.ndarray:
        r  = np.linalg.norm(dr)
        az = np.arctan2(dr[1], dr[0])
        el = np.arctan2(dr[2], np.sqrt(dr[0]**2 + dr[1]**2))
        return np.array([r, az, el])

    def _H_bearing(self, dr: np.ndarray) -> np.ndarray:
        """3×3 Jacobian of [range, az, el] w.r.t. Cartesian position."""
        x, y, z  = dr
        r        = np.linalg.norm(dr)
        r_xy2    = x**2 + y**2
        r_xy     = np.sqrt(r_xy2)
        if r < 1e-6 or r_xy < 1e-6:
            return np.zeros((3, 3))
        dR  = np.array([x/r, y/r, z/r])
        dAz = np.array([-y/r_xy2, x/r_xy2, 0.0])
        dEl = np.array([-x*z/(r**2*r_xy), -y*z/(r**2*r_xy), r_xy/r**2])
        return np.array([dR, dAz, dEl])

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + np.pi) % (2*np.pi) - np.pi

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        """Estimated LVLH position [m]."""
        return self._roe_to_lvlh(self.x)[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """Estimated LVLH velocity [m/s]."""
        return self._roe_to_lvlh(self.x)[3:6]

    @property
    def position_std(self) -> np.ndarray:
        """Position uncertainty 1-sigma [m], from covariance via Jacobian."""
        J   = self._jacobian_roe2pos()
        P_pos = J @ self.P @ J.T
        return np.sqrt(np.maximum(np.diag(P_pos), 0.0))